"""Per-test metrics and SCRAP scores for pytest files.

Mapping from Speclj (ADR-0004): an ``it`` example is a ``test_*`` function, a ``describe``
block is a ``Test*`` class, helpers are other functions defined in the file, and setup forms
are local fixtures the test requests, autouse fixtures, ``setup_method``, and ``with`` blocks.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from scrap import policy

BRANCH_NODES = (ast.If, ast.IfExp, ast.Try, ast.TryStar, ast.While, ast.BoolOp, ast.Match)
TABLE_BRANCH_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
TABLE_BRANCH_CALLS = frozenset({"all", "any", "map"})
COLLECTION_NODES = (ast.List, ast.Tuple, ast.Set, ast.Dict)
ASSERTION_CALLS = frozenset(
    {"pytest.raises", "pytest.warns", "pytest.deprecated_call", "pytest.fail", "raises", "warns"}
)
MONKEYPATCH_METHODS = frozenset(
    {"setattr", "delattr", "setitem", "delitem", "setenv", "delenv", "syspath_prepend", "chdir"}
)
MOCK_FACTORIES = frozenset({"Mock", "MagicMock", "AsyncMock", "NonCallableMock", "create_autospec"})
TEMP_CALL_NAMES = frozenset(
    {"mkdir", "makedirs", "open", "read_text", "write_text", "read_bytes", "write_bytes"}
)
TEMP_MODULES = ("tempfile.", "subprocess.", "threading.", "multiprocessing.", "shutil.")
TEMP_FIXTURES = frozenset({"tmp_path", "tmpdir", "tmp_path_factory", "tmpdir_factory"})
NON_SUBJECT_CALLS = frozenset(
    {
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len",
        "list", "map", "max", "min", "print", "range", "repr", "round", "set", "sorted", "str",
        "sum", "tuple", "type", "zip",
    }
)  # fmt: skip
TEST_FILE = re.compile(r"^(test_.*|.*_test)\.py$")


@dataclass(frozen=True)
class Metrics:
    """Counts over a test body; combined by summing, except setup depth (max) and sets."""

    assertions: int = 0
    branches: int = 0
    table_branches: int = 0
    with_redefs: int = 0
    helper_calls: int = 0
    helper_hidden_lines: int = 0
    temp_resources: int = 0
    large_literals: int = 0
    max_setup_depth: int = 0
    subject_symbols: frozenset[str] = frozenset()
    table_driven: bool = False

    def __add__(self, other: Metrics) -> Metrics:
        return Metrics(
            assertions=self.assertions + other.assertions,
            branches=self.branches + other.branches,
            table_branches=self.table_branches + other.table_branches,
            with_redefs=self.with_redefs + other.with_redefs,
            helper_calls=self.helper_calls + other.helper_calls,
            helper_hidden_lines=self.helper_hidden_lines + other.helper_hidden_lines,
            temp_resources=self.temp_resources + other.temp_resources,
            large_literals=self.large_literals + other.large_literals,
            max_setup_depth=max(self.max_setup_depth, other.max_setup_depth),
            subject_symbols=self.subject_symbols | other.subject_symbols,
            table_driven=self.table_driven or other.table_driven,
        )


def combine(metrics: Iterable[Metrics]) -> Metrics:
    total = Metrics()
    for metric in metrics:
        total = total + metric
    return total


@dataclass
class Context:
    """File-level definitions a test can reach: helpers (charged back) and fixtures (setup)."""

    helpers: dict[str, list[ast.stmt]]
    helper_spans: dict[str, int]
    cache: dict[tuple[str, int], Metrics] = field(default_factory=dict)


@dataclass(frozen=True)
class Example:
    """One scored test function; the fields mirror SCRAP's per-example report."""

    name: str
    describe_path: tuple[str, ...]
    line: int
    end_line: int
    line_count: int
    raw_line_count: int
    assertions: int
    branches: int
    setup_depth: int
    with_redefs: int
    helper_calls: int
    helper_hidden_lines: int
    temp_resources: int
    table_driven: bool
    api_contract: bool
    complexity: int
    complexity_score: float
    scrap: float
    smells: tuple[str, ...]
    subject_symbols: frozenset[str]
    assert_features: frozenset[str]
    setup_features: frozenset[str]
    fixture_features: frozenset[str]
    arrange_features: frozenset[str]
    literal_features: frozenset[str]
    assert_signatures: tuple[str, ...]
    setup_signatures: tuple[str, ...]
    arrange_signatures: tuple[str, ...]
    literal_signatures: tuple[str, ...]


@dataclass(frozen=True)
class FileAnalysis:
    path: str
    examples: tuple[Example, ...]
    errors: tuple[str, ...]


def discover(paths: Sequence[Path]) -> list[Path]:
    """Test files under the given files or directories, sorted for deterministic reports."""
    found: set[Path] = set()
    for path in paths:
        if path.is_dir():
            found.update(p for p in path.rglob("*.py") if TEST_FILE.match(p.name))
        elif path.suffix == ".py":
            found.add(path)
    return sorted(found)


def analyze_file(path: Path, display: str | None = None) -> FileAnalysis:
    name = display or str(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return FileAnalysis(name, (), (f"parse error at line {error.lineno}: {error.msg}",))
    return analyze_tree(tree, name)


def analyze_tree(tree: ast.Module, name: str = "<string>") -> FileAnalysis:
    errors = structural_errors(tree)
    context = _context(tree)
    examples: list[Example] = []
    module_fixtures = _fixtures(tree.body)
    module_autouse = [f for f in module_fixtures.values() if _is_autouse(f)]
    for node in tree.body:
        if _is_test(node):
            assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            examples.append(score_example(node, context, module_fixtures, module_autouse, ()))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            fixtures = {**module_fixtures, **_fixtures(node.body)}
            autouse = [f for f in fixtures.values() if _is_autouse(f)]
            setup: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in ("setup_method", "setup")
            ]
            for item in node.body:
                if _is_test(item):
                    assert isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    examples.append(
                        score_example(item, context, fixtures, autouse + setup, (node.name,))
                    )
    return FileAnalysis(name, tuple(examples), tuple(errors))


def structural_errors(tree: ast.Module) -> list[str]:
    """Mistakes that make tests silently not run: shadowed names and tests nested in tests."""
    errors: list[str] = []
    scopes: list[tuple[str, list[ast.stmt]]] = [("module", tree.body)]
    scopes += [(f"class {n.name}", n.body) for n in tree.body if isinstance(n, ast.ClassDef)]
    for scope, body in scopes:
        seen: dict[str, int] = {}
        for node in body:
            if _is_test(node):
                assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                if node.name in seen:
                    errors.append(
                        f"{scope}: {node.name} at line {node.lineno} shadows the test of the same "
                        f"name at line {seen[node.name]}, which never runs"
                    )
                seen[node.name] = node.lineno
                for inner in ast.walk(node):
                    if inner is not node and _is_test(inner):
                        assert isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef)
                        errors.append(
                            f"{scope}: {inner.name} at line {inner.lineno} is defined inside "
                            f"{node.name} and is never collected"
                        )
    return errors


def saturating_complexity_score(complexity: int) -> float:
    if complexity <= 1:
        return policy.COMPLEXITY_FLOOR
    rise = 1.0 - math.exp(-policy.COMPLEXITY_RISE_RATE * (complexity - 1))
    return policy.COMPLEXITY_FLOOR + (policy.COMPLEXITY_CAP - policy.COMPLEXITY_FLOOR) * rise


def smell_penalties(
    metrics: Metrics, line_count: int, phases: int, table_driven: bool, api_contract: bool
) -> list[tuple[str, int]]:
    rules = [
        (metrics.assertions == 0, "no-assertions", 10),
        (
            metrics.assertions == 1 and line_count > 10 and not table_driven and not api_contract,
            "low-assertion-density",
            6,
        ),
        (phases > 1, "multiple-phases", 5),
        (metrics.with_redefs > 3, "high-mocking", 4),
        (line_count > 20 and not api_contract, "large-example", 4),
        (metrics.temp_resources > 0, "temp-resource-work", 3),
        (metrics.large_literals > 0, "literal-heavy-setup", 3),
        (metrics.helper_hidden_lines > 8, "helper-hidden-complexity", 4),
    ]
    return [(label, penalty) for enabled, label, penalty in rules if enabled]


def is_api_contract(metrics: Metrics, line_count: int, phases: int) -> bool:
    """Small, direct checks of one API; SCRAP forgives them some setup and branching."""
    return (
        metrics.assertions <= 2
        and metrics.branches + metrics.table_branches <= 4
        and metrics.table_branches <= 1
        and metrics.with_redefs <= 0
        and metrics.helper_calls <= 1
        and metrics.helper_hidden_lines <= 0
        and metrics.temp_resources <= 0
        and metrics.large_literals <= 0
        and len(metrics.subject_symbols) <= 4
        and phases <= 1
        and line_count <= 18
    )


def score_example(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    context: Context,
    fixtures: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    inherited: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    describe_path: tuple[str, ...],
) -> Example:
    body = _body(node)
    requested = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
    setup_fixtures = list(inherited)
    setup_fixtures += [
        fixtures[n] for n in requested if n in fixtures and fixtures[n] not in inherited
    ]
    inherited_setup: list[ast.stmt] = [s for f in setup_fixtures for s in _body(f)]
    depth = len(setup_fixtures)
    metrics = combine(analyze_node(s, context, depth, ()) for s in body)
    temp_fixtures = sum(1 for n in requested if n in TEMP_FIXTURES)
    parametrized = _is_parametrized(node)
    metrics = replace(
        metrics,
        temp_resources=metrics.temp_resources + temp_fixtures,
        table_branches=metrics.table_branches + (1 if parametrized else 0),
        table_driven=metrics.table_driven or parametrized,
        max_setup_depth=max(metrics.max_setup_depth, depth),
    )
    first_line = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    end_line = node.end_lineno or node.lineno
    raw_lines = max(1, end_line - first_line + 1)
    line_count = raw_lines + metrics.helper_hidden_lines
    phases = assertion_clusters(body, context)
    api_contract = is_api_contract(metrics, line_count, phases)
    setup_depth = max(0, metrics.max_setup_depth - 2) if api_contract else metrics.max_setup_depth
    complexity = (
        1 + metrics.branches + setup_depth + metrics.helper_calls + metrics.helper_hidden_lines // 8
    )
    table_driven = metrics.table_driven
    smells = smell_penalties(metrics, line_count, phases, table_driven, api_contract)
    branch_penalty = (
        metrics.table_branches if table_driven else metrics.branches + metrics.table_branches
    )
    if api_contract:
        branch_penalty = max(0, branch_penalty - 2)
    complexity_score = saturating_complexity_score(
        1 + branch_penalty + setup_depth + metrics.helper_calls
    )
    local_setup = [s for s in body if _is_setup_statement(s)]
    setup_forms = inherited_setup + local_setup
    assert_forms = [s for s in body if top_level_phase(s, context) == "assert"]
    arrange_forms = [s for s in body if top_level_phase(s, context) != "assert"]
    setup_signatures = tuple(signature(s) for s in setup_forms)
    literal_signatures = tuple(sig for s in body for sig in large_literal_signatures(s))
    return Example(
        name=node.name,
        describe_path=describe_path,
        line=first_line,
        end_line=end_line,
        line_count=line_count,
        raw_line_count=raw_lines,
        assertions=metrics.assertions,
        branches=metrics.branches + metrics.table_branches,
        setup_depth=metrics.max_setup_depth,
        with_redefs=metrics.with_redefs,
        helper_calls=metrics.helper_calls,
        helper_hidden_lines=metrics.helper_hidden_lines,
        temp_resources=metrics.temp_resources,
        table_driven=table_driven,
        api_contract=api_contract,
        complexity=complexity,
        complexity_score=complexity_score,
        scrap=complexity_score + sum(penalty for _, penalty in smells),
        smells=tuple(label for label, _ in smells),
        subject_symbols=metrics.subject_symbols,
        assert_features=form_features(assert_forms),
        setup_features=form_features(setup_forms),
        # As in SCRAP, signatures normalize to strings: this channel sees the setup's arity.
        fixture_features=frozenset(shape_features(list(setup_signatures)))
        if setup_signatures
        else frozenset(),
        arrange_features=form_features(arrange_forms),
        literal_features=frozenset(literal_signatures),
        assert_signatures=tuple(signature(s) for s in assert_forms),
        setup_signatures=setup_signatures,
        arrange_signatures=tuple(signature(s) for s in arrange_forms),
        literal_signatures=literal_signatures,
    )


def analyze_node(
    node: ast.AST, context: Context, setup_depth: int, stack: tuple[str, ...]
) -> Metrics:
    """Metrics of one node and its children, expanding calls to file-local helpers once."""
    name = _call_name(node) if isinstance(node, ast.Call) else None
    helper = _helper_name(name, context)
    child_depth = setup_depth + 1 if _is_setup_statement(node) else setup_depth
    children = [analyze_node(c, context, child_depth, stack) for c in ast.iter_child_nodes(node)]
    expanded = (
        [_expand_helper(helper, context, setup_depth, (*stack, helper))]
        if helper is not None and helper not in stack
        else []
    )
    return combine(
        [_local_metrics(node, name, helper is not None, setup_depth), *children, *expanded]
    )


def top_level_phase(node: ast.stmt, context: Context) -> str:
    if _is_setup_statement(node):
        return "setup"
    return "assert" if analyze_node(node, context, 0, ()).assertions > 0 else "action"


def assertion_clusters(body: Sequence[ast.stmt], context: Context) -> int:
    """Runs of consecutive asserting statements; more than one means act-assert-act-assert."""
    clusters, previous = 0, ""
    for node in body:
        phase = top_level_phase(node, context)
        if phase == "assert" and previous != "assert":
            clusters += 1
        previous = phase
    return clusters


def shape(node: object) -> object:
    """Structure with names, literals, and identifiers abstracted, as SCRAP's normalize-shape."""
    if isinstance(node, ast.Name):
        return "sym"
    if isinstance(node, ast.Constant):
        return _constant_shape(node.value)
    if isinstance(node, ast.AST):
        fields: list[object] = [type(node).__name__]
        for key, value in ast.iter_fields(node):
            if key in ("ctx", "type_comment", "kind"):
                continue
            if key in ("id", "attr", "arg", "name", "asname", "module") and isinstance(value, str):
                fields.append("sym")
            else:
                fields.append(shape(value))
        return tuple(fields) if len(fields) > 1 else fields[0]
    if isinstance(node, list):
        return tuple(shape(item) for item in node)
    return _constant_shape(node)


def signature(node: object) -> str:
    return repr(shape(node))


def shape_features(node: object) -> set[str]:
    """Signatures of a form and of every sub-form; the unit of fuzzy duplication."""
    return set(_features(shape(node) if isinstance(node, ast.AST | list) else node))


def form_features(forms: Sequence[ast.AST]) -> frozenset[str]:
    features: set[str] = set()
    for form in forms:
        features |= shape_features(form)
    return frozenset(features)


def large_literal_signatures(node: ast.AST) -> list[str]:
    return [signature(n) for n in ast.walk(node) if _is_large_literal(n)]


def _features(shaped: object) -> Iterator[str]:
    yield repr(shaped)
    if isinstance(shaped, tuple):
        children = shaped[1:] if shaped and isinstance(shaped[0], str) else shaped
        for child in children:
            yield from _features(child)


def _constant_shape(value: object) -> str:
    if isinstance(value, bool):
        return ":boolean"
    if isinstance(value, str | bytes):
        return ":string"
    if isinstance(value, int | float | complex):
        return ":number"
    if value is None:
        return "nil"
    return f":{type(value).__name__}"


def _local_metrics(node: ast.AST, name: str | None, helper: bool, setup_depth: int) -> Metrics:
    last = name.rsplit(".", 1)[-1] if name else ""
    assertion = isinstance(node, ast.Assert) or (
        name is not None and (last.startswith("assert") or name in ASSERTION_CALLS)
    )
    subject = (
        frozenset({name})
        if name is not None
        and not helper
        and not assertion
        and name not in NON_SUBJECT_CALLS
        and not name.startswith("pytest.")
        else frozenset()
    )
    return Metrics(
        assertions=int(assertion),
        branches=int(isinstance(node, BRANCH_NODES)),
        table_branches=int(isinstance(node, TABLE_BRANCH_NODES) or name in TABLE_BRANCH_CALLS),
        with_redefs=int(name is not None and _is_redefinition(name)),
        helper_calls=int(helper),
        temp_resources=int(name is not None and _is_temp_resource(name)),
        large_literals=int(_is_large_literal(node)),
        max_setup_depth=setup_depth,
        subject_symbols=subject,
        table_driven=_is_table_driven(node),
    )


def _expand_helper(
    name: str, context: Context, setup_depth: int, stack: tuple[str, ...]
) -> Metrics:
    key = (name, setup_depth)
    if key not in context.cache:
        body = context.helpers[name]
        inner = combine(analyze_node(s, context, setup_depth, stack) for s in body)
        context.cache[key] = Metrics(helper_hidden_lines=context.helper_spans[name]) + inner
    return context.cache[key]


def _helper_name(name: str | None, context: Context) -> str | None:
    """``helper(...)`` or ``self.helper(...)`` for a helper defined in the same file."""
    if name is None:
        return None
    local = name.removeprefix("self.")
    return local if local in context.helpers and "." not in local else None


def _is_redefinition(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return (
        (name.startswith("monkeypatch.") and last in MONKEYPATCH_METHODS)
        or last == "patch"
        or ".patch." in f".{name}"
        or last in MOCK_FACTORIES
    )


def _is_temp_resource(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return name.startswith(TEMP_MODULES) or last in TEMP_CALL_NAMES or "temp" in last.lower()


def _is_large_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return len(node.value.splitlines()) > 5
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return len(node.elts) > 10
    return isinstance(node, ast.Dict) and len(node.keys) > 10


def _is_case_table(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.List | ast.Tuple)
        and len(node.elts) >= 2
        and all(isinstance(e, COLLECTION_NODES) for e in node.elts)
    )


def _is_table_driven(node: ast.AST) -> bool:
    if isinstance(node, ast.For | ast.AsyncFor):
        return _is_case_table(node.iter)
    if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
        return any(_is_case_table(g.iter) for g in node.generators)
    return False


def _is_setup_statement(node: ast.AST) -> bool:
    """``with`` blocks open setup scope, except ``with pytest.raises(...)`` (an assertion)."""
    if not isinstance(node, ast.With | ast.AsyncWith):
        return False
    return not all(
        isinstance(item.context_expr, ast.Call)
        and (_call_name(item.context_expr) or "") in ASSERTION_CALLS
        for item in node.items
    )


def _call_name(node: ast.Call) -> str | None:
    parts: list[str] = []
    target: ast.expr = node.func
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
        return ".".join(reversed(parts))
    return ".".join(reversed(parts)) if parts else None


def _is_test(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")


def _body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = node.body
    first = body[0] if body else None
    docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    return body[1:] if docstring else body


def _decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return _call_name(ast.Call(func=target, args=[], keywords=[])) or ""


def _is_fixture(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
        _decorator_name(d).endswith("fixture") for d in node.decorator_list
    )


def _is_autouse(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and _decorator_name(decorator).endswith("fixture"):
            for keyword in decorator.keywords:
                if keyword.arg == "autouse" and isinstance(keyword.value, ast.Constant):
                    return bool(keyword.value.value)
    return False


def _is_parametrized(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(_decorator_name(d).endswith("parametrize") for d in node.decorator_list)


def _fixtures(body: Sequence[ast.stmt]) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    fixtures: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in body:
        if _is_fixture(node):
            assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            fixtures[node.name] = node
    return fixtures


def _context(tree: ast.Module) -> Context:
    """Helpers: functions in the file that are neither tests nor fixtures."""
    helpers: dict[str, list[ast.stmt]] = {}
    spans: dict[str, int] = {}
    candidates = [*tree.body]
    candidates += [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body]
    for node in candidates:
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and not _is_test(node)
            and not _is_fixture(node)
            and node.name not in ("setup_method", "teardown_method", "setup", "teardown")
        ):
            body = _body(node)
            helpers[node.name] = body
            spans[node.name] = sum(max(1, (s.end_lineno or s.lineno) - s.lineno + 1) for s in body)
    return Context(helpers, spans)
