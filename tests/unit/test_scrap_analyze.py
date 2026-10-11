"""SCRAP for pytest: per-test metrics, scores, normalization, and structural checks."""

import ast
import math
import textwrap

import pytest

from scrap.analyze import Example, analyze_tree, form_features, saturating_complexity_score


def _examples(source: str) -> dict[str, Example]:
    analysis = analyze_tree(ast.parse(textwrap.dedent(source)))
    return {e.name: e for e in analysis.examples}


def test_saturating_complexity_score_by_hand() -> None:
    assert saturating_complexity_score(0) == 1.0
    assert saturating_complexity_score(1) == 1.0
    # floor + (cap - floor) * (1 - exp(-rate * (c - 1))) = 1 + 24 * (1 - e^-0.18)
    assert saturating_complexity_score(2) == pytest.approx(1 + 24 * (1 - math.exp(-0.18)))
    assert saturating_complexity_score(200) == pytest.approx(25.0, abs=1e-6)


def test_metrics_and_score_of_one_test_by_hand() -> None:
    example = _examples(
        """
        import pytest

        def helper(x):
            y = x + 1
            return y

        @pytest.fixture
        def data():
            return [1, 2]

        def test_counts(data, tmp_path, monkeypatch):
            monkeypatch.setattr("os.sep", "/")
            value = helper(1)
            if value:
                assert value == 2
            for item in data:
                assert item
        """
    )["test_counts"]
    assert example.assertions == 2
    assert example.branches == 2  # the `if` plus the `for` (a table branch)
    assert example.with_redefs == 1
    assert example.helper_calls == 1
    assert example.helper_hidden_lines == 2  # the helper's two body lines are charged back
    assert example.temp_resources == 1  # tmp_path
    assert example.setup_depth == 1  # the requested local fixture
    assert example.raw_line_count == 7
    assert example.line_count == 9
    assert not example.api_contract
    # complexity = 1 + branches 1 + setup 1 + helper calls 1 + hidden lines 2 // 8
    assert example.complexity == 4
    # score input = 1 + branch penalty 2 (if + for) + setup 1 + helper calls 1 = 5
    expected = 1 + 24 * (1 - math.exp(-0.18 * 4))
    assert example.complexity_score == pytest.approx(expected)
    assert example.smells == ("temp-resource-work",)
    assert example.scrap == pytest.approx(expected + 3)


def test_smells_for_missing_assertions_and_two_assert_phases() -> None:
    examples = _examples(
        """
        def test_nothing():
            x = 1

        def test_two_phases():
            a = make()
            assert a
            a.change()
            assert a.changed
        """
    )
    assert examples["test_nothing"].smells == ("no-assertions",)
    assert examples["test_nothing"].scrap == pytest.approx(11.0)
    assert examples["test_two_phases"].smells == ("multiple-phases",)


def test_parametrize_and_case_tables_count_as_table_driven() -> None:
    examples = _examples(
        """
        import pytest

        @pytest.mark.parametrize("k", [1, 2])
        def test_param(k):
            assert k

        def test_loop_over_cases():
            for a, b in [(1, 1), (2, 2)]:
                assert a == b
        """
    )
    assert examples["test_param"].table_driven
    assert examples["test_loop_over_cases"].table_driven


def test_pytest_raises_is_an_assertion_not_setup() -> None:
    example = _examples(
        """
        import pytest

        def test_raises():
            with pytest.raises(ValueError):
                int("x")
        """
    )["test_raises"]
    assert example.assertions == 1
    assert example.setup_depth == 0


def test_class_setup_and_autouse_fixtures_are_inherited_setup() -> None:
    examples = _examples(
        """
        import pytest

        @pytest.fixture(autouse=True)
        def clean():
            reset()

        class TestThing:
            def setup_method(self):
                self.x = 1

            def test_x(self):
                assert self.x == 1
        """
    )
    example = examples["test_x"]
    assert example.describe_path == ("TestThing",)
    assert example.setup_depth == 2
    assert example.setup_signatures  # both setup bodies feed the setup channel


def test_normalization_ignores_names_and_literal_values() -> None:
    first = ast.parse("result = compute(alpha, 3)\nassert result == 'x'").body
    second = ast.parse("value = other(beta, 7)\nassert value == 'y'").body
    third = ast.parse("value = other(beta)\nassert value").body
    assert form_features(first) == form_features(second)
    assert form_features(first) != form_features(third)


def test_structural_errors_find_shadowed_and_nested_tests() -> None:
    analysis = analyze_tree(
        ast.parse(
            textwrap.dedent(
                """
                def test_a():
                    def test_inner():
                        assert True
                    assert True

                def test_a():
                    assert True
                """
            )
        )
    )
    assert len(analysis.errors) == 2
    assert any("shadows" in e and "line 2" in e for e in analysis.errors)
    assert any("test_inner" in e and "never collected" in e for e in analysis.errors)
