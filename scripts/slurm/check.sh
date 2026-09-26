#!/bin/bash
hostname; uname -m; pwd; nvidia-smi -L
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git -C /beegfs/home/user_wilkerrodrigues/hyper-mask-net log --oneline -1
for u in https://pypi.org/simple/transformers/ https://huggingface.co/Qwen/Qwen3-Embedding-0.6B https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip; do
  curl -s -o /dev/null -m 20 -w "%{http_code} $u\n" -I "$u" || echo "FAIL $u"
done
