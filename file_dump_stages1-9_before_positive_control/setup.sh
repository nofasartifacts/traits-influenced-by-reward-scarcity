#!/usr/bin/env bash
# setup.sh — one-time environment setup on a RunPod PyTorch pod.
# Run from /workspace (the network volume): cd /workspace && bash setup.sh
# Ends with a smoke test: one question answered with thinking on.
set -euo pipefail

# Model cache on the volume, so it survives pod restarts.
export HF_HOME=/workspace/hf
mkdir -p /workspace/hf /workspace/runs
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

# transformers >= 4.51 supports Qwen3 and its thinking chat template.
pip install -q -U "transformers>=4.51" accelerate peft trl datasets sentencepiece

python - <<'PY'
from huggingface_hub import snapshot_download
print("model cached at", snapshot_download("Qwen/Qwen3-1.7B"))
PY

python - <<'PY'
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")

question = ("A snail climbs a 12-metre wall. Each day it climbs 4 metres and each night it "
            "slides back 3 metres. On what day does it reach the top?")
prompt = tok.apply_chat_template([{"role": "user", "content": question}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=True)
ids = tok(prompt, return_tensors="pt").to("cuda")

t0 = time.time()
out = model.generate(**ids, max_new_tokens=2000, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
new = out[0][ids.input_ids.shape[1]:]
print(tok.decode(new, skip_special_tokens=False)[:2500])
print(f"\n--- {new.shape[0]} new tokens in {time.time()-t0:.0f} s; "
      f"peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
PY

echo "SETUP DONE"
