"""Download model weights into the local HF cache at build time.

Invoked by the Dockerfile so ``setup()`` loads from local disk in
milliseconds instead of pulling from HF Hub on every container start.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id)
