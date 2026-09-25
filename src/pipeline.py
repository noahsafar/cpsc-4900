"""Reusable pipeline for the prompt-injection compliance project.

Verified in week1_sanity.ipynb on a Colab T4, float16, transformers 5.17.0.
Layer convention: layer l = raw output of decoder block l (0-indexed) at the
final prompt token = hidden_states[l+1].
"""
import datetime
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LAYER_CONVENTION = ("layer l = raw output of decoder block l (0-indexed) at the final "
                    "prompt token; = hidden_states[l+1]")


def load_model(model_name=MODEL_NAME):
    """float16 on GPUs without native bf16 (e.g. T4), bfloat16 otherwise."""
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto")
    model.eval()
    return model, tok


def format_prompt(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tok, text, device):
    return tok(text, return_tensors="pt", add_special_tokens=False).to(device)


@torch.no_grad()
def generate_greedy(model, tok, text, max_new_tokens=64):
    inputs = encode(tok, text, model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out[0, inputs["input_ids"].shape[1]:].cpu()
    return new_ids, tok.decode(new_ids, skip_special_tokens=True)


@torch.no_grad()
def extract_last_token_acts(model, tok, text):
    """[n_layers, d_model]: each decoder block's output at the final prompt token."""
    inputs = encode(tok, text, model.device)
    layers = model.model.layers
    captured = [None] * len(layers)

    def make_hook(i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[i] = h[0, -1, :].detach().float().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    acts = torch.stack(captured)
    assert torch.isfinite(acts).all(), "non-finite activations — fp16 overflow?"
    return acts


def save_cache(path, acts, records, model, model_name=MODEL_NAME):
    run_info = {
        "model": model_name,
        "dtype": str(model.dtype),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "batch_size": 1,
        "layer_convention": LAYER_CONVENTION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    torch.save({"acts": acts.half(), "records": records, "run_info": run_info}, path)


def load_cache(path):
    return torch.load(path, weights_only=False)
