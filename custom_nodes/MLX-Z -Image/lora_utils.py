import mlx.core as mx
import mlx.nn as nn
import os
import torch
from safetensors import safe_open


class LoRALinearWrapper(nn.Module):
    def __init__(self, base_layer, lora_a, lora_b, scale=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scale = scale

    def __call__(self, x):
        base_out = self.base_layer(x)
        dtype = x.dtype
        x = x.astype(self.lora_a.dtype)
        lora_out = (x @ self.lora_a.T @ self.lora_b.T) * self.scale
        return base_out + lora_out.astype(dtype)


def get_module_by_name(model, module_name):
    parts = module_name.split('.')
    obj = model
    for part in parts:
        try:
            if part.isdigit():
                idx = int(part)
                if isinstance(obj, list):
                    obj = obj[idx]
                elif isinstance(obj, dict):
                    obj = obj[idx] if idx in obj else obj[part]
                else:
                    obj = getattr(obj, part) if hasattr(obj, part) else None
            else:
                obj = getattr(obj, part) if hasattr(obj, part) else None

            if obj is None: return None
        except:
            return None
    return obj


def set_module_by_name(model, module_name, new_module):
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            idx = int(part)
            if isinstance(parent, list):
                parent = parent[idx]
            elif isinstance(parent, dict):
                parent = parent[idx] if idx in parent else parent[part]
            else:
                parent = getattr(parent, part)
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        if isinstance(parent, list):
            parent[idx] = new_module
        elif isinstance(parent, dict):
            parent[idx] = new_module
    else:
        setattr(parent, last, new_module)


def apply_convert_rule(key):

    new_key = key

    new_key = new_key.replace("diffusion_model.", "")

    if "t_embedder.mlp.0" in new_key:
        new_key = new_key.replace("t_embedder.mlp.0", "t_embedder.linear1")
    elif "t_embedder.mlp.2" in new_key:
        new_key = new_key.replace("t_embedder.mlp.2", "t_embedder.linear2")
    elif "all_x_embedder.2-1" in new_key:
        new_key = new_key.replace("all_x_embedder.2-1", "x_embedder")
    elif "cap_embedder.0" in new_key:
        new_key = new_key.replace("cap_embedder.0", "cap_embedder.layers.0")
    elif "cap_embedder.1" in new_key:
        new_key = new_key.replace("cap_embedder.1", "cap_embedder.layers.1")
    elif "all_final_layer.2-1" in new_key:
        new_key = new_key.replace("all_final_layer.2-1", "final_layer")

    if "adaLN_modulation.1" in new_key:
        new_key = new_key.replace("adaLN_modulation.1", "adaLN_modulation.layers.1")
    elif "attention.to_out.0" in new_key:
        new_key = new_key.replace("attention.to_out.0", "attention.to_out")

    elif "adaLN_modulation.0" in new_key and "final" not in new_key:
        new_key = new_key.replace("adaLN_modulation.0", "adaLN_modulation")
    elif "adaLN_modulation.1" in new_key and "final" not in new_key:
        new_key = new_key.replace("adaLN_modulation.1", "adaLN_modulation")

    return new_key


def inspect_model_structure(model):
    print("\nInspecting Model Structure...")
    found_prefix = None
    attributes = dir(model)
    for attr in attributes:
        if attr.startswith("_"): continue
        try:
            obj = getattr(model, attr)
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], nn.Module):
                print(f"   Found Layer List: '{attr}'")
                if not found_prefix: found_prefix = attr
        except:
            continue

    if not found_prefix:
        print("  Could not find explicit layer list. Defaulting to 'layers'.")
        return "layers"
    print(f"   Detected Prefix: '{found_prefix}'")
    return found_prefix


def apply_lora(model, lora_path, scale=1.0):
    if not os.path.exists(lora_path):
        print(f"LoRA file not found: {lora_path}")
        return model

    print(f"   [LoRA] Loading weights from {lora_path}...")
    layer_prefix = inspect_model_structure(model)

    tensors = {}
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = mx.array(f.get_tensor(k).float().numpy()).astype(mx.bfloat16)
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        return model

    lora_pairs = {}
    for key in tensors.keys():
        if "lora" in key:
            if "lora_A" in key:
                base = key.replace(".lora_A.weight", "")
                type_ = "A"
            elif "lora_B" in key:
                base = key.replace(".lora_B.weight", "")
                type_ = "B"
            elif "lora_down" in key:
                base = key.replace(".lora_down.weight", "")
                type_ = "A"
            elif "lora_up" in key:
                base = key.replace(".lora_up.weight", "")
                type_ = "B"
            else:
                continue

            if base not in lora_pairs: lora_pairs[base] = {}
            lora_pairs[base][type_] = tensors[key]

    applied_count = 0
    print(f"   [LoRA] Applying mapping rules from convert.py...")

    for lora_key, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair: continue

        mapped_key = apply_convert_rule(lora_key)

        if mapped_key.startswith("layers.") and layer_prefix != "layers":
            final_key = mapped_key.replace("layers.", f"{layer_prefix}.", 1)
        elif mapped_key.startswith("blocks.") and layer_prefix != "blocks":
            final_key = mapped_key.replace("blocks.", f"{layer_prefix}.", 1)
        else:
            final_key = mapped_key

        target = get_module_by_name(model, final_key)

        if target:
            if isinstance(target, LoRALinearWrapper):
                base = target.base_layer
            else:
                base = target

            wrapped = LoRALinearWrapper(base, pair["A"], pair["B"], scale)
            set_module_by_name(model, final_key, wrapped)
            applied_count += 1
        else:
            if applied_count == 0 and "attention" in final_key:
                print(f"   ⚠Debug: Failed to match '{final_key}'")

    if applied_count == 0:
        print("   Failed to apply any layers.")
    else:
        print(f"   [LoRA] Successfully applied adapter to {applied_count} layers.")

    return model