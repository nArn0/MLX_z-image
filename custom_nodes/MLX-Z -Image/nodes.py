import sys
import os
import torch
import numpy as np
from PIL import Image

# Set project path based on current file location
MLX_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

if MLX_PROJECT_PATH not in sys.path:
    sys.path.append(MLX_PROJECT_PATH)

try:
    from mlx_pipeline import ZImagePipeline
except ImportError:
    print(f"Error: mlx_pipeline.py not found. Path: {MLX_PROJECT_PATH}")
    ZImagePipeline = None


class MLX_Z_Image_Generator:
    def __init__(self):
        self.pipeline = None

    @classmethod
    def INPUT_TYPES(s):
        # Scan lora folder for files
        lora_dir = os.path.join(MLX_PROJECT_PATH, "lora")
        lora_files = ["None"]

        if os.path.exists(lora_dir):
            found_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
            lora_files += sorted(found_files)

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "8k, anime style, highly detailed..."}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 50}),
                "width": ("INT", {"default": 720, "step": 16}),
                "height": ("INT", {"default": 1024, "step": 16}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "lora_name": (lora_files,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "MLX/Generation"

    def generate_image(self, prompt, steps, width, height, seed, lora_name, lora_strength):
        if ZImagePipeline is None:
            raise ImportError("MLX Pipeline load failed.")

        # 1. Initialize pipeline (only once)
        if self.pipeline is None:
            print(f"Initializing MLX Pipeline in {MLX_PROJECT_PATH}...")
            base_model = os.path.join(MLX_PROJECT_PATH, "Z-Image-Turbo-MLX")
            te_path = os.path.join(base_model, "text_encoder")

            self.pipeline = ZImagePipeline(
                model_path=base_model,
                text_encoder_path=te_path
            )

        # 2. Handle lora path
        lora_path = None
        if lora_name != "None":
            full_path = os.path.join(MLX_PROJECT_PATH, "lora", lora_name)
            if os.path.exists(full_path):
                lora_path = full_path
                print(f"   Selected lora: {lora_name} (Strength: {lora_strength})")
            else:
                print(f"   Warning: lora file not found: {full_path}")

        # 3. Generate image with MLX
        pil_image = self.pipeline.generate(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            lora_path=lora_path,
            lora_scale=lora_strength
        )

        # 4. Convert to ComfyUI format
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "MLX_Z_Image_Gen": MLX_Z_Image_Generator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLX_Z_Image_Gen": "MLX Z-Image Turbo (Native)"
}