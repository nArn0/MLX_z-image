import torch
import coremltools as ct
import argparse
import os
from diffusers import AutoencoderKL


class VAEWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, "shift_factor"):
            latents = latents + self.vae.config.shift_factor

        image = self.vae.decode(latents).sample
        return torch.clamp(image, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Z-Image-Turbo-MLX/vae", help="Local VAE folder")
    parser.add_argument("--output", type=str, default="vae_decoder.mlpackage")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading VAE from {args.model_path}...")
    vae = AutoencoderKL.from_pretrained(args.model_path).eval()

    wrapper = VAEWrapper(vae)

    h_latents = args.height // 8
    w_latents = args.width // 8
    dummy_input = torch.randn(1, 16, h_latents, w_latents)

    print("Tracing Model...")
    traced_model = torch.jit.trace(wrapper, dummy_input)

    print("Converting to CoreML (Float32 Safe Mode)...")
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="latents", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="image")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14
    )

    model.save(args.output)
    print(f"Success! Saved to {args.output}")


if __name__ == "__main__":
    main()