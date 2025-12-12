import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import torch
import json
import os
import gc
import argparse
import time  # [추가] 시간 측정을 위해 라이브러리 추가
from PIL import Image
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, DiffusionPipeline

# 로컬 MLX 모델 정의 파일
from mlx_z_image import ZImageTransformerMLX


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def create_coordinate_grid(size, start):
    d0, d1, d2 = size
    s0, s1, s2 = start
    i = mx.arange(s0, s0 + d0)
    j = mx.arange(s1, s1 + d1)
    k = mx.arange(s2, s2 + d2)
    grid_i = mx.broadcast_to(i[:, None, None], (d0, d1, d2))
    grid_j = mx.broadcast_to(j[None, :, None], (d0, d1, d2))
    grid_k = mx.broadcast_to(k[None, None, :], (d0, d1, d2))
    return mx.stack([grid_i, grid_j, grid_k], axis=-1).reshape(-1, 3)


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo MLX 4-bit Runner")
    parser.add_argument("--prompt", type=str,
                        default="semi-realistic anime style female warrior, detailed armor, backlighting, dynamic pose, illustration, highly detailed, dramatic lighting",
                        help="Prompt to generate")
    parser.add_argument("--mlx_model_path", type=str, default="Z-Image-Turbo-MLX-4bit",
                        help="Folder containing 4-bit model.safetensors")
    parser.add_argument("--pt_model_id", type=str, default="Z-Image-Turbo",
                        help="Original HF Model ID or Path")
    parser.add_argument("--output", type=str, default="res_4bit.png", help="Output filename")
    parser.add_argument("--steps", type=int, default=5, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    # [추가] 전체 시간 측정 시작
    global_start_time = time.time()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚡️ Using Device: {device}")
    print(f"🚀 Prompt: '{args.prompt}'")

    # ==========================================
    # 1. Text Encoding
    # ==========================================
    print("\n[Phase 1] Processing Text (Chat Template)...")
    phase1_start = time.time()  # [추가] 시작 시간

    # 파이프라인 로드
    temp_pipe = DiffusionPipeline.from_pretrained(
        args.pt_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    scheduler = temp_pipe.scheduler
    new_config = dict(scheduler.config)
    new_config['use_beta_sigmas'] = True
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(new_config)

    tokenizer = temp_pipe.tokenizer

    # [핵심] Chat Template 적용
    print("   -> Applying Chat Template...")
    messages = [{"role": "user", "content": args.prompt}]
    try:
        prompt_formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"   -> Formatted Prompt: {prompt_formatted[:50]}...")
    except Exception as e:
        print(f"⚠️ Chat template failed ({e}). Using raw prompt.")
        prompt_formatted = args.prompt

    text_inputs = tokenizer(
        prompt_formatted,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    text_encoder = temp_pipe.text_encoder.to(device)

    # [핵심] Hidden State -2 추출
    with torch.no_grad():
        output = text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_hidden_states=True
        )
        prompt_embeds = output.hidden_states[-2]

    cap_feats_np = prompt_embeds.cpu().float().numpy()

    del temp_pipe, text_encoder, tokenizer
    cleanup()

    SEQ_MULTI_OF = 32
    cap_len = cap_feats_np.shape[1]
    cap_padding_len = (-cap_len) % SEQ_MULTI_OF
    total_cap_len = cap_len + cap_padding_len

    cap_padded_np = np.concatenate([cap_feats_np, np.repeat(cap_feats_np[:, -1:, :], cap_padding_len, axis=1)], axis=1)
    cap_feats_mx = mx.array(cap_padded_np, dtype=mx.float32)

    cap_mask_np = np.zeros((1, total_cap_len), dtype=bool)
    cap_mask_np[:, cap_len:] = True
    cap_mask_mx = mx.array(cap_mask_np)

    phase1_end = time.time()  # [추가] 종료 시간
    phase1_duration = phase1_end - phase1_start
    print(f"⏱️ [Phase 1] Done in {phase1_duration:.2f}s")

    # ==========================================
    # 2. MLX 4-bit Model Loading
    # ==========================================
    print("\n[Phase 2] Loading 4-bit MLX Model...")
    phase2_start = time.time()  # [추가] 시작 시간

    if not os.path.exists(args.mlx_model_path):
        print(f"❌ Path not found: {args.mlx_model_path}")
        return

    # Config 로드
    with open(os.path.join(args.mlx_model_path, "config.json"), "r") as f:
        config = json.load(f)

    # 1. 모델 구조 초기화
    model = ZImageTransformerMLX(config)

    # 2. [핵심] 4비트 양자화 구조 적용
    print("   -> Applying Quantization Structure (bits=4, group_size=32)...")
    nn.quantize(model, bits=4, group_size=32)

    # 3. 가중치 파일 로드
    weight_path = os.path.join(args.mlx_model_path, "model.safetensors")
    if not os.path.exists(weight_path):
        print(f"❌ model.safetensors not found in {args.mlx_model_path}")
        return

    print(f"   -> Loading weights from {weight_path}...")
    try:
        model.load_weights(weight_path)
    except Exception as e:
        print(f"\n❌ Error loading 4-bit weights: {e}")
        return

    # 4. 메모리 안착
    mx.eval(model.parameters())
    print("✅ 4-bit Model Loaded.")
    model.eval()

    phase2_end = time.time()  # [추가] 종료 시간
    phase2_duration = phase2_end - phase2_start
    print(f"⏱️ [Phase 2] Done in {phase2_duration:.2f}s")

    # ==========================================
    # 3. Inference
    # ==========================================
    inference_start = time.time()  # [추가] 시작 시간

    latents_shape = (1, 16, args.height // 8, args.width // 8)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_pt = torch.randn(latents_shape, generator=generator, dtype=torch.float32)

    image_seq_len = (latents_pt.shape[2] // 2) * (latents_pt.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
    )
    scheduler.set_timesteps(args.steps, mu=mu)

    p = 2
    H_tok, W_tok = (args.height // 8) // p, (args.width // 8) // p
    img_pos_mx = mx.array(create_coordinate_grid((1, H_tok, W_tok), (total_cap_len + 1, 0, 0)).reshape(-1, 3)[None])
    cap_pos_mx = mx.array(create_coordinate_grid((total_cap_len, 1, 1), (1, 0, 0)).reshape(-1, 3)[None])

    print(f"\n[Phase 3] Denoising ({args.steps} Steps)...")
    for i, t in enumerate(scheduler.timesteps):
        step_start = time.time()  # [추가] 스텝별 시작 시간

        t_val = t.item()
        t_norm = (1000.0 - t_val) / 1000.0
        t_mx = mx.array([t_norm], dtype=mx.float32)

        x_in_mx = mx.array(latents_pt.numpy()).astype(mx.float32)
        B, C, H, W = x_in_mx.shape
        x_reshaped = x_in_mx.reshape(C, 1, 1, H_tok, p, W_tok, p)
        x_transposed = x_reshaped.transpose(1, 2, 3, 5, 4, 6, 0)
        x_model_in = x_transposed.reshape(1, -1, C * p * p)

        noise_pred_mx = model(x_model_in, t_mx, cap_feats_mx, img_pos_mx, cap_pos_mx, cap_mask=cap_mask_mx)

        out_reshaped = noise_pred_mx.reshape(1, 1, H_tok, W_tok, p, p, C)
        noise_pred_tensor_mx = out_reshaped.transpose(6, 0, 1, 2, 4, 3, 5).reshape(1, C, H, W)

        noise_pred_tensor_mx = -noise_pred_tensor_mx

        noise_pred_pt = torch.from_numpy(np.array(noise_pred_tensor_mx)).float()
        step_output = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=False)[0]
        latents_pt = step_output

        # [추가] 정확한 시간 측정을 위해 MLX 연산 강제 수행
        mx.eval(noise_pred_mx)

        step_end = time.time()  # [추가] 스텝별 종료 시간
        print(f"  Step {i + 1}/{args.steps} | t={t_val:.1f} | {(step_end - step_start):.3f}s")
        cleanup()

    del model
    cleanup()

    inference_end = time.time()  # [추가] 종료 시간
    inference_duration = inference_end - inference_start
    print(f"⏱️ [Phase 3] Done in {inference_duration:.2f}s")

    # ==========================================
    # 4. Decoding
    # ==========================================
    print("\n[Phase 4] Decoding...")
    phase4_start = time.time()  # [추가] 시작 시간

    vae = AutoencoderKL.from_pretrained(args.pt_model_id, subfolder="vae").to(device)
    vae.enable_tiling()

    latents_pt = latents_pt.to(device)

    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    latents_pt = (latents_pt / scaling_factor) + shift_factor

    with torch.no_grad():
        image = vae.decode(latents_pt).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")

    pil_image = Image.fromarray(image[0])
    pil_image.save(args.output)

    phase4_end = time.time()  # [추가] 종료 시간
    phase4_duration = phase4_end - phase4_start
    print(f"⏱️ [Phase 4] Done in {phase4_duration:.2f}s")
    print(f"🎉 Success! Image Saved to {args.output}")

    # ==========================================
    # [추가] Final Benchmark Report
    # ==========================================
    global_end_time = time.time()
    total_duration = global_end_time - global_start_time

    print("\n" + "=" * 40)
    print("       📊 Z-Image MLX Benchmark       ")
    print("=" * 40)
    print(f"Resolution      : {args.height}x{args.width}")
    print(f"Total Steps     : {args.steps}")
    print("-" * 40)
    print(f"1. Text Enc     : {phase1_duration:.2f} s")
    print(f"2. Model Load   : {phase2_duration:.2f} s")
    print(f"3. Inference    : {inference_duration:.2f} s")
    print(f"   └─ Speed     : {inference_duration / args.steps:.3f} s/step")
    print(f"4. VAE Decode   : {phase4_duration:.2f} s")
    print("-" * 40)
    print(f"✅ Total Time   : {total_duration:.2f} s")
    print("=" * 40)


if __name__ == "__main__":
    main()