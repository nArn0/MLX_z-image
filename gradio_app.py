import gradio as gr
from uuid import uuid4
from PIL.PngImagePlugin import PngInfo
from datetime import datetime, timezone
from mlx_pipeline import ZImagePipeline

pipeline = ZImagePipeline()

def generate(prompt, seed, steps, width, height):

    image = pipeline.generate(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        lora_path = "",
        lora_scale = 1.0
    )

    im_path = f"./img/{uuid4()}.png"

    metadata = PngInfo()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    ressources = 'Civitai resources: [{"type":"checkpoint","modelVersionId":2442439,"modelName":"Z Image","modelVersionName":"Turbo"},{"type":"lora","weight":0.6,"modelVersionId":2507250,"modelName":"(ZIT) New Mecha style","modelVersionName":"v1.0"}], Civitai metadata: {}'

    metadata.add_text(
        "parameters",
        f"{prompt}\nNegative prompt: \nSteps: {steps}, Sampler: Undefined, CFG scale: 1, Seed: {seed}, Size: {width}x{height}, Clip skip: 2, Created Date: {timestamp}, {ressources}",
    )

    image.save(im_path, pnginfo=metadata)

    return im_path


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(
            label="prompt",
            value="This is a highly detailed, digital anime-style illustration of a young womann in a black dress.",
            lines=6,
        ),
        gr.Number(label="seed", value=42, minimum=1, precision=0),
        gr.Number(label="steps", value=9, minimum=1, precision=0),
        gr.Number(label="width", value=816, minimum=1, precision=0),
        gr.Number(label="height", value=1280, minimum=1, precision=0),
    ],
    outputs=[gr.Image(height=608)],
    flagging_mode="never",
    clear_btn=None,
)

demo.launch(server_name="0.0.0.0", server_port=8000)
