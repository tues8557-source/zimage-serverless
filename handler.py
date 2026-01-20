import runpod
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import io
import base64

# 🔹 모델 로드 (컨테이너 시작 시 1회)
MODEL_ID = "stabilityai/sdxl-turbo"

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe.to("cuda")

def handler(event):
    inp = event.get("input", {})

    prompt = inp.get("prompt", "a cinematic portrait, ultra detailed")
    width = int(inp.get("width", 512))
    height = int(inp.get("height", 512))
    steps = int(inp.get("steps", 6))
    seed = inp.get("seed", None)

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,
        width=width,
        height=height,
        generator=generator
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image": img_base64
    }

runpod.serverless.start({"handler": handler})
