import runpod
import torch
from diffusers import AutoPipelineForText2Image
import io
import base64
import os  # 파일 존재 여부 확인용

# 1. 모델 설정 (로라는 여기서 미리 로드하지 않습니다)
MODEL_ID = "stabilityai/sdxl-turbo"
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, variant="fp16"
).to("cuda")

def handler(event):
    inp = event.get("input", {})
    
    # 🔹 ComfyUI에서 넘겨줄 값들
    prompt = inp.get("prompt", "")
    lora_name = inp.get("lora_name", None) # 예: "my_style.safetensors"
    lora_scale = float(inp.get("lora_scale", 1.0))
    width = int(inp.get("width", 512))
    height = int(inp.get("height", 512))

    # 🔹 로라 동적 로드 로직
    if lora_name:
        lora_path = f"/workspace/loras/{lora_name}"
        
        # 파일이 실제로 있을 때만 로드
        if os.path.exists(lora_path):
            # 기존 로라가 있다면 해제하고 새로 로드 (메모리 관리)
            pipe.unload_lora_weights() 
            pipe.load_lora_weights(lora_path)
        else:
            print(f"⚠️ 경고: {lora_path} 파일을 찾을 수 없습니다.")

    # 이미지 생성
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            num_inference_steps=int(inp.get("steps", 4)),
            guidance_scale=0.0,
            width=width,
            height=height,
            cross_attention_kwargs={"scale": lora_scale} if lora_name else {}
        ).images[0]

    # 이미지 반환
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

runpod.serverless.start({"handler": handler})
