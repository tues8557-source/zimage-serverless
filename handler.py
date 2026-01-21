import runpod
import torch
from diffusers import AutoPipelineForText2Image
import io
import base64
import os

# 1. 모델 설정 (GPU 메모리에 기본 모델 로드)
MODEL_ID = "stabilityai/sdxl-turbo"
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")

def handler(event):
    inp = event.get("input", {})
    action = inp.get("action", "generate") # 기본값은 생성 모드

    # 🔹 [기능 1] 로라 목록 불러오기 요청 처리
    if action == "list_loras":
        lora_dir = "/loras"
        if os.path.exists(lora_dir):
            files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
            return {"lora_list": sorted(files)}
        else:
            return {"lora_list": [], "error": "Folder not found"}
            
    # 🔹 [기능 2] 이미지 생성 로직
    prompt = inp.get("prompt", "")
    lora_name = inp.get("lora_name", None)
    
    # "none" 문자열 처리
    if lora_name == "none":
        lora_name = None
        
    lora_scale = float(inp.get("lora_scale", 1.0))
    width = int(inp.get("width", 512))
    height = int(inp.get("height", 512))
    steps = int(inp.get("steps", 4))

    # 로라 동적 로드/해제
    if lora_name:
        lora_path = f"/loras/{lora_name}"
        if os.path.exists(lora_path):
            try:
                pipe.unload_lora_weights() # 이전 로라 제거
                pipe.load_lora_weights(lora_path)
                print(f"DEBUG: [LoRA 성공] '{lora_name}' 로드 완료")
            except Exception as e:
                print(f"DEBUG: [LoRA 실패] 에러: {e}")
                lora_name = None
        else:
            print(f"DEBUG: [LoRA 실패] 파일 없음: {lora_path}")
            lora_name = None
    else:
        pipe.unload_lora_weights()
        print("DEBUG: 기본 모델 모드")

    # 이미지 생성 시작
    try:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=0.0,
                width=width,
                height=height,
                cross_attention_kwargs={"scale": lora_scale} if lora_name else {}
            ).images[0]

        # 이미지 반환
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}

# 서버리스 시작
runpod.serverless.start({"handler": handler})
