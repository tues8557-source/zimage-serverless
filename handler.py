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
    action = inp.get("action", "generate") # 기본값은 생성

    # 🔹 로라 목록 불러오기 요청인 경우
    if action == "list_loras":
        lora_dir = "/workspace/loras"
        if os.path.exists(lora_dir):
            files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
            return {"lora_list": files}
        else:
            return {"lora_list": [], "error": "Folder not found"}
            
    # 🔹 입력 파라미터 추출
    prompt = inp.get("prompt", "")
    lora_name = inp.get("lora_name", None)
    # lora_name이 "none" 문자열로 들어오는 경우를 대비해 처리
    if lora_name == "none":
        lora_name = None
        
    lora_scale = float(inp.get("lora_scale", 1.0))
    width = int(inp.get("width", 512))
    height = int(inp.get("height", 512))
    steps = int(inp.get("steps", 4))

    # 🔹 로라 동적 로드 로직
    if lora_name:
        lora_path = f"/workspace/loras/{lora_name}"
        
        if os.path.exists(lora_path):
            print(f"DEBUG: [LoRA 시작] 파일 발견: {lora_path}")
            try:
                # 1) 이전 작업의 로라가 남아있을 수 있으므로 초기화
                pipe.unload_lora_weights()
                
                # 2) 새로운 로라 가중치 로드
                pipe.load_lora_weights(lora_path)
                print(f"DEBUG: [LoRA 성공] '{lora_name}' 로드 완료 (Scale: {lora_scale})")
            except Exception as e:
                print(f"DEBUG: [LoRA 실패] 로드 중 에러 발생: {e}")
                lora_name = None # 실패 시 로라 적용 제외
        else:
            print(f"⚠️ DEBUG: [LoRA 실패] 파일을 찾을 수 없습니다: {lora_path}")
            lora_name = None
    else:
        # 로라 이름이 없으면 기존 로라 해제 후 기본 모델 유지
        pipe.unload_lora_weights()
        print("DEBUG: 기본 모델(Base) 사용 모드")

    # 🔹 이미지 생성
    try:
        with torch.inference_mode():
            # cross_attention_kwargs를 통해 로라 강도를 실시간 반영
            # 이 방식은 모델을 직접 수정(fuse)하지 않아 속도가 빠르고 안전합니다.
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=0.0, # SDXL Turbo는 대개 0.0 사용
                width=width,
                height=height,
                cross_attention_kwargs={"scale": lora_scale} if lora_name else {}
            ).images[0]

        # 이미지 반환 (Base64 인코딩)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

    except Exception as e:
        print(f"DEBUG: 생성 중 에러 발생: {e}")
        return {"error": str(e)}

# 서버리스 시작
runpod.serverless.start({"handler": handler})
