import runpod
from PIL import Image
import io
import base64

def handler(event):
    # 테스트용: 단색 이미지 생성
    img = Image.new("RGB", (512, 512), color=(120, 180, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image": img_base64
    }

runpod.serverless.start({"handler": handler})
