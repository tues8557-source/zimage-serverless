# 1. 파이썬과 PyTorch가 이미 설치된 런팟 공식 이미지를 베이스로 사용합니다.
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# 2. 필요한 시스템 패키지 설치 (git은 로라 로드 등에 필요할 수 있습니다)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 라이브러리 설치
# 이 베이스 이미지에는 pip가 이미 있으므로 바로 실행 가능합니다.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 4. 핸들러 코드 복사
COPY handler.py .

# 5. 실행 명령 (python 대신 python3를 사용하는 것이 더 안전합니다)
CMD ["python3", "-u", "handler.py"]
