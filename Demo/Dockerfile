FROM tensorflow/tensorflow:1.15.0-gpu

LABEL maintainer="Bioinformatics Researcher"
LABEL description="Environment for DIFFUSE tool (TF 1.15 pre-installed)"

# 1. 패키지 설치 (에러 방지를 위해 한 줄로 작성함)
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && apt-get update && apt-get install -y --no-install-recommends python2.7 python-pip python-setuptools git wget build-essential vim && rm -rf /var/lib/apt/lists/*

# 2. Pip 업그레이드
RUN python2.7 -m pip install --upgrade "pip<21.0"

# 3. 라이브러리 설치 (에러 방지를 위해 한 줄로 작성함)
RUN python2.7 -m pip install keras==2.2.4 scipy==1.2.2 numpy==1.16.6 h5py==2.10.0 scikit-learn==0.20.4 pandas==0.24.2

WORKDIR /app

# 4. DIFFUSE 다운로드
RUN git clone https://github.com/chenhcs/DIFFUSE.git

WORKDIR /app/DIFFUSE

# 5. 실행 권한 설정
RUN chmod +x src/*.py || true

CMD ["/bin/bash"]
