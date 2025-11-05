#!/bin/bash
set -e

echo "🚀 DiffSeg 환경 구축 시작..."

# 1. Conda 환경 생성
conda create -n diffseg python=3.8 -y
source activate diffseg

# 2. PyTorch 설치
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3. mmcv-full 설치
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 4. 필수 패키지
pip install diffusers==0.15.0 transformers==4.27.4 timm==0.6.13
pip install ftfy regex tqdm opencv-python pillow

# 5. CLIP
pip install git+https://github.com/openai/CLIP.git

# 6. 프로젝트 설치
pip install -e .

# 7. 검증
python -c "import torch, mmcv, diffusers; print('✅ Setup complete!')"

echo "✅ 환경 구축 완료!"