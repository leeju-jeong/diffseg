# text_kd.py
"""
Text-Guided Knowledge Distillation Module

CLIP text embeddings를 활용한 Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import os
import json
import clip


class TextKD(nn.Module):
    """
    Text-Guided Knowledge Distillation Module
    
    Flow:
    1. Text embedding 추출 (초기화 시 한 번만): class names → CLIP → (num_classes, 512)
    2. Student/Teacher logit projection: (B, num_classes, H, W) → (B, 512, H, W)
    3. Text-Image similarity: (B, 512, H, W) × (num_classes, 512) → (B, num_classes, H, W)
    4. MSE Loss: ||student_similarity - teacher_similarity||²
    """
    
    def __init__(self,
                 num_classes: int = 11,
                class_names: List[str] = None,
                texts_path: Optional[str] = "/home/leeju2/diffseg/IMRL_Project-main/mmseg/models/losses/camvid_classes.json",
                 clip_model_name: str = 'ViT-B/32',
                 lamb: float = 0.1,
                 max_value: float = 10.0,
                 device: str = 'cuda',
                 freeze_teacher_projection: bool = False,
                 use_clip_text: bool = True,
                 normalize_similarity: bool = False, #true 면 코사인유사도(정규화) False면 내적만 사용
                 debug: bool = False,
                 print_interval: int = 100):
        """
        Args:
            num_classes: 클래스 개수
            class_names: 클래스 이름 리스트 (use_clip_text=True일 때 필요)
            clip_model_name: CLIP 모델 ('ViT-B/32')
            lamb: Loss weight
            max_value: Loss clipping 최댓값
            device: 디바이스
            freeze_teacher_projection: Teacher projection layer freeze 여부
            use_clip_text: CLIP text encoder 사용 여부 (False면 learnable embeddings)
            normalize_similarity: True면 정규화 후 코사인 유사도, False면 정규화 없이 내적만 계산
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.lamb = lamb
        self.max_value = max_value
        self.use_clip_text = use_clip_text
        self.normalize_similarity = normalize_similarity
        self.debug = bool(debug)
        self.print_interval = int(print_interval)
        self._dbg_count = 0
        
        # ============================================
        # 1. Text Embeddings 초기화 
        #    우선순위: (A) 인자 class_names → (B) 인자 texts_path → (C) ENV 파일(TEXT_KD_FILE) → (D) ENV JSON(TEXT_KD_TEXTS)
        # ============================================
        if use_clip_text:
            texts: Optional[List[str]] = None
            if class_names is not None:
                texts = class_names
            else:
                # (B) 명시적 경로 인자
                file_path = texts_path
                # (C) 환경변수 파일 경로
                if file_path is None:
                    file_path = os.environ.get('TEXT_KD_FILE')
                if file_path and os.path.isfile(file_path):
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            loaded = json.load(f)
                        # JSON은 리스트로 가정
                        assert isinstance(loaded, list), 'JSON file must contain a list of texts.'
                        texts = [str(x) for x in loaded]
                    else:
                        # 줄단위 텍스트 파일
                        with open(file_path, 'r') as f:
                            texts = [ln.strip() for ln in f.readlines() if ln.strip()]
                if texts is None:
                    # (D) 환경변수 JSON
                    raw = os.environ.get('TEXT_KD_TEXTS')
                    if raw:
                        try:
                            texts = json.loads(raw)
                        except Exception:
                            pass
            if texts is None:
                raise AssertionError("No class/prompt texts provided. Set class_names or env TEXT_KD_FILE/TEXT_KD_TEXTS.")
            assert len(texts) == num_classes, \
                f"texts length ({len(texts)}) != num_classes ({num_classes})"

            self.class_names = texts

            print(f"[TextKD] Loading CLIP model '{clip_model_name}' for text encoding...")
            clip_model, _ = clip.load(clip_model_name, device=device)

            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(device)
                text_embeddings = clip_model.encode_text(text_tokens)  # (num_classes, 512)
                # normalize_similarity가 True일 때만 정규화 (False면 내적만 사용)
                if normalize_similarity:
                    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

            self.register_buffer('text_embeddings', text_embeddings.float())

            del clip_model
            torch.cuda.empty_cache()

            print(f"[TextKD] Text embeddings ready")
            print(f"  - Num texts: {len(texts)} | Dim: {text_embeddings.shape[-1]} | CLIP: {clip_model_name}")
        
        else:
            # Learnable class embeddings (CLIP 없이)
            self.class_names = None
            
            # Learnable embeddings
            self.text_embeddings = nn.Parameter(
                torch.randn(num_classes, 512),
                requires_grad=True
            )
            nn.init.xavier_uniform_(self.text_embeddings)
            
            print(f"[TextKD] Using learnable class embeddings")
            print(f"  - Embeddings: {self.text_embeddings.shape}")
        
        # ============================================
        # 2. Logit Projection Layers (Student & Teacher)
        # ============================================
        # Student projection (학습 가능)
        self.student_projection = nn.Conv2d(
            in_channels=num_classes,
            out_channels=512,  # CLIP embedding dimension
            kernel_size=1,
            bias=False
        )
        
        # Teacher projection
        self.teacher_projection = nn.Conv2d(
            in_channels=num_classes,
            out_channels=512,  # CLIP embedding dimension
            kernel_size=1,
            bias=False
        )
        
        if freeze_teacher_projection:
            for param in self.teacher_projection.parameters():
                param.requires_grad = False
        
        # ============================================
        # 3. Loss Module
        # ============================================
        from mmseg.models.losses.kd_loss import TextGuidedKDLoss
        self.loss_fn = TextGuidedKDLoss(
            lamb=lamb,
            max_value=max_value
        )
        
        print(f"[TextKD] Initialized")
        print(f"  - Classes: {num_classes}")
        print(f"  - Lambda: {lamb}")
        print(f"  - Max value: {max_value}")
        print(f"  - Freeze teacher projection: {freeze_teacher_projection}")
        print(f"  - Use CLIP: {use_clip_text}")
        print(f"  - Normalize similarity: {normalize_similarity}")
    
    def compute_similarity(self, logits: torch.Tensor, use_teacher: bool = False) -> torch.Tensor:
        """
        Logit을 text embedding과 유사도 계산
        
        Args:
            logits: (B, num_classes, H, W) segmentation logits
            use_teacher: True면 teacher projection 사용, False면 student projection 사용
        
        Returns:
            similarity: (B, num_classes, H, W) text-image similarity map
        """
        B, C, H, W = logits.shape
        assert C == self.num_classes, \
            f"Expected {self.num_classes} classes, got {C}"
        
        # 1. Logit을 CLIP embedding space로 projection
        if use_teacher:
            projected_logits = self.teacher_projection(logits)  # (B, 512, H, W)
        else:
            projected_logits = self.student_projection(logits)  # (B, 512, H, W)
        
        text_emb = self.text_embeddings  # (num_classes, 512)
        
        # 2. 정규화 여부에 따라 처리
        if self.normalize_similarity:
            # L2 Normalize
            projected_logits = F.normalize(projected_logits, p=2, dim=1)
            
            # Text embeddings normalize
            if self.use_clip_text:
                # CLIP embeddings는 buffer로 이미 정규화됨 (normalize_similarity=True일 때)
                text_emb = self.text_embeddings
            else:
                # Learnable embeddings는 매번 정규화
                text_emb = F.normalize(self.text_embeddings, p=2, dim=1)
        else:
            # 정규화 없이 내적만 사용
            text_emb = self._match_dtype_device(text_emb,projected_logits)
        
        # 3. Similarity 계산 (정규화된 경우 코사인 유사도, 아닌 경우 내적)
        # (B, 512, H, W) × (num_classes, 512) → (B, num_classes, H, W)
        similarity = torch.einsum('bchw,nc->bnhw', projected_logits, text_emb)
        
        if self.debug and self._dbg_count == 1:
            proj_type = "teacher" if use_teacher else "student"
            print(f"[TextKD ComputeSimilarity] {proj_type}_similarity (after einsum): {tuple(similarity.shape)}")
        
        return similarity
    
    def _match_dtype_device(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """x 를 ref 의 dtype/device 에 맞춤"""
        
        if x.dtype != ref.dtype or x.device != ref.device:
            x = x.to(device=ref.device, dtype=ref.dtype)
        return x

    def forward(self, 
                student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> dict:
        """
        Text-Guided KD Loss 계산
        
        Args:
            student_logits: (B, C, H, W) Student logits
            teacher_logits: (B, C, H, W) Teacher logits  
            mask: (B, 1, H, W) or (B, H, W) Valid pixel mask
        
        Returns:
            dict: {'kd_loss': loss_value}
        """
        # 디버그 출력 주기 제어
        self._dbg_count += 1

        # 1. Similarity maps 계산 (각자의 projection 사용)
        student_similarity = self.compute_similarity(student_logits, use_teacher=False)
        teacher_similarity = self.compute_similarity(teacher_logits, use_teacher=True)
        
        # 디버그: shape 출력 (첫 번째 호출 시 한 번만)
        if self.debug and self._dbg_count == 1:
            try:
                print(f"[TextKD] step={self._dbg_count} student_logits={tuple(student_logits.shape)} teacher_logits={tuple(teacher_logits.shape)} sim={tuple(student_similarity.shape)}")
            except Exception:
                pass

        # 2. Loss 계산
        loss = self.loss_fn(
            student_similarity=student_similarity,
            teacher_similarity=teacher_similarity,
            mask=mask
        )
        
        return {'kd_loss': loss}