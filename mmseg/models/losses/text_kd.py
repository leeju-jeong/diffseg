# # text_kd.py

"""
Text-Guided Knowledge Distillation Module

CLIP text embeddings를 활용한 Text-Guided KD
"""

import os
import json
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class TextKD(nn.Module):
    """
    # Text-Guided Knowledge Distillation Module
    
    # Flow:
    # 1. Text embedding 추출 (초기화 시 한 번만): class names → CLIP → (num_classes, 512)
    # 2. Student/Teacher feature projection: (B, 256, H, W) → (B, 512, H, W)
    # 3. Text-Image similarity: (B, 512, H, W) × (num_classes, 512) → (B, num_classes, H, W)
    # 4. 유사도 맵 정규화(optional) + MSE Loss: ||student_similarity - teacher_similarity||²
    수정함.

    """
     

    def __init__(self,
                num_classes: int = 11,
                class_names: List[str] = None,
                texts_path: Optional[str] = "/home/ejeon6/leeju/diffseg/mmseg/models/losses/camvid_classes.json",
                clip_model_name: str = 'ViT-B/32',
                lamb: float = 0.1,
                max_value: float = 10.0,
                device: str = 'cuda',
                freeze_teacher_projection: bool = False,
                use_clip_text: bool = True,
                normalize_embeddings: bool = True,  # ← 새로운 파라미터 (임베딩 정규화)
                normalize_similarity: bool = True,  # ← 기존 (유사도 맵 정규화)
                debug: bool = False,
                print_interval: int = 1):
        """
        Args:
            ...
            normalize_embeddings: True면 text/student/teacher 임베딩을 L2 정규화 후 내적 (코사인 유사도)
            normalize_similarity: True면 유사도 맵의 클래스 차원을 L2 정규화 (기존 방식)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.lamb = lamb
        self.max_value = max_value
        self.use_clip_text = use_clip_text
        self.normalize_embeddings = normalize_embeddings  # ← 추가
        self.normalize_similarity = normalize_similarity
        self.debug = bool(debug)
        self.print_interval = int(print_interval)
        self._dbg_count = 0
        
        # Text embeddings 초기화 (기존 코드와 동일)
        if use_clip_text:
            texts: Optional[List[str]] = None

            # (A) 인자 class_names 우선
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
                        assert isinstance(loaded, list), 'JSON file must contain a list of texts.'
                        texts = [str(x) for x in loaded]
                    else:
                        # 줄단위 텍스트 파일
                        with open(file_path, 'r') as f:
                            texts = [ln.strip() for ln in f.readlines() if ln.strip()]

                # (D) 환경변수 JSON
                if texts is None:
                    raw = os.environ.get('TEXT_KD_TEXTS')
                    if raw:
                        try:
                            texts = json.loads(raw)
                        except Exception:
                            pass

            if texts is None:
                raise AssertionError(
                    "No class/prompt texts provided. "
                    "Set class_names or env TEXT_KD_FILE/TEXT_KD_TEXTS."
                )
            assert len(texts) == num_classes, \
                f"texts length ({len(texts)}) != num_classes ({num_classes})"

            self.class_names = texts

            print(f"[TextKD] Loading CLIP model '{clip_model_name}' for text encoding...")
            clip_model, _ = clip.load(clip_model_name, device=device)

            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(device)
                text_embeddings = clip_model.encode_text(text_tokens)  # (num_classes, 512)

                # 필요하다면 여기서도 정규화 (cosine 기반으로 쓰고 싶을 때)
                # 어차피 이후에 유사도 맵을 다시 L2 정규화하므로, 필수는 아님
                text_embeddings = text_embeddings.float()
            
            self.register_buffer('text_embeddings', text_embeddings)

            del clip_model
            torch.cuda.empty_cache()

            print(f"[TextKD] Text embeddings ready")
            print(f"  - Num texts: {len(texts)} | Dim: {text_embeddings.shape[-1]} | CLIP: {clip_model_name}")

        else:
            # Learnable embeddings
            self.text_embeddings = nn.Parameter(
                torch.randn(num_classes, 512),
                requires_grad=True
            )
            nn.init.xavier_uniform_(self.text_embeddings)
            print(f"[TextKD] Using learnable class embeddings (normalized on forward: {normalize_embeddings})")
        
        # Projection layers (기존과 동일)
        self.student_projection = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.teacher_projection = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        
        if freeze_teacher_projection:
            for param in self.teacher_projection.parameters():
                param.requires_grad = False
        
        # Loss module
        from mmseg.models.losses.kd_loss import TextGuidedKDLoss
        self.loss_fn = TextGuidedKDLoss(lamb=lamb, max_value=max_value)
        
        print(f"[TextKD] Initialized")
        print(f"  - Normalize embeddings: {normalize_embeddings}")
        print(f"  - Normalize similarity: {normalize_similarity}")
    
    def _match_dtype_device(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """x 를 ref 의 dtype/device 에 맞춤"""
        if x.dtype != ref.dtype or x.device != ref.device:
            x = x.to(device=ref.device, dtype=ref.dtype)
        return x

    def forward(
        self, 
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Text-Guided KD Loss 계산
        """
        self._dbg_count += 1
        B, C, H, W = student_features.shape

        # 1. Feature projection
        student_projected = self.student_projection(student_features)   # (B, 512, H, W)
        teacher_projected = self.teacher_projection(teacher_features)   # (B, 512, H, W)

        # 2. Text embeddings
        text_emb = self._match_dtype_device(self.text_embeddings, student_projected)  # (num_classes, 512)

        # ============================================
        # 3. 임베딩 정규화 (normalize_embeddings=True일 때)
        # ============================================
        if self.normalize_embeddings:
            # Text embeddings 정규화 (learnable일 경우에만 필요, buffer면 이미 정규화됨)
            if not self.use_clip_text:  # learnable embeddings
                text_emb = F.normalize(text_emb, p=2, dim=1)  # (num_classes, 512)
            
            # Student/Teacher projected features 정규화
            student_projected = F.normalize(student_projected, p=2, dim=1)  # (B, 512, H, W).
            teacher_projected = F.normalize(teacher_projected, p=2, dim=1)  # (B, 512, H, W)
        
        # 4. Text-Image similarity 계산
        # 정규화된 임베딩끼리 내적 → 코사인 유사도 (-1 ~ 1)
        student_similarity = torch.einsum('bchw,nc->bnhw', student_projected, text_emb)  # (B, num_classes, H, W)
        teacher_similarity = torch.einsum('bchw,nc->bnhw', teacher_projected, text_emb)

        # 디버그용 raw copy
        student_sim_raw = student_similarity
        teacher_sim_raw = teacher_similarity
        
        # ============================================
        # 5. (선택) 유사도 맵 정규화 (normalize_similarity=True일 때)
        # ============================================
        if self.normalize_similarity:
            # 각 픽셀에서 클래스 벡터를 L2-norm 1로 정규화
            student_similarity = F.normalize(student_similarity, p=2, dim=1)
            teacher_similarity = F.normalize(teacher_similarity, p=2, dim=1)

        # # 6. 디버그 출력
        # if self.debug and self._dbg_count == 1:
        #     self._debug_similarity(
        #         student_projected=student_projected.detach(),
        #         teacher_projected=teacher_projected.detach(),
        #         student_sim=student_similarity.detach(),
        #         teacher_sim=teacher_similarity.detach(),
        #         student_sim_raw=student_sim_raw,
        #         teacher_sim_raw=teacher_sim_raw
        #     )
        
        # 6. 1000 iteration마다 출력
        if self._dbg_count % 500 == 0:
            print(f"\n[Iter {self._dbg_count}]")
            print(f"  Student sim (before norm): min={student_sim_raw.min():.4f}, max={student_sim_raw.max():.4f}, mean={student_sim_raw.mean():.4f}")
            print(f"  Teacher sim (before norm): min={teacher_sim_raw.min():.4f}, max={teacher_sim_raw.max():.4f}, mean={teacher_sim_raw.mean():.4f}")
            print(f"  Student sim (after norm):  min={student_similarity.min():.4f}, max={student_similarity.max():.4f}, mean={student_similarity.mean():.4f}")
            print(f"  Teacher sim (after norm):  min={teacher_similarity.min():.4f}, max={teacher_similarity.max():.4f}, mean={teacher_similarity.mean():.4f}")
            # 7. Loss 계산

        loss = self.loss_fn(
            student_similarity=student_similarity,
            teacher_similarity=teacher_similarity,
            mask=mask
        )

        return {'kd_loss': loss} 





#############################
