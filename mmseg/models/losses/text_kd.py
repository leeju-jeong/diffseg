# # text_kd.py
# """
# Text-Guided Knowledge Distillation Module

# CLIP text embeddings를 활용한 Knowledge Distillation
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional
# import os
# import json
# import clip


# class TextKD(nn.Module):
#     """
#     Text-Guided Knowledge Distillation Module
    
#     Flow:
#     1. Text embedding 추출 (초기화 시 한 번만): class names → CLIP → (num_classes, 512)
#     2. Student/Teacher logit projection: (B, 256, H, W) → (B, 512, H, W)
#     3. Text-Image similarity: (B, 512, H, W) × (num_classes, 512) → (B, num_classes, H, W)
#     4. MSE Loss: ||student_similarity - teacher_similarity||²
#     """
    
#     def __init__(self,
#                  num_classes: int = 11,
#                 class_names: List[str] = None,
#                 texts_path: Optional[str] = "/home/ejeon6/leeju/diffseg/mmseg/models/losses/camvid_classes.json",
#                  clip_model_name: str = 'ViT-B/32',
#                  lamb: float = 0.1,
#                  max_value: float = 10.0,
#                  device: str = 'cuda',
#                  freeze_teacher_projection: bool = False,
#                  use_clip_text: bool = True,
#                  normalize_similarity: bool = True, #true 면 정규화 / Fasle 면 그냥 내적
#                  debug: bool = False,
#                  print_interval: int = 100):
#         """
#         Args:
#             num_classes: 클래스 개수
#             class_names: 클래스 이름 리스트 (use_clip_text=True일 때 필요)
#             clip_model_name: CLIP 모델 ('ViT-B/32')
#             lamb: Loss weight
#             max_value: Loss clipping 최댓값
#             device: 디바이스
#             freeze_teacher_projection: Teacher projection layer freeze 여부
#             use_clip_text: CLIP text encoder 사용 여부 (False면 learnable embeddings)
#             normalize_similarity: True면 정규화 후 코사인 유사도, False면 정규화 없이 내적만 계산
#         """
#         super().__init__()
        
#         self.num_classes = num_classes
#         self.device = device
#         self.lamb = lamb
#         self.max_value = max_value
#         self.use_clip_text = use_clip_text
#         self.normalize_similarity = normalize_similarity
#         self.debug = bool(debug)
#         self.print_interval = int(print_interval)
#         self._dbg_count = 0
        
#         # ============================================
#         # 1. Text Embeddings 초기화 
#         #    우선순위: (A) 인자 class_names → (B) 인자 texts_path → (C) ENV 파일(TEXT_KD_FILE) → (D) ENV JSON(TEXT_KD_TEXTS)
#         # ============================================
#         if use_clip_text:
#             texts: Optional[List[str]] = None
#             if class_names is not None:
#                 texts = class_names
#             else:
#                 # (B) 명시적 경로 인자
#                 file_path = texts_path
#                 # (C) 환경변수 파일 경로
#                 if file_path is None:
#                     file_path = os.environ.get('TEXT_KD_FILE')
#                 if file_path and os.path.isfile(file_path):
#                     if file_path.endswith('.json'):
#                         with open(file_path, 'r') as f:
#                             loaded = json.load(f)
#                         # JSON은 리스트로 가정
#                         assert isinstance(loaded, list), 'JSON file must contain a list of texts.'
#                         texts = [str(x) for x in loaded]
#                     else:
#                         # 줄단위 텍스트 파일
#                         with open(file_path, 'r') as f:
#                             texts = [ln.strip() for ln in f.readlines() if ln.strip()]
#                 if texts is None:
#                     # (D) 환경변수 JSON
#                     raw = os.environ.get('TEXT_KD_TEXTS')
#                     if raw:
#                         try:
#                             texts = json.loads(raw)
#                         except Exception:
#                             pass
#             if texts is None:
#                 raise AssertionError("No class/prompt texts provided. Set class_names or env TEXT_KD_FILE/TEXT_KD_TEXTS.")
#             assert len(texts) == num_classes, \
#                 f"texts length ({len(texts)}) != num_classes ({num_classes})"

#             self.class_names = texts

#             print(f"[TextKD] Loading CLIP model '{clip_model_name}' for text encoding...")
#             clip_model, _ = clip.load(clip_model_name, device=device)

#             with torch.no_grad():
#                 text_tokens = clip.tokenize(texts).to(device)
#                 text_embeddings = clip_model.encode_text(text_tokens)  # (num_classes, 512)
#                 # normalize_similarity가 True일 때만 정규화 (False면 내적만 사용)
#                 if normalize_similarity:
#                     text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

#             self.register_buffer('text_embeddings', text_embeddings.float())

#             del clip_model
#             torch.cuda.empty_cache()

#             print(f"[TextKD] Text embeddings ready")
#             print(f"  - Num texts: {len(texts)} | Dim: {text_embeddings.shape[-1]} | CLIP: {clip_model_name}")
        
#         else:
#             # Learnable class embeddings (CLIP 없이)
#             self.class_names = None
            
#             # Learnable embeddings
#             self.text_embeddings = nn.Parameter(
#                 torch.randn(num_classes, 512),
#                 requires_grad=True
#             )
#             nn.init.xavier_uniform_(self.text_embeddings)
            
#             print(f"[TextKD] Using learnable class embeddings")
#             print(f"  - Embeddings: {self.text_embeddings.shape}")
        
#         # ============================================
#         # 2. feature Projection Layers (Student & Teacher)
#         # ============================================
#         # Student projection (학습 가능)
#         self.student_projection = nn.Conv2d(
#             in_channels=256,
#             out_channels=512,  # CLIP embedding dimension
#             kernel_size=1,
#             bias=False
#         )
        
#         # Teacher projection
#         self.teacher_projection = nn.Conv2d(
#             in_channels=256,
#             out_channels=512,  # CLIP embedding dimension
#             kernel_size=1,
#             bias=False
#         )
        
#         if freeze_teacher_projection:
#             for param in self.teacher_projection.parameters():
#                 param.requires_grad = False
        
#         # ============================================
#         # 3. Loss Module
#         # ============================================
#         from mmseg.models.losses.kd_loss import TextGuidedKDLoss
#         self.loss_fn = TextGuidedKDLoss(
#             lamb=lamb,
#             max_value=max_value
#         )
        
#         print(f"[TextKD] Initialized")
#         print(f"  - Classes: {num_classes}")
#         print(f"  - Lambda: {lamb}")
#         print(f"  - Max value: {max_value}")
#         print(f"  - Freeze teacher projection: {freeze_teacher_projection}")
#         print(f"  - Use CLIP: {use_clip_text}")
#         print(f"  - Normalize similarity: {normalize_similarity}")

    
#     def _match_dtype_device(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
#         """x 를 ref 의 dtype/device 에 맞춤"""
        
#         if x.dtype != ref.dtype or x.device != ref.device:
#             x = x.to(device=ref.device, dtype=ref.dtype)
#         return x

#     def forward(self, 
#         student_features: torch.Tensor,
#         teacher_features: torch.Tensor,
#         mask: Optional[torch.Tensor] = None) -> dict:
#         """
#         Text-Guided KD Loss 계산
#         """
#         self._dbg_count += 1

#         B, C, H, W = student_features.shape

#         # 1. Feature projection
#         student_projected = self.student_projection(student_features)  # (B, 512, H, W)
#         teacher_projected = self.teacher_projection(teacher_features)  # (B, 512, H, W)

#         # 2. Text embeddings 가져오기 + dtype/device 맞추기
#         text_emb = self._match_dtype_device(self.text_embeddings, student_projected)
#         # text_emb: (num_classes, 512)

#         # 3. Text-Image similarity 계산 (정규화 없이 raw similarity)
#         student_similarity = torch.einsum('bchw,nc->bnhw', student_projected, text_emb)  # (B, num_classes, H, W)
#         teacher_similarity = torch.einsum('bchw,nc->bnhw', teacher_projected, text_emb)

#         # 4. 유사도 맵 정규화 (여기가 핵심)
#         if self.normalize_similarity:
#             # 각 (b, h, w) 위치에서 클래스 벡터를 L2-norm 1로 정규화 ()
#             student_similarity = F.normalize(student_similarity, p=2, dim=1)
#             teacher_similarity = F.normalize(teacher_similarity, p=2, dim=1)

#         # 5. 디버그 출력
#         if self.debug and self._dbg_count == 1:
#             print(f"[TextKD] step={self._dbg_count}")
#             print(f"  student_features: {tuple(student_features.shape)}")
#             print(f"  teacher_features: {tuple(teacher_features.shape)}")
#             print(f"  student_projected: {tuple(student_projected.shape)}")
#             print(f"  teacher_projected: {tuple(teacher_projected.shape)}")
#             print(f"  student_similarity: {tuple(student_similarity.shape)}")
#             print(f"  teacher_similarity: {tuple(teacher_similarity.shape)}")

#         # 6. Loss 계산
#         loss = self.loss_fn(
#             student_similarity=student_similarity,
#             teacher_similarity=teacher_similarity,
#             mask=mask
#         )

#         return {'kd_loss': loss}

# text_kd.py
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
    Text-Guided Knowledge Distillation Module
    
    Flow:
    1. Text embedding 추출 (초기화 시 한 번만): class names → CLIP → (num_classes, 512)
    2. Student/Teacher feature projection: (B, 256, H, W) → (B, 512, H, W)
    3. Text-Image similarity: (B, 512, H, W) × (num_classes, 512) → (B, num_classes, H, W)
    4. 유사도 맵 정규화(optional) + MSE Loss: ||student_similarity - teacher_similarity||²
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
                 normalize_similarity: bool = True,  # True면 유사도 맵을 class 차원 기준 L2 정규화
                 debug: bool = False,
                 print_interval: int = 1): # 디버그 출력 주기해제. 현재는 안씀.
        """
        Args:
            num_classes: 클래스 개수
            class_names: 클래스 이름 리스트 (use_clip_text=True일 때 우선 사용)
            texts_path: 클래스 이름이 들어있는 json 또는 txt 파일 경로
            clip_model_name: CLIP 모델 이름 (예: 'ViT-B/32')
            lamb: KD loss weight (TextGuidedKDLoss 내부에서 사용)
            max_value: KD loss clipping 최댓값
            device: 디바이스 문자열 ('cuda', 'cuda:0' 등)
            freeze_teacher_projection: Teacher projection conv를 freeze할지 여부
            use_clip_text: True면 CLIP text encoder로 텍스트 임베딩 추출, False면 learnable embedding 사용
            normalize_similarity:
                True  → (B, C, H, W) 유사도 맵에서 C=num_classes 축 기준 L2 정규화 후 MSE
                False → 정규화 없이 raw similarity 맵으로 MSE
            debug: True면 일정 step마다 간단한 통계/예시 출력
            print_interval: debug 출력 주기 (step % print_interval == 0일 때 출력)
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
        #    우선순위: (A) class_names → (B) texts_path → (C) ENV TEXT_KD_FILE → (D) ENV TEXT_KD_TEXTS
        # ============================================
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
            # Learnable class embeddings (CLIP 없이)
            self.class_names = None
            self.text_embeddings = nn.Parameter(
                torch.randn(num_classes, 512),
                requires_grad=True
            )
            nn.init.xavier_uniform_(self.text_embeddings)
            
            print(f"[TextKD] Using learnable class embeddings")
            print(f"  - Embeddings: {self.text_embeddings.shape}")
        
        # ============================================
        # 2. Feature Projection Layers (Student & Teacher)
        # ============================================
        # Student projection (학습 가능)
        self.student_projection = nn.Conv2d(
            in_channels=256,
            out_channels=512,  # CLIP embedding dimension
            kernel_size=1,
            bias=False
        )
        
        # Teacher projection
        self.teacher_projection = nn.Conv2d(
            in_channels=256,
            out_channels=512,  # CLIP embedding dimension
            kernel_size=1,
            bias=False
        )
        
        if freeze_teacher_projection:
            for param in self.teacher_projection.parameters():
                param.requires_grad = False

        # if debug:
        #     for name, p in self.teacher_projection.named_parameters():
        #         def _make_hook(n):
        #             def _hook(grad):
        #                 if grad is not None:
        #                     print(f"[TextKD][GRAD] teacher_projection.{n} grad_norm={grad.norm().item():.6f}")
        #                 else:
        #                     print(f"[TextKD][GRAD] teacher_projection.{n} grad=None")
        #             return _hook
        #         p.register_hook(_make_hook(name))
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
    
    def _match_dtype_device(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """x 를 ref 의 dtype/device 에 맞춤"""
        if x.dtype != ref.dtype or x.device != ref.device:
            x = x.to(device=ref.device, dtype=ref.dtype)
        return x

    @torch.no_grad()
    def _debug_similarity(self,
                          student_projected: torch.Tensor,
                          teacher_projected: torch.Tensor,
                          student_sim: torch.Tensor,
                          teacher_sim: torch.Tensor,
                          student_sim_raw: Optional[torch.Tensor] = None,
                          teacher_sim_raw: Optional[torch.Tensor] = None) -> None:
        """유사도 맵 정규화 전/후를 간단히 보는 디버깅 출력"""
        B, Cp, H, W = student_projected.shape
        _, Cc, _, _ = student_sim.shape  # Cc = num_classes

        print("\n[TextKD DEBUG]")
        print(f"  step              : {self._dbg_count}")
        print(f"  proj shape        : {tuple(student_projected.shape)}")
        print(f"  sim shape         : {tuple(student_sim.shape)}")
        print(f"  num_classes       : {Cc}")
        print(f"  normalize_sim     : {self.normalize_similarity}")

        def _stats(name: str, t: torch.Tensor):
            t_min = float(t.min())
            t_max = float(t.max())
            t_mean = float(t.mean())
            t_std = float(t.std())
            print(f"  [{name}] min={t_min:.4f}, max={t_max:.4f}, mean={t_mean:.4f}, std={t_std:.4f}")

        if student_sim_raw is not None:
            _stats("student_sim_raw", student_sim_raw)
            _stats("teacher_sim_raw", teacher_sim_raw)
        _stats("student_sim", student_sim)
        _stats("teacher_sim", teacher_sim)

        # 예시 픽셀 하나에서 클래스별 유사도 벡터 출력
        b = 0
        h = H // 2
        w = W // 2

        print(f"\n  Sample location   : b={b}, h={h}, w={w}")
        if student_sim_raw is not None:
            s_raw_vec = student_sim_raw[b, :, h, w].cpu()
            t_raw_vec = teacher_sim_raw[b, :, h, w].cpu()
            print(f"  student_raw_vec   : {s_raw_vec.numpy()}")
            print(f"  teacher_raw_vec   : {t_raw_vec.numpy()}")

        s_vec = student_sim[b, :, h, w].cpu()
        t_vec = teacher_sim[b, :, h, w].cpu()
        print(f"  student_sim_vec   : {s_vec.numpy()}")
        print(f"  teacher_sim_vec   : {t_vec.numpy()}")

        s_norm = float(torch.norm(s_vec, p=2))
        t_norm = float(torch.norm(t_vec, p=2))
        print(f"  ||student_sim_vec||_2 = {s_norm:.4f}")
        print(f"  ||teacher_sim_vec||_2 = {t_norm:.4f}")

        # projection feature norm도 참고용 출력
        sp_vec = student_projected[b, :, h, w].cpu()
        tp_vec = teacher_projected[b, :, h, w].cpu()
        sp_norm = float(torch.norm(sp_vec, p=2))
        tp_norm = float(torch.norm(tp_vec, p=2))
        print(f"  ||student_proj_vec||_2 = {sp_norm:.4f} (512-dim)")
        print(f"  ||teacher_proj_vec||_2 = {tp_norm:.4f} (512-dim)")                
        print("[TextKD DEBUG END]\n")

    def forward(
        self, 
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Text-Guided KD Loss 계산
        
        Args:
            student_features: (B, 256, H, W) student feature map
            teacher_features: (B, 256, H, W) teacher feature map
            mask: Optional (B, 1, H, W) 또는 (B, H, W) kd 적용 영역 마스크
        Returns:
            {'kd_loss': scalar_tensor}
        """
        self._dbg_count += 1

        B, C, H, W = student_features.shape

        # 1. Feature projection
        student_projected = self.student_projection(student_features)   # (B, 512, H, W)
        teacher_projected = self.teacher_projection(teacher_features)   # (B, 512, H, W)

        # 2. Text embeddings + dtype/device 맞추기
        text_emb = self._match_dtype_device(self.text_embeddings, student_projected)  # (num_classes, 512)

        # 3. Text-Image similarity 계산 (정규화 전 raw similarity)
        student_similarity = torch.einsum('bchw,nc->bnhw', student_projected, text_emb)  # (B, num_classes, H, W)
        teacher_similarity = torch.einsum('bchw,nc->bnhw', teacher_projected, text_emb)

        # 디버그용 raw copy
        student_sim_raw = None
        teacher_sim_raw = None
        if self.debug and self._dbg_count == 1:
            student_sim_raw = student_similarity.detach().clone()
            teacher_sim_raw = teacher_similarity.detach().clone()

        # 4. 유사도 맵 정규화
        if self.normalize_similarity:
            # 각 (b, h, w) 위치에서 클래스 벡터(num_classes)를 L2-norm 1로 정규화
            student_similarity = F.normalize(student_similarity, p=2, dim=1)
            teacher_similarity = F.normalize(teacher_similarity, p=2, dim=1)

        # 5. 디버그 출력 (간단 버전)
        if self.debug and self._dbg_count == 1:
            self._debug_similarity(
                student_projected=student_projected.detach(),
                teacher_projected=teacher_projected.detach(),
                student_sim=student_similarity.detach(),
                teacher_sim=teacher_similarity.detach(),
                student_sim_raw=student_sim_raw,
                teacher_sim_raw=teacher_sim_raw
            )

        # 6. Loss 계산
        loss = self.loss_fn(
            student_similarity=student_similarity,
            teacher_similarity=teacher_similarity,
            mask=mask
        )

        return {'kd_loss': loss}

