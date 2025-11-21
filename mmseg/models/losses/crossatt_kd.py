
#crossatt_kd
import os
import json
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CrossAttentionKD(nn.Module):
    """
    Cross Attention Knowledge Distillation
    
    KD 과정에서:
    - Teacher: features → cross attention layers → attention map 생성 (이 layers들도 학습됨)
    - Student: features → cross attention layers → attention map 생성 (이 layers들도 학습됨)
    - Loss: MSE(student_attn, teacher_attn) → Student cross attention layers 학습
    """
    
    def __init__(self, 
                 num_classes: int = 11,
                 class_names: List[str] = None,
                 texts_path: Optional[str] = "./mmseg/models/losses/camvid_classes.json",
                 feature_dim: int = 256, 
                 text_dim: int = 512, 
                 num_heads: int = 4,
                 lamb_i2t: float = 0.1,
                 lamb_t2i: float = 0.1,
                 clip_model_name: str = 'ViT-B/32',
                 use_clip_text: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.lamb_i2t = lamb_i2t
        self.lamb_t2i = lamb_t2i
        self.device = device
        self.use_clip_text = use_clip_text
        
        # Text embeddings 초기화
        if use_clip_text:
            texts: Optional[List[str]] = None
            
            if class_names is not None:
                texts = class_names
            else:
                file_path = texts_path
                if file_path is None:
                    file_path = os.environ.get('TEXT_KD_FILE')
                    
                if file_path and os.path.isfile(file_path):
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            loaded = json.load(f)
                        texts = [str(x) for x in loaded]
                    else:
                        with open(file_path, 'r') as f:
                            texts = [ln.strip() for ln in f.readlines() if ln.strip()]
                
                if texts is None:
                    raw = os.environ.get('TEXT_KD_TEXTS')
                    if raw:
                        try:
                            texts = json.loads(raw)
                        except Exception:
                            pass
            
            if texts is None:
                raise AssertionError("No class/prompt texts provided for CrossAttentionKD")
            
            assert len(texts) == num_classes
            self.class_names = texts
            
            print(f"[CrossAttentionKD] Loading CLIP model '{clip_model_name}'...")
            clip_model, _ = clip.load(clip_model_name, device=device)
            
            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(device)
                text_embeddings = clip_model.encode_text(text_tokens)
                text_embeddings = text_embeddings.float()
            
            self.register_buffer('text_embeddings', text_embeddings) #text embedding 고정 및 생성. # (num_classes, text_dim)
            
            del clip_model
            torch.cuda.empty_cache()
            
            print(f"[CrossAttentionKD] Text embeddings loaded: {text_embeddings.shape}") # 11, 512
        else:
            self.text_embeddings = nn.Parameter(
                torch.randn(num_classes, text_dim),
                requires_grad=True
            )
            nn.init.xavier_uniform_(self.text_embeddings)
            print(f"[CrossAttentionKD] Using learnable text embeddings")
        
        # ===== Student Cross Attention Layers (KD 중 학습됨) =====
        self.student_i2t_q = nn.Conv2d(feature_dim, text_dim, 1) # 1x1 conv
        self.student_i2t_k = nn.Linear(text_dim, text_dim) # 512 -> 512 (nn.linear는 마지막 차원에 적용(11,512))
        self.student_i2t_v = nn.Linear(text_dim, text_dim)
        
        self.student_t2i_q = nn.Linear(text_dim, text_dim)
        self.student_t2i_k = nn.Conv2d(feature_dim, text_dim, 1)
        self.student_t2i_v = nn.Conv2d(feature_dim, text_dim, 1)
        
        # ===== Teacher Cross Attention Layers (KD 중 학습됨) =====
        self.teacher_i2t_q = nn.Conv2d(feature_dim, text_dim, 1)
        self.teacher_i2t_k = nn.Linear(text_dim, text_dim)
        self.teacher_i2t_v = nn.Linear(text_dim, text_dim)
        
        self.teacher_t2i_q = nn.Linear(text_dim, text_dim)
        self.teacher_t2i_k = nn.Conv2d(feature_dim, text_dim, 1)
        self.teacher_t2i_v = nn.Conv2d(feature_dim, text_dim, 1)
        
        print(f"[CrossAttentionKD] Initialized")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Feature dim: {feature_dim} -> Text dim: {text_dim}")
        print(f"  - Num heads: {num_heads}, Head dim: {self.head_dim}")
        print(f"  - Lambda I2T: {lamb_i2t}, Lambda T2I: {lamb_t2i}")
        print(f"  - Teacher cross attention: TRAINABLE (학습됨)")
        print(f"  - Student cross attention: TRAINABLE (학습됨)")
    
    def _match_dtype_device(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.dtype != ref.dtype or x.device != ref.device:
            x = x.to(device=ref.device, dtype=ref.dtype)
        return x
    
    def compute_attention(self, features, text_embeds, is_teacher=False):
        """
        Compute bidirectional cross attention
        
        Args:
            features: (B, feature_dim, H, W)
            text_embeds: (B, num_classes, text_dim)
            is_teacher: Use teacher or student layers
            
        Returns:
            out_i2t: (B, num_heads, H*W, num_classes)
            out_t2i: (B, num_heads, num_classes, H*W)
        """
        B, C, H, W = features.shape
        N = text_embeds.shape[1]  # num_classes
        
        # Layer 선택
        if is_teacher:
            i2t_q_layer = self.teacher_i2t_q
            i2t_k_layer = self.teacher_i2t_k
            i2t_v_layer = self.teacher_i2t_v
            t2i_q_layer = self.teacher_t2i_q
            t2i_k_layer = self.teacher_t2i_k
            t2i_v_layer = self.teacher_t2i_v
        else:
            i2t_q_layer = self.student_i2t_q
            i2t_k_layer = self.student_i2t_k
            i2t_v_layer = self.student_i2t_v
            t2i_q_layer = self.student_t2i_q
            t2i_k_layer = self.student_t2i_k
            t2i_v_layer = self.student_t2i_v
        
        # ===== Image-to-Text Attention =====
        Q_i2t = i2t_q_layer(features).flatten(2).permute(0, 2, 1)  # (B,512,H*W) -> (B, H*W, text_dim)
        K_i2t = i2t_k_layer(text_embeds)  # (B, N, text_dim)
        V_i2t = i2t_v_layer(text_embeds)  # (B, N, text_dim)
        
        Q_i2t = Q_i2t.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_i2t = K_i2t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_i2t = V_i2t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_i2t = (Q_i2t @ K_i2t.transpose(-2, -1)) * self.scale
        attn_i2t = attn_i2t.softmax(dim=-1)  # (B, num_heads, H*W, N)
        

        # ===== Text-to-Image Attention =====
        Q_t2i = t2i_q_layer(text_embeds)  # (B, N, text_dim)
        K_t2i = t2i_k_layer(features).flatten(2).permute(0, 2, 1)  # (B, H*W, text_dim)
        V_t2i = t2i_v_layer(features).flatten(2).permute(0, 2, 1)  # (B, H*W, text_dim)
        
        Q_t2i = Q_t2i.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_t2i = K_t2i.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_t2i = V_t2i.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_t2i = (Q_t2i @ K_t2i.transpose(-2, -1)) * self.scale
        attn_t2i = attn_t2i.softmax(dim=-1)  # (B, num_heads, N, H*W)


        return attn_i2t, attn_t2i
    
    def forward(self, 
                student_features: torch.Tensor,
                teacher_features: torch.Tensor,
                return_attn_maps: bool = False) -> dict:
        """
        Compute cross attention KD loss
        
        Flow:
        1. Teacher features → teacher cross attention layers → teacher attention map (gradient 계산됨)
        2. Student features → student cross attention layers → student attention map (gradient 계산됨)
        3. Loss = MSE(student_attn, teacher_attn)
        4. Backward 시 student cross attention layers가 업데이트됨
        
        Args:
            student_features: (B, feature_dim, H, W)
            teacher_features: (B, feature_dim, H, W)
            return_attn_maps: If True, return attention maps for logging
            
        Returns:
            dict: {
                'kd_loss': total loss,
                'i2t_loss': image-to-text attention loss,
                't2i_loss': text-to-image attention loss,
                'student_i2t': (optional) student i2t attention map,
                'student_t2i': (optional) student t2i attention map,
                'teacher_i2t': (optional) teacher i2t attention map,
                'teacher_t2i': (optional) teacher t2i attention map
            }
        """
        B = student_features.shape[0]
        
        # Text embeddings
        text_emb = self._match_dtype_device(self.text_embeddings, student_features)
        text_emb_batch = text_emb.unsqueeze(0).expand(B, -1, -1)  # (B, num_classes, text_dim)
        
        # Student attention maps (gradient 계산, student cross attention layers 학습됨)
        student_i2t, student_t2i = self.compute_attention(
            student_features, 
            text_emb_batch, 
            is_teacher=False
        )
        
        # Teacher attention maps (gradient 계산, teacher cross attention layers 학습됨)
        teacher_i2t, teacher_t2i = self.compute_attention(
            teacher_features, 
            text_emb_batch, 
            is_teacher=True
        )
        
        # Loss 계산 (각각 다른 lambda 적용)
        # detach() 없음 → teacher도 gradient 계산되지만, 주로 student 학습에 영향
        i2t_loss = F.mse_loss(student_i2t, teacher_i2t) * self.lamb_i2t
        t2i_loss = F.mse_loss(student_t2i, teacher_t2i) * self.lamb_t2i
        
        total_loss = i2t_loss + t2i_loss
        
        result = {
            'kd_loss': total_loss,
            'i2t_loss': i2t_loss,
            't2i_loss': t2i_loss
        }
        if return_attn_maps:
            result['student_i2t'] = student_i2t
            result['student_t2i'] = student_t2i
            result['teacher_i2t'] = teacher_i2t
            result['teacher_t2i'] = teacher_t2i
    
        return result