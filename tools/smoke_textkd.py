import os
import sys
import types
import torch
import importlib.util

TEXT_KD_PATH = "/home/leeju2/diffseg/IMRL_Project-main/mmseg/models/losses/text_kd.py"


def ensure_pkg(path):
    if path not in sys.modules:
        sys.modules[path] = types.ModuleType(path)
    return sys.modules[path]


def register_dummy_kd_loss():
    ensure_pkg("mmseg")
    ensure_pkg("mmseg.models")
    ensure_pkg("mmseg.models.losses")
    mod = types.ModuleType("mmseg.models.losses.kd_loss")
    import torch.nn.functional as F
    import torch as _torch

    class TextGuidedKDLoss:
        def __init__(self, lamb: float = 0.1, max_value: float = 10.0):
            self.lamb = lamb
            self.max_value = max_value
        def __call__(self, student_similarity, teacher_similarity, mask=None):
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                mask = mask.expand_as(student_similarity)
                student_similarity = student_similarity[mask]
                teacher_similarity = teacher_similarity[mask]
            loss = F.mse_loss(student_similarity, teacher_similarity)
            loss = loss * self.lamb
            loss = _torch.clamp(loss, max=self.max_value)
            return loss

    mod.TextGuidedKDLoss = TextGuidedKDLoss
    sys.modules["mmseg.models.losses.kd_loss"] = mod


def import_textkd(path):
    register_dummy_kd_loss()
    spec = importlib.util.spec_from_file_location("text_kd_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module.TextKD


def main():
    texts_json = "/home/leeju2/diffseg/IMRL_Project-main/mmseg/models/losses/camvid_classes.json"
    assert os.path.isfile(texts_json), f"Not found: {texts_json}"

    num_classes = 11
    B, C, H, W = 2, num_classes, 72, 96

    torch.manual_seed(0)
    student_logits = torch.randn(B, C, H, W)
    teacher_logits = torch.randn(B, C, H, W)

    TextKD = import_textkd(TEXT_KD_PATH)

    tkd = TextKD(
        num_classes=num_classes,
        texts_path=texts_json,
        clip_model_name='ViT-B/32',
        lamb=0.1,
        max_value=10.0,
        device='cpu',
        freeze_teacher_projection=False,
        use_clip_text=True,
        debug=True,
        print_interval=1,
    )

    out = tkd(student_logits=student_logits, teacher_logits=teacher_logits, mask=None)
    print("[SMOKE] kd_loss:", float(out['kd_loss']))


if __name__ == '__main__':
    main()
