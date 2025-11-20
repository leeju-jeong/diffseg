_base_ = [
    '../_base_/models/kd_diff_segformer_textkd.py',
    '../_base_/datasets/kd_camvid_512x384_f1.py', 
    '../_base_/schedules/poly10warm.py',  
    '../_base_/default_runtime.py'
]


# Teacher 체크포인트 경로 
teacher_checkpoint = '/root/diff/diffseg/work_dirs/Teacher/img_only/fold1_best_mIoU_iter_25000.pth'

model = dict(
    # KD 파라미터 오버라이드'
    use_kd=True,        # KD
    kd_type='crossatt_kd',  # textkd
    # kd_lamb=0.001,        # KD loss weight
    kd_lamb_i2t=0.01,
    kd_lamb_t2i=0.01,
    # normalize_similarity=False,  # True 면 코사인유사도(정규화) False면 내적만 사용
    diff_train=False
) 

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            # Student Cross Attention Layers (높은 lr)
            'cross_attn_kd.student_i2t_q': dict(lr_mult=10.0),
            'cross_attn_kd.student_i2t_k': dict(lr_mult=10.0),
            'cross_attn_kd.student_i2t_v': dict(lr_mult=10.0),
            'cross_attn_kd.student_t2i_q': dict(lr_mult=10.0),
            'cross_attn_kd.student_t2i_k': dict(lr_mult=10.0),
            'cross_attn_kd.student_t2i_v': dict(lr_mult=10.0),
            
            # Teacher Cross Attention Layers (높은 lr)
            'cross_attn_kd.teacher_i2t_q': dict(lr_mult=10.0),
            'cross_attn_kd.teacher_i2t_k': dict(lr_mult=10.0),
            'cross_attn_kd.teacher_i2t_v': dict(lr_mult=10.0),
            'cross_attn_kd.teacher_t2i_q': dict(lr_mult=10.0),
            'cross_attn_kd.teacher_t2i_k': dict(lr_mult=10.0),
            'cross_attn_kd.teacher_t2i_v': dict(lr_mult=10.0),
            
            # Decode Head
            'decode_head': dict(lr_mult=10.0),
            
            # 기타
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

# 학습률 스케줄러
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=750, # batch 4 ->1500 / 8 -> 750
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(multi_scale=True)
)

optimizer_config = dict()

# 체크포인트 및 평가 설정

runner = dict(type='IterBasedRunner', max_iters=15000) 
evaluation = dict(interval=1000, metric='mIoU', save_best = 'mIoU')
checkpoint_config = dict(by_epoch=False, interval=15000)


# 작업 디렉토리
work_dir = './work_dirs/kd/atttextkd_0.01_imgonly_pre_student/fold1'

# GPU 설정 추가
gpu_ids = range(0, 1)
#PYTHONPATH=$(pwd):$PYTHONPATH python tools/train.py configs/KD/camvid_DIFF2Seg_512t384s_at_fold1.py