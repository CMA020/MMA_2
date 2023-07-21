import os

_base_ = [
    '../../_base_/models/tsn_r50.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=2
    ))

# dataset settings
dataset_type = 'VideoDataset'
data_root_val = os.path.expanduser('content/MMA_2/MMA_data/val')
ann_file_val = os.path.expanduser('content/MMA_2/MMA_data/val_video.txt')

file_client_args = dict(io_backend='disk')

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(1920,1080)),
    dict(type='TenCrop', crop_size=1920),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = None

# optimizer
optim_wrapper = None

# evaluator
val_evaluator = dict(type='TopKAccuracy', k=1)

default_hooks = dict()

load_from =  os.path.expanduser('content/MMA_2/work_dirs/tsn_ucf101/epoch_66.pth')