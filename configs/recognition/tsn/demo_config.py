import os
_base_ = [
    '../../_base_/models/tsn_r50.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=2  # change from 400 to 2 for your custom model
    ))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = os.path.expanduser('/content/MMA_2/MMA_data/demo')  # Update the path to your demo video directory
ann_file = os.path.expanduser('/content/MMA_2/MMA_data/demo/demo_video.txt')  # Update the path to the annotation file

file_client_args = dict(io_backend='disk')

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    #dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# DataLoader settings for prediction
test_dataloader = dict(
    batch_size=1,  # Set batch_size to 1 for inference
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='SequentialSampler'),  # Use SequentialSampler for prediction
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True
    )
)

# Load the custom weight file for prediction
load_from = '/content/MMA_2/work_dirs/tsn_ucf101/epoch_48.pth'  # Update the path to your custom weight file

# Remove learning policy, optimizer, evaluators, and hooks since we are performing prediction, not training

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=1)