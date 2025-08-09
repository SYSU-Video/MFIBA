import os
import subprocess
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
mask = ['target']#['cam', 'cam++']
noise = ['jpeg']
lamda = [10, 30, 50, 100, 150, 200, 300]#[1, 15, 2, 3, 35, 4, 5, 6]
# cmd = ['python', '/home/ta/liujunle/sda2/FasterRCNN-master/train.py',
#                     '--data-path', '/home/ta/liujunle/coco', '--dataset', 'coco', '--num-classes', '90',
#                     '--model', 'resnet50', '--batch-size', '16', '--pretrained', '--test-only',
#                     '--noise_layer', '-1', '--noise', 'jpeg', '--noise_l', str(10), '--mask_type', 'target', '--mask_roi', str(1)]
# subprocess.run(cmd, check=True)
for mask_type in mask:
    for noise_type in noise:
        for lam in lamda:
            cmd = ['python', '/home/ta/liujunle/sda2/FasterRCNN-master/train.py',
                    '--data-path', '/home/ta/liujunle/coco', '--dataset', 'coco', '--num-classes', '90',
                    '--model', 'resnet50', '--batch-size', '16', '--pretrained', '--test-only',
                    '--noise_layer', '-1', '--noise', noise_type, '--noise_l', str(lam), '--mask_type', mask_type, '--mask_roi', str(1)]
            subprocess.run(cmd, check=True)

for mask_type in mask:
    for noise_type in noise:
        for lam in lamda:
            cmd = ['python', '/home/ta/liujunle/sda2/FasterRCNN-master/train.py',
                    '--data-path', '/home/ta/liujunle/coco', '--dataset', 'coco', '--num-classes', '90',
                    '--model', 'resnet50', '--batch-size', '16', '--pretrained', '--test-only',
                    '--noise_layer', '-1', '--noise', noise_type, '--noise_l', str(lam), '--mask_type', mask_type, '--mask_roi', str(0)]
            subprocess.run(cmd, check=True)