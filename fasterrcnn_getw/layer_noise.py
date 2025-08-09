import os
import subprocess
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
layer = ['all']#['0', '1', '2', '3', 'pool']
# noise = ['gau', 'pois']#, 'pepper'
lamda = [1, 15, 2, 3, 35, 4, 5, 6]#[100, 150, 200, 250]
# checkboard_mask = ['1', '2', '3', '4']
# for mask_type in checkboard_mask:
# for noi in noise:
for noise_layer in layer:
    for lam in lamda:
        cmd = ['python', '/home/ta/liujunle/sda2/FasterRCNN-master/train.py',
                '--data-path', '/home/ta/liujunle/coco', '--dataset', 'coco', '--num-classes', '90',
                '--model', 'resnet50', '--batch-size', '16', '--pretrained', '--test-only',
                '--noise_layer', noise_layer, '--noise', 'elic', '--noise_l', str(lam)]
        subprocess.run(cmd, check=True)

