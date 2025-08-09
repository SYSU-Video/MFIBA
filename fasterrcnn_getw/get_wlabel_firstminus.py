from process_w import get_fixed_w
import os

import subprocess
import numpy as np
import json
def update_points():
    cmd = ['python', '/home/ta/liujunle/sda2/fasterrcnn_getw/train.py',
           '--data-path', '/home/ta/liujunle/coco', '--dataset', 'coco', '--num-classes', '90',
           '--model', 'resnet50', '--batch-size', '16', '--pretrained', '--test-only', '--lmbda_for_update', '900']
    subprocess.run(cmd, check=True)
    cmd = ['python', '/home/ta/liujunle/sda2/fasterrcnn_getw/train.py',
           '--data-path', '/home/ta/liujunle/coco', '--dataset', 'coco', '--num-classes', '90',
           '--model', 'resnet50', '--batch-size', '16', '--pretrained', '--test-only', '--lmbda_for_update', '1100']
    subprocess.run(cmd, check=True)
    points = [np.load('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/point1_bits_map.npy'),
              np.load('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/temp_bits_map.npy'),
              np.load('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/point3_bits_map.npy')]
    return points

def find_k_by_v(dir, target_v):
    keys = []
    for k, v in dir.items():
        if abs(v-target_v) < 0.0001:
            keys.append(k)
    return keys

def is_above(x1, y1, x2, y2, x, y):
    slope = (y2-y1)/(x2-x1)
    y_on_line = slope*(x-x1)+y1
    return y>y_on_line

def save_w(dir, temp):
    if temp:
        with open('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/temp_weight.json', 'w') as file:
            json.dump(dir, file)
    else:
        with open('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/new_weight.json', 'w') as file:
            json.dump(dir, file)

def test_w(points):
    cmd = ['python', '/home/ta/liujunle/sda2/fasterrcnn_getw/train.py',
           '--data-path', '/home/ta/liujunle/coco', '--dataset', 'coco', '--num-classes', '90',
           '--model', 'resnet50', '--batch-size', '16', '--pretrained', '--test-only',]
    subprocess.run(cmd, check=True)
    bits_map = np.load('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/temp_bits_map.npy')
    print('bits_map:', bits_map)
    print('test_points:', points)
    bits = bits_map[0]
    map = bits_map[1]
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    if bits<x1 or bits>x3:
        return False, points
    elif bits<=x2:
        if is_above(x1, y1, x2, y2, bits, map):
            points = update_points()
            print('updated points', points)
            return True,  points
        else:
            return False, points
    elif bits>x2:
        if is_above(x2, y2, x3, y3, bits, map):
            points = update_points()
            print('updated points', points)
            return True,  points
        else:
            return False, points
    #update line

if __name__ == "__main__":
    lmbda = 1000
    D = [i/100 for i in range(1, 100, 2)]
    W = get_fixed_w()
    new_W = W.copy()
    change_flag_add = {}
    change_flag_minus = {}
    org_flag = {}
    for k in W.keys():
        org_flag[k] = 1
    for d in D:
        print("d:", d)
        keys = find_k_by_v(W, d)
        print('ken_num:', len(keys))
        flag_add = 1
        flag_minus = 1
        add = 0
        minus = 0
        points = [np.load('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/point1_bits_map.npy'),
                  np.load('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/point2_bits_map.npy'),
                  np.load('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/point3_bits_map.npy')]
        #try minus
        temp_W = new_W.copy()
        while flag_minus:
            minus = minus - 0.01
            for k in keys:
                if org_flag[k] == 1:
                    temp_W[k] = temp_W[k] - 0.01
                    if temp_W[k]<0:
                        print('error')
            save_w(temp_W, temp=True)
            # test
            flag_minus, points = test_w(points)
            if flag_minus:
                print('minus:', minus)
                new_W = temp_W.copy()
                save_w(new_W, temp=False)
                change_flag_minus[str(d)] = minus
            else:
                break
        #try add
        temp_W = new_W.copy()
        while flag_add:
            add = add + 0.01
            for k in keys:
                if org_flag[k] == 1:
                    temp_W[k] = temp_W[k]+0.01
            save_w(temp_W, temp=True)
            #test
            flag_add, points = test_w(points)
            if flag_add:
                print('add:', add)
                new_W = temp_W.copy()
                save_w(new_W, temp=False)
                change_flag_add[str(d)] = add
            else:
                break
        print('--------------------------------------------------------------------------')
        for k in keys:
            org_flag[k] = 0

    with open('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/change_flag_add.json', 'w') as file:
        json.dump(change_flag_add, file)
    with open('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_minus/change_flag_minus.json', 'w') as file:
        json.dump(change_flag_minus, file)