from count_weight import get_D, get_weight, get_target_bpp
import math
import json

def process_w(weight, segments=50):
    width = 1.0/segments
    processed_w = {}
    for k, v in weight.items():
        index = math.floor(v / width)
        if index == segments:
            index = index-1
        new_v = (index + 0.5)*width
        processed_w[k] = new_v
    return processed_w

def get_fixed_w():
    quality = [1, 15, 2, 3, 35, 4, 5, 6]
    layer = ['0', '1', '2', '3', 'pool']
    names, all_D, remove_names = get_D(quality, layer)
    weight = get_weight(names, all_D, quality, layer)
    fixed_weight = process_w(weight)
    return fixed_weight

def count_weigt_get_target_bpp(lmbda):
    quality = [1, 15, 2, 3, 35, 4, 5, 6]
    layer = ['0', '1', '2', '3', 'pool']
    names, all_D, remove_names = get_D(quality, layer)
    weight = get_weight(names, all_D, quality, layer)
    # fixed_weight = process_w(weight)
    with open('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian/new_weight.json', 'r') as file:
        new_weight = json.load(file)
    # elic pretrained
    a = {'0': 10.953, '1': 10.567, '2': 22.266, '3': 25.237, 'pool': 10.241}
    b = {'0': -0.552, '1': -0.678, '2': -0.64, '3': -0.741, 'pool': -0.742}
    # lictcm pretrained a,b
    # a = {'0': 14.92, '1': 15.144, '2': 30.523, '3': 38.994, 'pool': 15.766}
    # b = {'0': -0.652, '1': -0.791, '2': -0.828, '3': -0.952, 'pool': -0.789}
    target_bpp = get_target_bpp(names, layer, new_weight, a, b, lmbda)
    return target_bpp, remove_names
