import math
import sys
import time
import os
import torch

import torchvision.models.detection.mask_rcnn
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import numpy as np
import utils
from process_w import count_weigt_get_target_bpp # count_weight->org_weights  process_w->caliberated_weights
from assign_per_image import count_weigt_get_target_bpp #fit curve for singe image
from mse_bpp import get_mse_bpp_dict, get_quality, count_mse_bpp_per_layer

import csv

def get_bits_dict():
    merge_data = {}
    filename1 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_1/result.csv'
    filename15 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_15/result.csv'
    filename2 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_2/result.csv'
    filename3 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_3/result.csv'
    filename35 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_35/result.csv'
    filename4 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_4/result.csv'
    filename5 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_5/result.csv'
    filename6 = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_6/result.csv'

    with open(filename1) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_1'] = [line['bits'], line['pixels']]
    with open(filename15) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_15'] = [line['bits'], line['pixels']]
    with open(filename2) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_2'] = [line['bits'], line['pixels']]
    with open(filename3) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_3'] = [line['bits'], line['pixels']]
    with open(filename35) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_35'] = [line['bits'], line['pixels']]
    with open(filename4) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_4'] = [line['bits'], line['pixels']]
    with open(filename5) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_5'] = [line['bits'], line['pixels']]
    with open(filename6) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_6'] = [line['bits'], line['pixels']]
    return merge_data

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets=targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, lmbda):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    # keypoint
    # iou_types = ["bbox", "keypoints"]
    # mask
    # iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    i=0
    start_time = time.time()
    target_bpp_dict, remove_names = count_weigt_get_target_bpp(lmbda=lmbda)
    print('lmbda:', lmbda)
    mse_dict, bpp_dict = get_mse_bpp_dict(quality = [1, 15, 2, 3, 35, 4, 5, 6])
    end_time = time.time()
    print('run_time1:', end_time-start_time)
    # mse_bpp_dict = count_mse_bpp_per_layer(quality = [1, 15, 2, 3, 35, 4, 5, 6], mse_dict=mse_dict, bpp_dict=bpp_dict)
    sum_mask_bpp = 0
    sum_mask_bits= 0
    img_bpp_dict = get_bits_dict()
    for images, targets in metric_logger.log_every(data_loader, 1000, header):
        images = list(img.to(device) for img in images)

        id = targets[0]['image_id'].item()
        id_str = str(id).zfill(12)
        start_time = time.time()
        if id_str in remove_names:
            Q_dict = {'0': '1', '1': '1', '2': '1', '3': '1', '4': '1'}
        else:
            Q_dict = get_quality(layer_names=['0', '1', '2', '3', 'pool'], bpp_dict=bpp_dict, quality=[1, 15, 2, 3, 35, 4, 5, 6], target_bpp_dict=target_bpp_dict, id_str=id_str)#mse_bpp_dict
        end_time = time.time()
        print('run_time2:', end_time-start_time)
        num_pixels = (float(img_bpp_dict[id_str + '-0.png' + '_' + str(int(Q_dict['0']))][1]) + float(
            img_bpp_dict[id_str + '-1.png' + '_' + str(int(Q_dict['1']))][1]) \
                      + float(img_bpp_dict[id_str + '-2.png' + '_' + str(int(Q_dict['2']))][1]) + float(
                    img_bpp_dict[id_str + '-3.png' + '_' + str(int(Q_dict['3']))][1]) \
                      + float(img_bpp_dict[id_str + '-pool.png' + '_' + str(int(Q_dict['4']))][1]))
        # for img in images:
        #     val = img.shape[-2:]
        #     assert len(val) == 2
        # num_pixels = num_pixels + val[0] * val[1]
        print('Q_dict:', Q_dict)
        sum_mask_bpp = sum_mask_bpp + (float(img_bpp_dict[id_str + '-0.png' + '_' + str(int(Q_dict['0']))][0]) + float(
            img_bpp_dict[id_str + '-1.png' + '_' + str(int(Q_dict['1']))][0]) \
                                       + float(
                    img_bpp_dict[id_str + '-2.png' + '_' + str(int(Q_dict['2']))][0]) + float(
                    img_bpp_dict[id_str + '-3.png' + '_' + str(int(Q_dict['3']))][0]) \
                                       + float(
                    img_bpp_dict[id_str + '-pool.png' + '_' + str(int(Q_dict['4']))][0])) / num_pixels
        sum_mask_bits = sum_mask_bits + (
                    float(img_bpp_dict[id_str + '-0.png' + '_' + str(int(Q_dict['0']))][0]) + float(
                img_bpp_dict[id_str + '-1.png' + '_' + str(int(Q_dict['1']))][0]) \
                    + float(img_bpp_dict[id_str + '-2.png' + '_' + str(int(Q_dict['2']))][0]) + float(
                img_bpp_dict[id_str + '-3.png' + '_' + str(int(Q_dict['3']))][0]) \
                    + float(img_bpp_dict[id_str + '-pool.png' + '_' + str(int(Q_dict['4']))][0]))

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images, id=id_str, Q_dict=Q_dict)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        i = i + 1
        if i == 2000:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('mask_noise_bpp', sum_mask_bpp/2000)
    print('mask_noise_bits', sum_mask_bits / 2000)
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # bits_map = np.array([sum_mask_bits / 2000, coco_evaluator.coco_eval['bbox'].stats[0]])
    # if lmbda == 500:
    #     np.save('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_500/temp_bits_map', bits_map)
    #     print('save temp')
    # elif lmbda == 400:
    #     np.save('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_500/point3_bits_map', bits_map)
    #     print('save point3')
    # elif lmbda == 600:
    #     np.save('/home/ta/liujunle/sda2/fasterrcnn_getw/update_qiexian_500/point1_bits_map', bits_map)
    #     print('save point1')
    torch.set_num_threads(n_threads)
    return coco_evaluator
