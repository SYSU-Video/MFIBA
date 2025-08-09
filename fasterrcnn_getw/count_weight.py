import csv
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt

def get_D(quality, layer):
    file_base = '/home/ta/liujunle/sda2/FasterRCNN-master/single_img_result/-1_-1_result.csv'
    all_D = {}
    baseline = {}
    names = []
    remove_name = []
    with open(file_base) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            if float(line['mean'])>0:
                baseline[line['name']] = float(line['mean'])
                names.append(line['name'])
            else:
                remove_name.append(line['name'])
    for q in quality:
        for l in layer:
            file = '/home/ta/liujunle/sda2/FasterRCNN-master/single_img_result/' + l + '_' + str(int(q)) + '_result.csv'
            with open(file) as csvfile:
                reader = csv.DictReader(csvfile)
                for line in reader:
                    if line['name'] not in remove_name:
                        if baseline[line['name']] - float(line['mean']) < 0.01:
                            all_D[line['name'] + '_' + line['noise_layer'] + '_' + str(int(q))] = 0
                        else:
                            all_D[line['name'] + '_' + line['noise_layer'] + '_' + str(int(q))] = baseline[line['name']] - float(line['mean'])
    return names, all_D, remove_name

def get_weight(names, dis, quality, layer):
    #weight_per_image_per_quality guiyihua
    w = {}
    weight = {}
    m = nn.Sigmoid()#not sigmoid
    for q in quality:
        for n in names:
            max_d = -100
            min_d = 100
            for l in layer:
                if dis[n + '_' + l + '_' +str(int(q))]>max_d:
                    max_d = dis[n + '_' + l + '_' +str(int(q))]
                if dis[n + '_' + l + '_' +str(int(q))]<min_d:
                    min_d = dis[n + '_' + l + '_' +str(int(q))]
            for l in layer:
                if max_d-min_d == 0:
                    w[n + '_' + l + '_' + str(int(q))] = 1
                else:
                    w[n + '_' + l + '_' + str(int(q))] = (dis[n + '_' + l + '_' +str(int(q))]-min_d)/(max_d-min_d)
    # weight_per_image
    for n in names:
        for l in layer:
            sum_w = 0
            for q in quality:
                sum_w = sum_w + w[n + '_' + l + '_' + str(int(q))]
            weight[n + '_' + l] = sum_w
    # weight guiyihua
    for n in names:
        sum_weight = 0
        for l in layer:
            sum_weight = sum_weight + weight[n + '_' + l]
        for l in layer:
            # weight[n + '_' + l] = m(torch.tensor((weight[n + '_' + l] / sum_weight)*10 - 5)).item()
            # weight[n + '_' + l] = (np.power(2, weight[n + '_' + l]/sum_weight)-1)/3
            weight[n + '_' + l] = weight[n + '_' + l] / sum_weight

    return weight

def get_target_bpp(names, layers, weight, a, b, lmbda):
    #names-->image id
    target_bpp = {}
    for n in names:
        for l in layers:
            if l=='0':
                la = lmbda
            elif l=='1':
                la = lmbda/4
            elif l=='2':
                la = lmbda/16
            elif l=='3':
                la = lmbda/60
            elif l == 'pool':
                la = lmbda/60
            else:
                print('error')
            # la = lmbda
            w = weight[n + '_' + l]
            target_bpp[n + '_' + l] = (-la/(max(w, 1e-8)*a[l]*b[l]))**(1/(b[l]-1))
    return target_bpp

def count_weigt_get_target_bpp(lmbda):
    quality = [1, 2, 3, 4, 5, 6]
    layer = ['0', '1', '2', '3', 'pool']
    names, all_D, remove_names = get_D(quality, layer)
    weight = get_weight(names, all_D, quality, layer)
    # elic pretrained a,b
    a = {'0': 10.953, '1': 10.567, '2': 22.266, '3': 25.237, 'pool': 10.241}
    b = {'0': -0.552, '1': -0.678, '2': -0.64, '3': -0.741, 'pool': -0.742}
    # lictcm pretrained a,b
    # a = {'0': 14.92, '1': 15.144, '2': 30.523, '3': 38.994, 'pool': 15.766}
    # b = {'0': -0.652, '1': -0.791, '2': -0.828, '3': -0.952, 'pool': -0.789}
    target_bpp = get_target_bpp(names, layer, weight, a, b, lmbda)
    return target_bpp, remove_names

if __name__ == "__main__":
    quality = [1, 2, 3, 4, 5, 6]
    layer = ['0', '1', '2', '3', 'pool']
    names, all_D, remove_name = get_D(quality, layer)
    # i = 0
    # for n in names:
    #     i = i+1
    #     if i == 20:
    #         break
    #     x = [1, 2, 3, 4, 5, 6, 7, 8]
    #     y_0 = [all_D[n+'_0'+'_'+str(int(1))], all_D[n+'_0'+'_'+str(int(15))], all_D[n+'_0'+'_'+str(int(2))], all_D[n+'_0'+'_'+str(int(3))],
    #          all_D[n+'_0'+'_'+str(int(35))], all_D[n+'_0'+'_'+str(int(4))], all_D[n+'_0'+'_'+str(int(5))], all_D[n+'_0'+'_'+str(int(6))]]
    #     y_1 = [all_D[n + '_1' + '_' + str(int(1))], all_D[n + '_1' + '_' + str(int(15))],all_D[n + '_1' + '_' + str(int(2))], all_D[n + '_1' + '_' + str(int(3))],
    #            all_D[n + '_1' + '_' + str(int(35))], all_D[n + '_1' + '_' + str(int(4))],all_D[n + '_1' + '_' + str(int(5))], all_D[n + '_1' + '_' + str(int(6))]]
    #     y_2 = [all_D[n + '_2' + '_' + str(int(1))], all_D[n + '_2' + '_' + str(int(15))],all_D[n + '_2' + '_' + str(int(2))], all_D[n + '_2' + '_' + str(int(3))],
    #            all_D[n + '_2' + '_' + str(int(35))], all_D[n + '_2' + '_' + str(int(4))],all_D[n + '_2' + '_' + str(int(5))], all_D[n + '_2' + '_' + str(int(6))]]
    #     y_3 = [all_D[n + '_3' + '_' + str(int(1))], all_D[n + '_3' + '_' + str(int(15))],all_D[n + '_3' + '_' + str(int(2))], all_D[n + '_3' + '_' + str(int(3))],
    #            all_D[n + '_3' + '_' + str(int(35))], all_D[n + '_3' + '_' + str(int(4))],all_D[n + '_3' + '_' + str(int(5))], all_D[n + '_3' + '_' + str(int(6))]]
    #     y_4 = [all_D[n + '_pool' + '_' + str(int(1))], all_D[n + '_pool' + '_' + str(int(15))],all_D[n + '_pool' + '_' + str(int(2))], all_D[n + '_pool' + '_' + str(int(3))],
    #            all_D[n + '_pool' + '_' + str(int(35))], all_D[n + '_pool' + '_' + str(int(4))],all_D[n + '_pool' + '_' + str(int(5))], all_D[n + '_pool' + '_' + str(int(6))]]
    #     plt.plot(x, y_0, 'bo-', alpha=1, linewidth=1, label='layer0')
    #     plt.plot(x, y_1, 'ro-', alpha=1, linewidth=1, label='layer1')
    #     plt.plot(x, y_2, 'go-', alpha=1, linewidth=1, label='layer2')
    #     plt.plot(x, y_3, 'ko-', alpha=1, linewidth=1, label='layer3')
    #     plt.plot(x, y_4, 'yo-', alpha=1, linewidth=1, label='layer4')
    #     plt.legend()
    #     plt.xlabel('Q')
    #     plt.ylabel('D')
    #     plt.show()

    weight = get_weight(names, all_D, quality, layer)
    # a = {'0':10.953, '1':10.567, '2':22.266, '3':25.237, 'pool': 10.241}
    # b = {'0':-0.552, '1':-0.678, '2':-0.64, '3':-0.741, 'pool': -0.742}
    # count_weigt_get_target_bpp(5)

    # print(weight)

