import csv
import numpy as np
from torch import nn
import torch
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
import matplotlib
print(font_manager.font_family_aliases)
# matplotlib.use('Agg')
# 设置全局字体为Times New Roman
rcParams['font.family'] = 'serif'
rcParams.update({'font.size':12})

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
            merge_data[line['name'] + '_1'] = line['bpp']
    with open(filename15) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_15'] = line['bpp']
    with open(filename2) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_2'] = line['bpp']
    with open(filename3) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_3'] = line['bpp']
    with open(filename35) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_35'] = line['bpp']
    with open(filename4) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_4'] = line['bpp']
    with open(filename5) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_5'] = line['bpp']
    with open(filename6) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            merge_data[line['name'] + '_6'] = line['bpp']
    return merge_data

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
                        if baseline[line['name']] - float(line['mean']) < 0.001:
                            all_D[line['name'] + '_' + line['noise_layer'] + '_' + str(int(q))] = 0
                        else:
                            all_D[line['name'] + '_' + line['noise_layer'] + '_' + str(int(q))] = baseline[line['name']] - float(line['mean'])
    return names, all_D, remove_name


def draw(x, y, a, b, image_id, layer):
    b_gen = -0.62 #{'0': -0.552, '1': -0.678, '2': -0.64, '3': -0.741, 'pool': -0.742}
    x_draw = np.linspace(0, 1, 100)
    y_draw = a*(np.power(x_draw, b))
    y_gen_draw = a*(np.power(x_draw, b_gen))
    y_fit = a*(np.power(x, b))
    y_gen_fit = a * (np.power(x, b_gen))
    RMSE = np.sqrt(np.mean((y-y_fit)**2))
    CC = np.corrcoef(y, y_fit)[0, 1]
    RMSE_gen = np.sqrt(np.mean((y - y_gen_fit) ** 2))
    CC_gen = np.corrcoef(y, y_gen_fit)[0, 1]
    plt.figure()
    plt.scatter(x, y, color='royalblue', label='Original task loss')
    plt.plot(x_draw, y_draw, color='lightcoral', label='Image-specific task loss-rate model')
    plt.plot(x_draw, y_gen_draw, color='turquoise', label='General task loss-rate model')
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.xlabel('Bpp')
    plt.ylabel('Task loss')
    plt.legend(loc='upper right')
    plt.title(f'Image-specific : RMSE: {RMSE:.3f}, Correlation: {CC:.3f} \n General : RMSE: {RMSE_gen:.3f}, Correlation: {CC_gen:.3f}')
    plt.savefig('./'+ image_id + str(layer) + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()




def fit_curve(all_D, all_bpp, image_id, quality, layer): #count a,b per-layer per-image
    x_data = []
    y_data = []
    for q in quality:
        x_data.append(float(all_bpp[image_id + '-' + layer + '.png' + '_' + str(int(q))]))
        y_data.append(float(all_D[image_id + '_' + layer + '_' + str(int(q))]))

    # if max(y_data) < 1e-4:
    #     a = 1e-8
    #     b = 1e-8
    #     return a,b

    def power_func(x, a, b):
        return a * (x ** b)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    initial_guess = (1e-4, -1e-4)
    a, b = initial_guess
    try:
        params, params_covariance = curve_fit(
            power_func,
            x_data,
            y_data,
            p0=initial_guess,
            bounds=([1e-10, -np.inf], [np.inf, 0]),  # a下限接近0，b上限为0
            method = 'trf'
        )
        a, b = params
    except RuntimeError as e:
        print("run time error:", x_data, y_data)
    draw(x_data, y_data, a, b, image_id, layer)

    return a,b

def get_target_bpp(all_D, all_bpp, names, quality, layers, lmbda):
    #names-->image id
    target_bpp = {}
    for n in names:
        for l in layers:
            a,b = fit_curve(all_D, all_bpp, n, quality, l)
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
            target_bpp[n + '_' + l] = (-la/(a*b))**(1/(b-1))
    return target_bpp

def count_weigt_get_target_bpp(lmbda):
    quality = [1, 15, 2, 3, 35, 4, 5, 6]
    layer = ['0', '1', '2', '3', 'pool']
    names, all_D, remove_names = get_D(quality, layer)
    all_bpp = get_bits_dict()
    target_bpp = get_target_bpp(all_D, all_bpp, names, quality, layer, lmbda)
    return target_bpp, remove_names

if __name__ == "__main__":
    count_weigt_get_target_bpp(5)