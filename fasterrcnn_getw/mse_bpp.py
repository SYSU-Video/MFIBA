import csv

def get_mse_bpp_dict(quality):
    mse_dict = {}
    bpp_dict = {}
    for q in quality:
        filename = '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_'+str(int(q)) + '/result.csv'
        # filename = '/home/ta/liujunle/sda2/LIC-TCM/features_for_lic/feature_decode_' + str(int(q)) + '/result.csv'
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                bpp_dict[line['name'] + '_'+str(int(q))] = float(line['bpp'])
                mse_dict[line['name'] + '_'+str(int(q))] = float(line['mse'])
    return mse_dict, bpp_dict

def count_mse_bpp_per_layer(quality, mse_dict, bpp_dict):
    num = len(mse_dict)/40
    result = {}
    for q in quality:
        layer0_mse = 0
        layer0_bpp = 0
        layer1_mse = 0
        layer1_bpp = 0
        layer2_mse = 0
        layer2_bpp = 0
        layer3_mse = 0
        layer3_bpp = 0
        layer4_mse = 0
        layer4_bpp = 0
        for k,v in mse_dict.items():
            if k.endswith('-0.png' + '_' + str(int(q))):
                layer0_mse = layer0_mse + v
                layer0_bpp = layer0_bpp + bpp_dict[k]
            elif k.endswith('-1.png' + '_' + str(int(q))):
                layer1_mse = layer1_mse + v
                layer1_bpp = layer1_bpp + bpp_dict[k]
            elif k.endswith('-2.png' + '_' + str(int(q))):
                layer2_mse = layer2_mse + v
                layer2_bpp = layer2_bpp + bpp_dict[k]
            elif k.endswith('-3.png' + '_' + str(int(q))):
                layer3_mse = layer3_mse + v
                layer3_bpp = layer3_bpp + bpp_dict[k]
            elif k.endswith('-pool.png' + '_' + str(int(q))):
                layer4_mse = layer4_mse + v
                layer4_bpp = layer4_bpp + bpp_dict[k]
        result['layer0_mse_' + str(int(q))] = layer0_mse / num
        result['layer0_bpp_' + str(int(q))] = layer0_bpp / num
        result['layer1_mse_' + str(int(q))] = layer1_mse / num
        result['layer1_bpp_' + str(int(q))] = layer1_bpp / num
        result['layer2_mse_' + str(int(q))] = layer2_mse / num
        result['layer2_bpp_' + str(int(q))] = layer2_bpp / num
        result['layer3_mse_' + str(int(q))] = layer3_mse / num
        result['layer3_bpp_' + str(int(q))] = layer3_bpp / num
        result['layer4_mse_' + str(int(q))] = layer4_mse / num
        result['layer4_bpp_' + str(int(q))] = layer4_bpp / num
    return result

def get_quality(layer_names, bpp_dict, quality, target_bpp_dict, id_str):
    Q = {}
    for layer_name in layer_names:
        bpp_dif = 100
        target_bpp = target_bpp_dict[id_str + '_' + layer_name]
        # if layer_name == 'pool':
        #     layer_name = '4'
        for q in quality:
            if abs(bpp_dict[id_str+'-'+layer_name+'.png_'+str(int(q))]-target_bpp)<bpp_dif:#bpp_dict['layer'+layer_name+'_bpp_' + str(int(q))]
                if layer_name == 'pool':
                    Q['4'] = str(int(q))
                else:
                    Q[layer_name] = str(int(q))
                bpp_dif = abs(bpp_dict[id_str+'-'+layer_name+'.png_'+str(int(q))]-target_bpp)#bpp_dict['layer'+layer_name+'_bpp_' + str(int(q))]
    return Q


if __name__ == "__main__":
    layers = ['0', '1', '2', '3', 'pool']
    quality = [1,  2, 3, 4, 5, 6]
    mse_dict, bpp_dict = get_mse_bpp_dict(quality)
    result = count_mse_bpp_per_layer(quality, mse_dict, bpp_dict)
    for k,v in result.items():
        if 'layer0_mse' in k:
            print('-------------------------------------------------------------------------------------')
        print(k,v)