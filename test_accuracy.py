import numpy as np
import json
import os

from weighting import *


def get_label(video_name):
    label = video_name.split('_')[-1]
    if label == 'fake':
        return True
    else:
        return False


def get_label_deepfakes(video_name):
    if len(video_name) > 3:
        return True
    else:
        return False


def VCW(values):
    values = np.array(values)
    if np.sum(values >= 0.5) > np.floor(len(values)/2):
        return np.sum(values[values >= 0.5])/np.sum(values >= 0.5)
    else:
        return np.sum(values[values < 0.5])/np.sum(values < 0.5)


def test_on_subset(model, data_set, sub_set, g):
    if sub_set == "test_images":
        with open("./result/" + model + '/' + data_set + '/' + sub_set + '/result.json', "r") as f:
            data = json.load(f)
        if data_set == "Deepfakes":
            check_name = get_label_deepfakes
        else:
            check_name = get_label
        avg_ff = 0
        avg_fr = 0
        avg_rf = 0
        avg_rr = 0
        avg_robust = 0
        biw_ff = 0
        biw_fr = 0
        biw_rf = 0
        biw_rr = 0
        biw_robust = 0
        vcw_ff = 0
        vcw_fr = 0
        vcw_rf = 0
        vcw_rr = 0
        vcw_robust = 0
        for k, v in data.items():
            avg = sum(v) / len(v)
            avg_robust += abs(avg - 0.5)
            biw = g.cal_result(v)
            biw_robust += abs(biw - 0.5)
            vcw = VCW(v)
            vcw_robust += abs(vcw - 0.5)
            if not check_name(k) and avg < 0.5:
                avg_rr += 1
            elif not check_name(k) and avg >= 0.5:
                avg_rf += 1
            elif check_name(k) and avg >= 0.5:
                avg_ff += 1
            elif check_name(k) and avg < 0.5:
                avg_fr += 1
            if not check_name(k) and biw < 0.5:
                biw_rr += 1
            elif not check_name(k) and biw >= 0.5:
                biw_rf += 1
            elif check_name(k) and biw >= 0.5:
                biw_ff += 1
            elif check_name(k) and biw < 0.5:
                biw_fr += 1
            if not check_name(k) and vcw < 0.5:
                vcw_rr += 1
            elif not check_name(k) and vcw >= 0.5:
                vcw_rf += 1
            elif check_name(k) and vcw >= 0.5:
                vcw_ff += 1
            elif check_name(k) and vcw < 0.5:
                vcw_fr += 1

        return {"avg": {'accuracy': (avg_ff + avg_rr) / len(data.keys()), 'ff': avg_ff, 'fr': avg_fr, 'rf': avg_rf,
                        'rr': avg_rr, "robustness": avg_robust / len(data.keys())},
                "biw": {'accuracy': (biw_ff + biw_rr) / len(data.keys()), 'ff': biw_ff, 'fr': biw_fr, 'rf': biw_rf,
                        'rr': biw_rr, "robustness": biw_robust / len(data.keys())},
                "vcw": {'accuracy': (vcw_ff + vcw_rr) / len(data.keys()), 'ff': vcw_ff, 'fr': vcw_fr, 'rf': vcw_rf,
                        'rr': vcw_rr, "robustness": vcw_robust / len(data.keys())}}
    else:
        results = {}
        for rate in os.listdir("./result/" + model + '/' + data_set + '/' + sub_set):
            with open("./result/" + model + '/' + data_set + '/' + sub_set + '/' + rate + '/result.json', "r") as f:
                data = json.load(f)
            if data_set == "Deepfakes":
                check_name = get_label_deepfakes
            else:
                check_name = get_label
            avg_ff = 0
            avg_fr = 0
            avg_rf = 0
            avg_rr = 0
            biw_ff = 0
            biw_fr = 0
            biw_rf = 0
            biw_rr = 0
            vcw_ff = 0
            vcw_fr = 0
            vcw_rf = 0
            vcw_rr = 0
            for k, v in data.items():
                avg = sum(v) / len(v)
                biw = g.cal_result(v)
                vcw = VCW(v)
                if not check_name(k) and avg < 0.5:
                    avg_rr += 1
                elif not check_name(k) and avg >= 0.5:
                    avg_rf += 1
                elif check_name(k) and avg >= 0.5:
                    avg_ff += 1
                elif check_name(k) and avg < 0.5:
                    avg_fr += 1
                if not check_name(k) and biw < 0.5:
                    biw_rr += 1
                elif not check_name(k) and biw >= 0.5:
                    biw_rf += 1
                elif check_name(k) and biw >= 0.5:
                    biw_ff += 1
                elif check_name(k) and biw < 0.5:
                    biw_fr += 1
                if not check_name(k) and vcw < 0.5:
                    vcw_rr += 1
                elif not check_name(k) and vcw >= 0.5:
                    vcw_rf += 1
                elif check_name(k) and vcw >= 0.5:
                    vcw_ff += 1
                elif check_name(k) and vcw < 0.5:
                    vcw_fr += 1
                if (avg - 0.5) * (biw - 0.5) < 0:
                    print(k)
            results[rate] = {
                "avg": {'accuracy': (avg_ff + avg_rr) / len(data.keys()), 'ff': avg_ff, 'fr': avg_fr, 'rf': avg_rf,
                        'rr': avg_rr},
                "biw": {'accuracy': (biw_ff + biw_rr) / len(data.keys()), 'ff': biw_ff, 'fr': biw_fr, 'rf': biw_rf,
                        'rr': biw_rr},
                "vcw": {'accuracy': (vcw_ff + vcw_rr) / len(data.keys()), 'ff': vcw_ff, 'fr': vcw_fr, 'rf': vcw_rf,
                        'rr': vcw_rr}}
        return results


if __name__ == "__main__":
    models = ["Meso4", "MesoInception4", "Xception", "EfficientNet"]
    data_sets = ["Deepfakes", "Face2Face", "FaceSwap",
                 "FaceShifter", "NeuralTextures", "DFDC", "FMFCC"]
    sub_sets = ["test_images", "fl2_test_images",
                "ei2_test_images", 'rn2_test_images', 'ce2_test_images']
    with open("./val_data.json", "r") as f:
        biw_config = json.load(f)
    all_results = {}
    for model in models:
        model_results = {}
        for data_set in data_sets:
            config_info = biw_config[model][data_set]
            data_set_result = {}
            for sub_set in sub_sets:
                if sub_set in ['test_images', 'ei2_test_images', 'rn2_test_images']:
                    biw = BIW(config_info['ff'], config_info['rf'],
                              config_info['fr'], config_info['rr'], 100)
                else:
                    biw = BIW(config_info['ff'], config_info['rf'], config_info['fr'] + int(config_info['rr'] * 0.4),
                              config_info['rr'], 100)
                result = test_on_subset(model, data_set, sub_set, biw)
                data_set_result[sub_set] = result
            model_results[data_set] = data_set_result
        all_results[model] = model_results

    with open('./all_result_with_robustness.json', "w") as f:
        json.dump(all_results, f)
