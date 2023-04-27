import os
import itertools

import keras.layers
import numpy as np
import tensorflow as tf
import tqdm
import time
import timeit
import json
from weighting import BIW


def video_to_images(video, image_list):
    video_images = []
    n = itertools.count(1)
    for i in n:
        if video + '_' + str(i - 1) + ".png" in image_list:
            video_images.append(video + '_' + str(i - 1) + ".png")
        else:
            break
    return video_images


def load_images(image_list, data_path, target_size):
    images = np.zeros(shape=(len(image_list), target_size[0], target_size[1], 3))
    for i, image in enumerate(image_list):
        img = tf.keras.preprocessing.image.load_img(data_path + image, target_size=target_size)
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        images[i] = input_arr
    return images


def get_videos(data_path, data_set, sub_set, label):
    data_path = data_path + data_set + '/' + sub_set + '/' + label + '/'
    images = os.listdir(data_path)
    videos = list(set(['_'.join(image.split('_')[:-1]) for image in images]))
    return videos


def detect_video(model, video, image_list, data_path, target_size):
    video_images = video_to_images(video, image_list)
    images = load_images(video_images, data_path, target_size)
    results = model.predict(images)
    result = results[:, 0].mean()
    if result > 0.5:
        return True, results[:, 0].tolist()
    else:
        return False, results[:, 0].tolist()


def test_on_subset(model, biw, data_path, data_set, sub_set, target_size):
    real_path = data_path + data_set + '/' + sub_set + '/real/'
    real = os.listdir(real_path)
    videos = list(set(['_'.join(image.split('_')[:-1]) for image in real]))
    video_results = {}
    for video in tqdm.tqdm(videos):
        video_images = video_to_images(video, real)
        images = load_images(video_images, real_path, target_size)
        pre_start = time.time()
        results = model.predict(images)
        pre_end = time.time()
        avg_start = time.time()
        results[:, 0].mean()
        avg_end = time.time()
        biw_start = time.time()
        biw.cal_result(results[:, 0])
        biw_end = time.time()
        video_results[video + "_real"] = {"avg": avg_end - avg_start, "biw": biw_end - biw_start,
                                          "pre": pre_end - pre_start}
    fake_path = data_path + data_set + '/' + sub_set + '/fake/'
    fake = os.listdir(fake_path)
    videos = list(set(['_'.join(image.split('_')[:-1]) for image in fake]))
    for video in tqdm.tqdm(videos):
        video_images = video_to_images(video, fake)
        images = load_images(video_images, fake_path, target_size)
        pre_start = time.time()
        results = model.predict(images)
        pre_end = time.time()
        avg_start = time.time()
        results[:, 0].mean()
        avg_end = time.time()
        biw_start = time.time()
        biw.cal_result(results[:, 0])
        biw_end = time.time()
        video_results[video + "_fake"] = {"avg": avg_end - avg_start, "biw": biw_end - biw_start,
                                          "pre": pre_end - pre_start}
    return video_results


def run_on_dataset(data_set, model_name, model, config_info, target_size):
    sub_set = "test_images"
    biw = BIW(config_info['ff'], config_info['rf'], config_info['fr'], config_info['rr'], 100)
    data_path = "./data/"
    video_results = test_on_subset(model, biw, data_path, data_set, sub_set, target_size)
    count = 0
    time_sum = 0
    time_biw = 0
    perception = 0
    for k, v in video_results.items():
        count += 1
        time_sum += v["pre"] + v["biw"]
        time_biw += v["biw"]
        perception += time_biw / time_sum
    return perception / count, time_sum/count


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    with open("./val_data.json", "r") as f:
        biw_config = json.load(f)
    data_sets = ["FaceShifter", "NeuralTextures", "DFDC", "FMFCC"]
    sub_set = "test_images"
    model_names = ["Meso4", "MesoInception4", "Xception", "EfficientNet"]
    iteration = 10
    all_result = {}
    for data_set in data_sets:
        all_result[data_set] = {}
        for model_name in model_names:
            config_info = biw_config[model_name][data_set]
            if model_name == "EfficientNet" and data_set == "NeuralTextures":
                model = tf.keras.models.load_model("./saved_data/%s/%s/" % (data_set, model_name))
            else:
                model = tf.keras.models.load_model("./saved_data/%s/%s/model_2.hdf5" % (data_set, model_name))
            if model_name == "EfficientNet" or model_name == "Xception":
                target_size = (128, 128, 3)
            else:
                target_size = (256, 256, 3)
            result = [0]*iteration
            runtimes = 0
            for i in range(iteration):
                result[i], runtime = run_on_dataset(data_set, model_name, model, config_info, target_size)
                runtimes += runtime
            all_result[data_set][model_name] = result
            print("%s, %s, mean %s, max %s, min %s, runtime %s" % (data_set, model_name, str(sum(result)/iteration),
                                                       str(max(result)), str(min(result)), str(runtimes/iteration)))
    with open("./efficiency.json", "r") as f:
        json.dump(all_result, f)
