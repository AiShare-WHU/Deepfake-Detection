import os
import itertools
import numpy as np
import tensorflow as tf
import tqdm


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


def detect_video(model, video, image_list, data_path, target_size):
    video_images = video_to_images(video, image_list)
    images = load_images(video_images, data_path, target_size)
    results = model.predict(images)
    result = results[:, 0].mean()
    if result > 0.5:
        return True, results[:, 0].tolist()
    else:
        return False, results[:, 0].tolist()


def get_videos(data_path, data_set, sub_set, label):
    data_path = data_path + data_set + '/' + sub_set + '/' + label + '/'
    images = os.listdir(data_path)
    videos = list(set(['_'.join(image.split('_')[:-1]) for image in images]))
    return videos


def test_on_subset(model, data_path, data_set, sub_set, target_size):
    real_path = data_path + data_set + '/' + sub_set + '/real/'
    real = os.listdir(real_path)
    videos = list(set(['_'.join(image.split('_')[:-1]) for image in real]))
    real_real = 0
    real_fake = 0
    video_results = {}
    for video in tqdm.tqdm(videos):
        r, result = detect_video(model, video, real, real_path, target_size)
        video_results[video + "_real"] = result
        if r:
            real_fake += 1
        else:
            real_real += 1
    fake_path = data_path + data_set + '/' + sub_set + '/fake/'
    fake = os.listdir(fake_path)
    videos = list(set(['_'.join(image.split('_')[:-1]) for image in fake]))
    fake_real = 0
    fake_fake = 0
    for video in tqdm.tqdm(videos):
        r, result = detect_video(model, video, fake, fake_path, target_size)
        video_results[video + "_fake"] = result
        if r:
            fake_fake += 1
        else:
            fake_real += 1
    return fake_fake, fake_real, real_fake, real_real, video_results


if __name__ == "__main__":
    data_path = "./data/Deepfakes/train_images/real/"
    real = os.listdir(data_path)
    videos = list(set([image[:3] for image in real]))
    model = tf.keras.models.load_model("./saved_data/Deepfakes/Xception/model_2.hdf5")
    count = 0
    for video in tqdm.tqdm(videos):
        r = detect_video(model, video, real, data_path)
        if r:
            count += 1
    print(count)
