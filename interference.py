import os
import tqdm
import cv2
import numpy as np
import json
import random
import skimage
import sys
import shutil
import itertools
from tensorflow import keras


def frame_loss_image(fake_image_paths, real_image_paths, output_dir_path, loss_rate):
    images_len = min(len(fake_image_paths), len(real_image_paths))
    n = int(images_len * loss_rate)
    index = np.zeros(images_len)
    choice = random.choices(list(range(images_len)), k=n)
    index[choice] = 1
    for i in range(images_len):
        if index[i] == 0:
            shutil.copy2(fake_image_paths[i], output_dir_path)
        else:
            shutil.copy2(real_image_paths[i], output_dir_path + fake_image_paths[i].split('/')[-1])


def code_error_image(fake_image_paths, real_image_paths, output_dir_path, error_rate):
    for image_path in fake_image_paths:
        img = keras.preprocessing.image.load_img(image_path)
        img_array = keras.preprocessing.image.img_to_array(img)
        s = int(error_rate * img_array.shape[0] * img_array.shape[1])
        x = np.random.randint(0, int(error_rate * img_array.shape[0]))
        y = np.random.randint(0, int(img_array.shape[1] - (s / (img_array.shape[0] - x))))
        h = np.random.randint(int(s / (img_array.shape[1] - y)) + 1, img_array.shape[0] - x)
        w = int(s / h)
        img_array[x:x + h, y:y + w] = [0, 0, 0]
        keras.preprocessing.image.save_img(output_dir_path + image_path.split('/')[-1], img_array)


def random_noise_image(fake_image_paths, real_image_paths, output_dir_path, noise_rate):
    for i, image_path in enumerate(fake_image_paths):
        img = keras.preprocessing.image.load_img(image_path)
        img_array = keras.preprocessing.image.img_to_array(img)
        frame_fake_float = img_array / 255
        frame_fake_noise = skimage.util.random_noise(frame_fake_float, mode='s&p', seed=int(img_array[42][42][0] + i),
                                                     amount=noise_rate)
        frame_fake_int = (frame_fake_noise * 255).astype(np.uint8)
        keras.preprocessing.image.save_img(output_dir_path + image_path.split('/')[-1], frame_fake_int)


def electromagnetic_interference_image(fake_image_paths, real_image_paths, output_dir_path, interference_rate):
    for image_path in fake_image_paths:
        img = keras.preprocessing.image.load_img(image_path)
        img_array = keras.preprocessing.image.img_to_array(img)
        h_sum = int(interference_rate * img_array.shape[0])
        index = np.random.choice(np.arange(0, img_array.shape[0]), h_sum, replace=False)
        img_array[index, :] = [0, 0, 0]
        keras.preprocessing.image.save_img(output_dir_path + image_path.split('/')[-1], img_array)


def frame_loss(fake_video_path, real_video_path, output_dir_path, loss_rate):
    # loss_rate即替换帧的数目占总帧数的比例
    if os.path.exists(output_dir_path + str(loss_rate)[0:3] + "/" + fake_video_path.split("/")[-1]):
        return
    fake = cv2.VideoCapture(fake_video_path)
    real = cv2.VideoCapture(real_video_path)
    real_fps = int(real.get(cv2.CAP_PROP_FPS))
    real_video_width = int(real.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_video_height = int(real.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_frame_count = int(real.get(cv2.CAP_PROP_FRAME_COUNT))
    fake_fps = int(fake.get(cv2.CAP_PROP_FPS))
    fake_video_width = int(fake.get(cv2.CAP_PROP_FRAME_WIDTH))
    fake_video_height = int(fake.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fake_frame_count = int(fake.get(cv2.CAP_PROP_FRAME_COUNT))
    # if real_fps != fake_fps:  # or real_frame_count != fake_frame_count:
    #     raise ValueError("read file is not the pair of fake file!")
    if not os.path.exists(output_dir_path + str(loss_rate)[0:3] + "/"):
        os.makedirs(output_dir_path + str(loss_rate)[0:3] + "/")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_dir_path + str(loss_rate)[0:3] + "/" + fake_video_path.split("/")[-1], fourcc, 1,
                          (fake_video_width, fake_video_height))
    n = int(fake_frame_count * loss_rate)
    index = np.zeros(fake_frame_count)
    choice = random.choices(list(range(fake_frame_count)), k=n)
    index[choice] = 1
    for i in range(fake_frame_count):
        frameId = fake.get(1)
        ret_true, frame_true = real.read()
        ret_fake, frame_fake = fake.read()
        if frameId % ((int(fake_fps) + 1) * 1) == 0:
            if index[i] == 1 and ret_true == True:
                out.write(cv2.resize(frame_true, (fake_video_width, fake_video_height)))
            if index[i] == 0 and ret_fake == True:
                out.write(frame_fake)
    out.release()


def code_error(fake_video_path, real_video_path, output_dir_path, error_rate):
    # error_rate即白块占视频大小的比例
    # 分两步随机选块，第一步随机选择锚点，第二步随机选择宽度
    # 随机选择锚点需要计算可选位置（双曲线）
    # 随机选择宽度需要计算可选范围（双曲线）
    # 设高度为2a，宽度为2b，若白块面积小于ab则所有的点都是可选锚点，因为为了保证所有的点都是可选的，error_rate应小于0.25
    # 为了保证公平性，选择向右下的方向绘制图形，那么
    if os.path.exists(output_dir_path + str(error_rate)[0:4] + "/" + fake_video_path.split("/")[-1]):
        return
    if error_rate > 0.25:
        raise OverflowError("error_rate should be less than 1/4")
    fake = cv2.VideoCapture(fake_video_path)
    fake_fps = int(fake.get(cv2.CAP_PROP_FPS))
    fake_video_width = int(fake.get(cv2.CAP_PROP_FRAME_WIDTH))
    fake_video_height = int(fake.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fake_frame_count = int(fake.get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(output_dir_path + str(error_rate)[0:4] + "/"):
        os.makedirs(output_dir_path + str(error_rate)[0:4] + "/")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_dir_path + str(error_rate)[0:4] + "/" + fake_video_path.split("/")[-1], fourcc, 1,
                          (fake_video_width, fake_video_height))
    s = int(error_rate * fake_video_height * fake_video_width)
    while fake.isOpened():
        frameId = fake.get(1)
        ret_fake, frame_fake = fake.read()
        if not ret_fake:
            break
        if frameId % ((int(fake_fps) + 1) * 1) == 0:
            x = np.random.randint(0, int(error_rate * fake_video_height))
            y = np.random.randint(0, int(fake_video_width - (s / (fake_video_height - x))))
            h = np.random.randint(int(s / (fake_video_width - y)) + 1, fake_video_height - x)
            w = int(s / h)
            frame_fake[x:x + h, y:y + w] = [0, 0, 0]
            out.write(frame_fake)
    out.release()


def random_noise(fake_video_path, real_video_path, output_dir_path, noise_rate):
    # noise_rate即噪声占视频像素点的比例
    if os.path.exists(output_dir_path + str(noise_rate)[0:3] + "/" + fake_video_path.split("/")[-1]):
        return
    fake = cv2.VideoCapture(fake_video_path)
    fake_fps = int(fake.get(cv2.CAP_PROP_FPS))
    fake_video_width = int(fake.get(cv2.CAP_PROP_FRAME_WIDTH))
    fake_video_height = int(fake.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fake_frame_count = int(fake.get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(output_dir_path + str(noise_rate)[0:3] + "/"):
        os.makedirs(output_dir_path + str(noise_rate)[0:3] + "/")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_dir_path + str(noise_rate)[0:3] + "/" + fake_video_path.split("/")[-1], fourcc, 1,
                          (fake_video_width, fake_video_height))
    while fake.isOpened():
        frame_id = fake.get(1)
        ret_fake, frame_fake = fake.read()
        if not ret_fake:
            break
        if frame_id % ((int(fake_fps) + 1) * 1) == 0:
            frame_fake_float = frame_fake / 255

            frame_fake_noise = skimage.util.random_noise(frame_fake_float, mode='s&p', seed=int(frame_id),
                                                         amount=noise_rate)
            frame_fake_int = (frame_fake_noise * 255).astype(np.uint8)
            out.write(frame_fake_int)
    out.release()


def electromagnetic_interference(fake_video_path, real_video_path, output_dir_path, interference_rate):
    # interference_rate即横条的高度总和占视频总高度的比例
    if os.path.exists(output_dir_path + str(interference_rate)[0:3] + "/" + fake_video_path.split("/")[-1]):
        return
    fake = cv2.VideoCapture(fake_video_path)
    fake_fps = int(fake.get(cv2.CAP_PROP_FPS))
    fake_video_width = int(fake.get(cv2.CAP_PROP_FRAME_WIDTH))
    fake_video_height = int(fake.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fake_frame_count = int(fake.get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(output_dir_path + str(interference_rate)[0:3] + "/"):
        os.makedirs(output_dir_path + str(interference_rate)[0:3] + "/")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_dir_path + str(interference_rate)[0:3] + "/" + fake_video_path.split("/")[-1], fourcc,
                          1,
                          (fake_video_width, fake_video_height))
    h_sum = int(interference_rate * fake_video_height)
    while fake.isOpened():
        frameId = fake.get(1)
        ret_fake, frame_fake = fake.read()
        if not ret_fake:
            break
        if frameId % ((int(fake_fps) + 1) * 1) == 0:
            index = np.random.choice(np.arange(0, fake_video_height), h_sum, replace=False)
            frame_fake[index, :] = [0, 0, 0]
            out.write(frame_fake)
    out.release()


def dfdc_original_video(fake_video_path):
    original_video_name = "_".join(fake_video_path.split('/')[-1].split("_")[1:3])
    original_videos = []
    for name in os.listdir("./data/DFDC/train_videos/real/"):
        if original_video_name in name:
            original_videos.append("./data/DFDC/train_videos/real/" + name)
    for name in os.listdir("./data/DFDC/test_videos/real/"):
        if original_video_name in name:
            original_videos.append("./data/DFDC/test_videos/real/" + name)
    fake = cv2.VideoCapture(fake_video_path)
    fake_video_width = int(fake.get(cv2.CAP_PROP_FRAME_WIDTH))
    fake_video_height = int(fake.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret_fake, frame_fake = fake.read()
    measure = np.zeros(len(original_videos))
    for i, original_video in enumerate(original_videos):
        real = cv2.VideoCapture(original_video)
        ret_real, frame_real = real.read()
        measure[i] = np.sum(np.square(cv2.resize(frame_real, (fake_video_width, fake_video_height)) - frame_fake))
    return original_videos[measure.argmin()]


def ff_original_video(fake_video_path):
    original_video_name = fake_video_path.split('/')[-1][0:3] + ".mp4"
    data_path = '/'.join(fake_video_path.split('/')[:-2])
    return data_path + "/real/" + original_video_name


def fmfcc_original_video(fake_video_path):
    original_video_name = fake_video_path.split('/')[-1].split('.')[0]
    original_video = '/'.join(fake_video_path.split('/')[:-2]) + '/real/' + original_video_name + ".pair.mp4"
    return original_video

def fmfcc_original_images(fake_image_path):
    original_images = [path.replace('fake', 'real') for path in fake_image_path]
    return original_images

def ff_original_images(fake_images_path):
    original_images = ['/'.join(p.split("/")[:-2]) + '/real/' + p.split('/')[-1][:3] + p.split('/')[-1][7:] for p in
                       fake_images_path]
    original_images_true = []
    for path in original_images:
        if os.path.exists(path):
            original_images_true.append(path)
    return original_images_true


def video_to_images(video, image_list):
    video_images = []
    n = itertools.count(1)
    for i in n:
        if video[:-4] + '_' + str(i - 1) + ".png" in image_list:
            video_images.append(video[:-4] + '_' + str(i - 1) + ".png")
        else:
            break
    return video_images


if __name__ == "__main__":
    data_set = "FMFCC" #sys.argv[1]
    fake_image_list = os.listdir("./data/%s/test_images/fake/" % data_set)
    fake_video_list = os.listdir("./data/%s/test_videos/fake/" % data_set)

    for rate in np.linspace(0.1, 0.5, 5):
        for video in tqdm.tqdm(fake_video_list):
            fake_images = video_to_images(video, fake_image_list)
            fake_images_path = ["./data/" + data_set + '/test_images/fake/' + image for image in fake_images]
            if data_set in ['Deepfakes', 'Face2Face', 'FaceSwap', 'FaceShifter', 'NeuralTextures']:
                original_images_path = ff_original_images(fake_images_path)
            elif data_set == "DFDC":
                original_video = dfdc_original_video("./data/%s/test_videos/fake/" % data_set + video)
                original_images_path = video_to_images(original_video.split('/')[-1], os.listdir("./data/%s/test_images/real/" % data_set))
                original_images_path = ["./data/%s/test_images/real/" % data_set + name for name in original_images_path]
            elif data_set == "FMFCC":
                original_images_path = fmfcc_original_images(fake_images_path)
            if rate != 0.5:
                if not os.path.exists("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/fake/'):
                    os.makedirs("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/fake/')
                frame_loss_image(fake_images_path, original_images_path, "./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/fake/', rate)
                if not os.path.exists("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/real/'):
                    os.makedirs("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/real/')
                frame_loss_image(original_images_path, fake_images_path, "./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/real/',rate)
            else:
                if not os.path.exists("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/fake/'):
                    os.makedirs("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/fake/')
                frame_loss_image(fake_images_path, original_images_path, "./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/fake/', rate)
                if not os.path.exists("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/real/'):
                    os.makedirs("./data/" + data_set + '/fl2_test_images/' + str(rate)[0:3]  + '/real/')

            if not os.path.exists("./data/" + data_set + '/ce2_test_images/' + str(rate/2.5)[0:4] + '/fake/'):
                os.makedirs("./data/" + data_set + '/ce2_test_images/' + str(rate/2.5)[0:4] + '/fake/')
            code_error_image(fake_images_path, original_images_path,
                             "./data/" + data_set + '/ce2_test_images/' + str(rate/2.5)[0:4] + '/fake/', rate/2.5)
            if not os.path.exists("./data/" + data_set + '/ce2_test_images/' + str(rate/2.5)[0:4] + '/real/'):
                os.makedirs("./data/" + data_set + '/ce2_test_images/' + str(rate/2.5)[0:4] + '/real/')
            code_error_image(original_images_path, fake_images_path,
                             "./data/" + data_set + '/ce2_test_images/' + str(rate/2.5)[0:4] + '/real/', rate/2.5)

            if not os.path.exists("./data/" + data_set + '/rn2_test_images/' + str(rate/5)[0:4] + '/fake/'):
                os.makedirs("./data/" + data_set + '/rn2_test_images/' + str(rate/5)[0:4] + '/fake/')
            random_noise_image(fake_images_path, original_images_path,
                             "./data/" + data_set + '/rn2_test_images/' + str(rate/5)[0:4] + '/fake/', rate/5)
            if not os.path.exists("./data/" + data_set + '/rn2_test_images/' + str(rate/5)[0:4] + '/real/'):
                os.makedirs("./data/" + data_set + '/rn2_test_images/' + str(rate/5)[0:4] + '/real/')
            random_noise_image(original_images_path, fake_images_path,
                             "./data/" + data_set + '/rn2_test_images/' + str(rate/5)[0:4] + '/real/', rate/5)

            if not os.path.exists("./data/" + data_set + '/ei2_test_images/' + str(rate/2.5)[0:4] + '/fake/'):
                os.makedirs("./data/" + data_set + '/ei2_test_images/' + str(rate/2.5)[0:4] + '/fake/')
            electromagnetic_interference_image(fake_images_path, original_images_path,
                             "./data/" + data_set + '/ei2_test_images/' + str(rate/2.5)[0:4] + '/fake/', rate/2.5)
            if not os.path.exists("./data/" + data_set + '/ei2_test_images/' + str(rate/2.5)[0:4] + '/real/'):
                os.makedirs("./data/" + data_set + '/ei2_test_images/' + str(rate/2.5)[0:4] + '/real/')
            electromagnetic_interference_image(original_images_path, fake_images_path,
                             "./data/" + data_set + '/ei2_test_images/' + str(rate/2.5)[0:4] + '/real/', rate/2.5)

    # data_set = 'FMFCC'# sys.argv[1]
    # for fake_video in tqdm.tqdm(os.listdir("./data/%s/test_videos/fake/" % data_set)):
    #     original_video = fmfcc_original_video(("./data/%s/test_videos/fake/" % data_set) + fake_video)
    #     for i in np.linspace(0.1,0.5,5):
    #         if i != 0.5:
    #             frame_loss(original_video, ("./data/%s/test_videos/fake/" % data_set) + fake_video, "./data/%s/fl_test_videos/" % data_set, i)
    #         code_error(original_video, original_video, "./data/%s/ce_test_videos/" % data_set, i/2)
    #         random_noise(original_video, original_video, "./data/%s/rn_test_videos/" % data_set, i)
    #         electromagnetic_interference(original_video, original_video, "./data/%s/ei_test_videos/" % data_set, i)
    # test_sets = ['fl_test_videos', 'ce_test_videos', 'ei_test_videos', 'rn_test_videos']
    #
    # for test_set in test_sets:
    #     for rate in os.listdir("./data/" + data_set + '/' + test_set):
    #         if not os.path.exists("./data/" + data_set + '/' + test_set + '/' + rate + '/real/'):
    #             os.makedirs("./data/" + data_set + '/' + test_set + '/' + rate + '/real/')
    #         for file in os.listdir("./data/" + data_set + '/' + test_set + '/' + rate):
    #             if ".mp4" in file:
    #                 shutil.move("./data/" + data_set + '/' + test_set + '/' + rate + '/' + file,
    #                             "./data/" + data_set + '/' + test_set + '/' + rate + '/real/')
    #
    # for fake_video in tqdm.tqdm(os.listdir("./data/%s/test_videos/fake/" % data_set)):
    #     original_video = fmfcc_original_video(("./data/%s/test_videos/fake/" % data_set) + fake_video)
    #     for i in np.linspace(0.1,0.5,5):
    #         frame_loss(("./data/%s/test_videos/fake/" % data_set) + fake_video, original_video, "./data/%s/fl_test_videos/" % data_set, i)
    #         code_error(("./data/%s/test_videos/fake/" % data_set) + fake_video, original_video, "./data/%s/ce_test_videos/" % data_set, i / 2)
    #         random_noise(("./data/%s/test_videos/fake/" % data_set) + fake_video, original_video, "./data/%s/rn_test_videos/" % data_set, i)
    #         electromagnetic_interference(("./data/%s/test_videos/fake/" % data_set) + fake_video, original_video, "./data/%s/ei_test_videos/" % data_set, i)
    # for test_set in test_sets:
    #     for rate in os.listdir("./data/" + data_set + '/' + test_set):
    #         if not os.path.exists("./data/" + data_set + '/' + test_set + '/' + rate + '/fake/'):
    #             os.makedirs("./data/" + data_set + '/' + test_set + '/' + rate + '/fake/')
    #         for file in os.listdir("./data/" + data_set + '/' + test_set + '/' + rate):
    #             if ".mp4" in file:
    #                 shutil.move("./data/" + data_set + '/' + test_set + '/' + rate + '/' + file,
    #                             "./data/" + data_set + '/' + test_set + '/' + rate + '/fake/')
