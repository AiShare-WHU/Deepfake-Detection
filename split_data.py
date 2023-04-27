import os
import shutil
import json

data_set_list = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

with open("./splits/train.json") as f:
    train_split = json.load(f)

with open("./splits/test.json") as f:
    test_split = json.load(f)

with open("./splits/val.json") as f:
    val_split = json.load(f)

for name in data_set_list:
    if not os.path.exists("./data/%s/train_videos/fake/" % name):
        os.makedirs("./data/%s/train_videos/fake/" % name)

    if not os.path.exists("./data/%s/train_videos/real/" % name):
        os.makedirs("./data/%s/train_videos/real/" % name)

    if not os.path.exists("./data/%s/test_videos/fake/" % name):
        os.makedirs("./data/%s/test_videos/fake/" % name)

    if not os.path.exists("./data/%s/test_videos/real/" % name):
        os.makedirs("./data/%s/test_videos/real/" % name)

    if not os.path.exists("./data/%s/val_videos/fake/" % name):
        os.makedirs("./data/%s/val_videos/fake/" % name)

    if not os.path.exists("./data/%s/val_videos/real/" % name):
        os.makedirs("./data/%s/val_videos/real/" % name)

    data_path = "./manipulated_videos/%s/manipulated_sequences/%s/raw/videos/" % (name, name)
    for split in train_split:
        fake_path = [data_path + split[0] + "_" + split[1] + ".mp4", data_path + split[1] + "_" + split[0] + ".mp4"]
        real_path = ["./original_videos/converted_videos/" + split[0] + ".mp4", "./original_videos/converted_videos/" + split[1] + ".mp4"]
        for fake in fake_path:
            shutil.move(fake, "./data/%s/train_videos/fake/" % name)
        for real in real_path:
            if not os.path.exists("./data/%s/train_videos/real/" % name + real.split("/")[-1]):
                shutil.copy2(real, "./data/%s/train_videos/real/" % name)
    for split in val_split:
        fake_path = [data_path + split[0] + "_" + split[1] + ".mp4", data_path + split[1] + "_" + split[0] + ".mp4"]
        real_path = ["./original_videos/converted_videos/" + split[0] + ".mp4",
                     "./original_videos/converted_videos/" + split[1] + ".mp4"]
        for fake in fake_path:
            shutil.move(fake, "./data/%s/val_videos/fake/" % name)
        for real in real_path:
            if not os.path.exists("./data/%s/val_videos/real/" % name + real.split("/")[-1]):
                shutil.copy2(real, "./data/%s/val_videos/real/" % name)
    for split in test_split:
        fake_path = [data_path + split[0] + "_" + split[1] + ".mp4", data_path + split[1] + "_" + split[0] + ".mp4"]
        real_path = ["./original_videos/converted_videos/" + split[0] + ".mp4",
                     "./original_videos/converted_videos/" + split[1] + ".mp4"]
        for fake in fake_path:
            shutil.move(fake, "./data/%s/test_videos/fake/" % name)
        for real in real_path:
            if not os.path.exists("./data/%s/test_videos/real/" % name + real.split("/")[-1]):
                shutil.copy2(real, "./data/%s/test_videos/real/" % name)


