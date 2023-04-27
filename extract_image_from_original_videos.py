import os
from os.path import join
from tqdm import tqdm
import shutil
import json
import cv2

with open("./original_videos/conversion_dict.json", "r") as f:
    convert_dict = json.load(f)

for k, v in tqdm(convert_dict.items()):
    file_name = v[:11]
    frame_sequences = v[12:13]
    file_path = "./original_videos/downloaded_videos/" + file_name + "/" + file_name + ".mp4"
    frame_sequences = "./original_videos/downloaded_videos/" + file_name + "/" + "extracted_sequences/" + frame_sequences + ".json"
    write_path = "./original_videos/converted_videos/" + k + ".mp4"
    cap = cv2.VideoCapture(file_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frameRate = cap.get(5)
    width = int(cap.get(3))
    hight = int(cap.get(4))
    writer = cv2.VideoWriter(write_path, fourcc, frameRate, (width, hight))
    imageid = 0
    with open(frame_sequences, "r") as f:
        sequences = json.load(f)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId in sequences:
            writer.write(frame)
    writer.release()
