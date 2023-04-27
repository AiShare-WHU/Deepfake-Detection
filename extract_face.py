import cv2
import dlib
import os
import sys
import json
from tqdm import tqdm


def mkdir(data_set):
    if not os.path.exists("./data/%s/train_images/fake/" % data_set):
        os.makedirs("./data/%s/train_images/fake/" % data_set)

    if not os.path.exists("./data/%s/train_images/real/" % data_set):
        os.makedirs("./data/%s/train_images/real/" % data_set)

    if not os.path.exists("./data/%s/test_images/fake/" % data_set):
        os.makedirs("./data/%s/test_images/fake/" % data_set)

    if not os.path.exists("./data/%s/test_images/real/" % data_set):
        os.makedirs("./data/%s/test_images/real/" % data_set)

    if not os.path.exists("./data/%s/val_images/fake/" % data_set):
        os.makedirs("./data/%s/val_images/fake/" % data_set)

    if not os.path.exists("./data/%s/val_images/real/" % data_set):
        os.makedirs("./data/%s/val_images/real/" % data_set)


def extract_face(video_folder, image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    list_of_train_data = [f for f in os.listdir(
        video_folder) if f.endswith('.mp4')]
    detector = dlib.get_frontal_face_detector()
    face_boxes = {}
    for vid in tqdm(list_of_train_data):
        count = 0
        cap = cv2.VideoCapture(os.path.join(video_folder, vid))
        frameRate = cap.get(5)
        face_box = []
        while cap.isOpened():
            frameId = cap.get(1)
            ret, frame = cap.read()
            if ret != True:
                break
            if frameId % ((int(frameRate) + 1) * 1) == 0:
                face_rects, scores, idx = detector.run(frame, 0)
                image_box = []
                for i, d in enumerate(face_rects):
                    x1 = d.left()
                    y1 = d.top()
                    x2 = d.right()
                    y2 = d.bottom()
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        continue
                    image_box.append([x1, y1, x2, y2])
                    cv2.imwrite(image_folder + vid.split('.')[0] + '_' + str(count) + '.png',
                               cv2.resize(crop_img, (256, 256)))
                    count += 1
                face_box.append(image_box)
        face_boxes[vid] = face_box
    return face_boxes


def extract_attacked_face(video_folder, image_folder, box_path):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    with open(box_path, "r") as f:
        boxes = json.load(f)
    list_of_videos = [f for f in os.listdir(
        video_folder) if f.endswith('.mp4')]
    for video in tqdm(list_of_videos):
        face_boxes = boxes.get(video)
        if not face_boxes:
            continue
        count = 0
        face_count = 0
        cap = cv2.VideoCapture(os.path.join(video_folder, video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            if count == len(face_boxes):
                break
            if not ret:
                break
            for face in face_boxes[count]:
                crop_img = frame[face[1]:face[3], face[0]:face[2]]
                if crop_img.size == 0:
                    continue
                cv2.imwrite(image_folder + video.split('.')[0] + '_' + str(face_count) + '.png')
                face_count += 1
            count += 1


if __name__ == "__main__":
    data_set = sys.argv[1]
    # attacks = ['fl_test', 'rn_test', 'ei_test', 'ce_test']
    # for attack in attacks:
    #     for rate in os.listdir("./data/%s/%s_videos/" % (data_set, attack)):
    #         extract_attacked_face("./data/%s/%s_videos/%s/%s/" % (data_set, attack, rate, 'fake'),
    #                               "./data/%s/%s_images/%s/%s/" % (data_set, attack, rate, 'fake'),
    #                               "./data/%s/test_videos/%s.json" % (data_set, 'fake'))
    #         if attack == 'fl_test' and rate == '0.5':
    #             continue
    #         extract_attacked_face("./data/%s/%s_videos/%s/%s/" % (data_set, attack, rate, 'real'),
    #                               "./data/%s/%s_images/%s/%s/" % (data_set, attack, rate, 'real'),
    #                               "./data/%s/test_videos/%s.json" % (data_set, 'real'))
    split = sys.argv[2]
    label = sys.argv[3]

    face_boxes = extract_face("./data/%s/%s_videos/%s/" % (data_set, split, label), "./data/%s/%s_images/%s/" % (data_set, split, label))
    with open("./data/%s/%s_videos/%s.json" % (data_set, split, label), "w") as f:
        json.dump(face_boxes, f)
