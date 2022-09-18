import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
import os

import warnings
warnings.filterwarnings("ignore")

class ShapesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 7  # 修改成你的class數量 + 1
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # the arhcors
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)

    TRAIN_ROIS_PER_IMAGE = 100
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50



class DrugDataset(utils.Dataset):
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        # Add classes
        self.add_class("shapes", 1, "hamburger")
        self.add_class("shapes", 2, "french_fries")
        self.add_class("shapes", 3, "friedfood")
        self.add_class("shapes", 4, "dog")
        self.add_class("shapes", 5, "cat")
        self.add_class("shapes", 6, "animal_eye")
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i], width=cv_img.shape[1],
                           height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            # 這邊填入class的名稱(範例為cat)
            if labels[i].find("hamburger") != -1:
                labels_form.append("hamburger")
            elif labels[i].find("french_fries") != -1:
                labels_form.append("french_fries")
            elif labels[i].find("friedfood") != -1:
                labels_form.append("friedfood")
            elif labels[i].find("dog") != -1:
                labels_form.append("dog")
            elif labels[i].find("cat") != -1:
                labels_form.append("cat")
            elif labels[i].find("animal_eye") != -1:
                labels_form.append("animal_eye")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

    def get_ax(rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax



def train():
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "model", "mask_RCNN")
    iter_num = 0
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model", "mask_RCNN", "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = ShapesConfig()
    config.display()

    # 基礎設置
    dataset_root_path = "dataset/mask_RCNN/"  # 設定自己的位置
    img_floder = dataset_root_path + "pic"
    mask_floder = dataset_root_path + "cv2_mask"
    imglist = os.listdir(img_floder)
    count = len(imglist)
    # train與val數據集準備
    print('preparing dataset')
    dataset_train = DrugDataset()
    dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_train.prepare()
    dataset_val = DrugDataset()
    dataset_val.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_val.prepare()
    print('preparing model')
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    # Which weights to start with?
    init_with = "coco"
    if init_with == "imagenet":

        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # Load the last model you trained and continue training
    elif init_with == "last":
        model.load_weights(model.find_last()[1], by_name=True)
    # Train the head branches
    print('start training')
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')
    # Fine tune all layers
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=10, layers="all")

if __name__=='__main__':
    train()