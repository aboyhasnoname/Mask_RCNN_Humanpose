# coding: utf-8
import json
import skimage
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
#import visualize
from model import log
dataset_dir = './data'
annotations = "via_region_data.json"

class MosquitoesConfig(coco.Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mosquitoes"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    #GPU_COUNT = 2

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes
    NUM_CLASSES = 1 + 2  # Person and background

    NUM_KEYPOINTS = 3
    MINI_MASK_SHAPE = [256, 256]
    MASK_SHAPE = [28, 28]
    KEYPOINT_MASK_SHAPE = [128,128]
    # DETECTION_MAX_INSTANCES = 50
    TRAIN_ROIS_PER_IMAGE = 50
    MAX_GT_INSTANCES = 128
    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    USE_MINI_MASK = True
    MASK_POOL_SIZE = 14
    KEYPOINT_MASK_POOL_SIZE = 7
    LEARNING_RATE = 0.002
    STEPS_PER_EPOCH = 10
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005

config = MosquitoesConfig()
config.display()


class MosquitoesDataset(utils.Dataset):
    def __init__(self):
        super().__init__(MosquitoesDataset)
        num_classes = 3
        self.task_type = "person_keypoints"
        # the connection between 2 close keypoints
        self._skeleton = []
        # keypoint names
        # ["prob","head","tail"]
        self._keypoint_names = []

    def load_dataset(self, dataset_dir, subset='train'):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
#         self.add_class("balloon", 1, "balloon")

#         # Train or validation dataset?
        assert subset in ["train", "val"]
#         dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        if subset=='train':
             annotations = json.load(open(os.path.join(dataset_dir, "annotations","train.json")))
        else:
             annotations = json.load(open(os.path.join(dataset_dir, "annotations","val.json")))

        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            image_path = os.path.join(dataset_dir, "images", a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            num_mosquitoes = 0
            cl = list()
            bb = list()
            kp = list()
            for index, attr in enumerate(a['regions']):
                #print(index,attr)
                if (index%5==0):
                    #attr['region_attributes'].setdefault('class','1')# if the 'class' is missing, fill out automatically, but this error seldom happens
                    if not 'class' in attr['region_attributes']:
                        attr['region_attributes'].setdefault('class','1')# if the 'class' is missing, fill out automatically, but this error seldom happens
                    if not attr['region_attributes']['class']:#if the va;ues of 'class' is missing, fill out.
                        cl.append(str(random.randint(1,2)))
                    if attr['region_attributes']['class']:
                        cl.append(int(attr['region_attributes']['class']))
                    if 'y' in attr['shape_attributes']:# eror: if a extral point is marked without sense
                        bb.append([attr['shape_attributes']['y'], attr['shape_attributes']['x'], attr['shape_attributes']['height'], attr['shape_attributes']['width']])

                elif (index%5==1):
                    pass
                else:
                    kp.append(( attr['shape_attributes']['cy'], attr['shape_attributes']['cx']))

                num_mosquitoes += 1

            num_mosquitoes = int(num_mosquitoes/5)

            self.add_image(
                "mosquitoes",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                cl = cl,
                bounding_box = bb,
                key_points = kp,
                num_mosquitoes = num_mosquitoes
                )

    def load_bbox(self, image_id):
        bounding_box = self.image_info[image_id]['bounding_box']
        num_mosquitoes = self.image_info[image_id]['num_mosquitoes']
        bounding_box = np.reshape(bounding_box, (-1,4))
        bounding_box[:,2] += bounding_box[:,0]
        bounding_box[:,3] += bounding_box[:,1]
        return bounding_box

    def load_image(self, image_id):
        image_path = self.image_info[image_id]['path']
        image = skimage.io.imread(image_path)
        return image

    def load_keypoints(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        """
        # If not a COCO image, delegate to parent class.
#         print(self.image_info)
#         image_info = self.image_info[image_id]
#         if image_info["source"] != "coco":
#             return super(CocoDataset, self).load_mask(image_id)

        keypoints = []
        class_ids = []
        instance_masks = []
        info = self.image_info[image_id]
        num_mosquitoes = info['num_mosquitoes']
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for index in range(0, int(info['num_mosquitoes'])):
            class_id = info['cl'][index]

            m = np.zeros((info['height'], info['width']), dtype=np.uint8)
            # generate masks
            for m_index in range(0,3):
                #m_index = index*3 + m_index
                #m = np.zeros((info['height'], info['width']), dtype=np.uint8)
                x = info['key_points'][m_index][0]
                y = info['key_points'][m_index][1]
                m[x,y] = 255
            instance_masks.append(m)
            #load keypoints
            keypoints = info["key_points"]
            keypoints = np.reshape(keypoints,(-1,2))
            new_col = np.ones((keypoints.shape[0],1))+1
            keypoints = np.hstack((keypoints, new_col))
            keypoints = np.reshape(keypoints, (num_mosquitoes,3,3))
#             keypoints.append(keypoint)
            class_ids.append(class_id)
        # Pack instance masks into an array
#         if class_ids:
        keypoints = np.array(keypoints,dtype=np.int32)
        class_ids = np.array(class_ids, dtype=np.int32)
        masks = np.stack(instance_masks, axis=2)

        return keypoints, masks, class_ids
#         else:
#             # Call super class to return an empty mask
#             return super(CocoDataset, self).load_keypoints(image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


if __name__=='__main__':

    train_dataset_keypoints = MosquitoesDataset()
    train_dataset_keypoints.load_dataset(dataset_dir, "train")
    train_dataset_keypoints.prepare()

    val_dataset_keypoints = MosquitoesDataset()
    val_dataset_keypoints.load_dataset(dataset_dir, "val")
    val_dataset_keypoints.prepare()

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    config = MosquitoesConfig()
# Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Create model object in inference mode.
    model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
    #model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    #print("Loading weights from ", COCO_MODEL_PATH)

# Training - Stage 1
    print("Train heads")
    model.train(train_dataset_keypoints, val_dataset_keypoints,\
            learning_rate=config.LEARNING_RATE,\
            epochs=15,\
            layers="all")
