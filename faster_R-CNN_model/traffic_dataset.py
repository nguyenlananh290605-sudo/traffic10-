import os
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision
from matplotlib import patches, text, patheffects
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, indexes):
        self.root = root
        self.transforms = transforms
        self.img = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))
        self.indexes = indexes
        self.width, self.height = 224, 224

    def get_valid_index(self):
        index = []
        count = 0
        for filename in self.labels:
            if os.stat(os.path.join(self.root, "labels", filename)).st_size == 0:
                pass
            else:
                index.append(count)
            count = count + 1
        return index

    def getBBFromFile(self, labelFile):
        file1 = open(labelFile, 'r')
        lines = file1.readlines()
        finalList = []
        classes = []
        for line in lines:
            lineList = line.split()
            lineList = [float(i) for i in lineList]
            finalList.append(lineList[1:])
            classes.append(lineList[0] + 1)
        return finalList, classes

    def yolobbox2bbox(self, x, y, w, h):
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
        return [x1*640, y1*640, x2*640, y2*640]

    def scale(self, bb_list):
        factor = 224.0 / 640.0
        x1, y1 = bb_list[0]*factor, bb_list[1]*factor
        x2, y2 = bb_list[2]*factor, bb_list[3]*factor
        return [x1 , y1, x2, y2]

    def __getitem__(self, idx):
        if idx in self.indexes:
            pass
        else:
            return None

        img_path = os.path.join(self.root, "images", self.img[idx])
        labels_path = os.path.join(self.root, "labels", self.labels[idx])
        img = read_image(img_path)
        resize = torchvision.transforms.Resize(224)
        img = resize(img)
        img = tv_tensors.Image(img)
        boxes, classes = self.getBBFromFile(labels_path)
        if len(classes) > 0:
            boxes = [self.scale(self.yolobbox2bbox(box[0],box[1],box[2],box[3])) for box in boxes]
            area = [box[2] * box[3] for box in boxes]
            iscrowd = torch.zeros((len(classes),), dtype=torch.int64)
            target = {}
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img), dtype=torch.float32)
            target["labels"] = torch.as_tensor(classes, dtype=torch.int64)
            target["image_id"] = self.img[idx]
            target["area"] = torch.as_tensor(area)
            target["iscrowd"] = iscrowd
        else:
            return None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img)