from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from albumentations.pytorch import ToTensorV2
import cv2


class TrafficDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = np.array(Image.open(img_path).convert("RGB"))
        img = enhance_image_np(img)  # <-- Add this line for CLAHE + gamma
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, x_c, y_c, w, h = map(float, line.split())
                    h_img, w_img = img.shape[:2]
                    x_c, y_c = x_c * w_img, y_c * h_img
                    w, h = w * w_img, h * h_img
                    x0, y0 = x_c - w/2, y_c - h/2
                    x1, y1 = x_c + w/2, y_c + h/2
                    boxes.append([x0, y0, x1, y1])
                    labels.append(int(cls) + 1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.numel() > 0:
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        if self.transforms:
            labs = labels.tolist()
            transformed = self.transforms(image=img, bboxes=target["boxes"], labels=labs)
            img = transformed["image"]
            target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        else:
            img = ToTensorV2()(image=img)["image"]
        return img, target

def enhance_image_np(img, gamma=1.2, clipLimit=2.0, tileGridSize=(8,8)):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    invGamma = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** invGamma * 255).astype('uint8')
    img_gamma = cv2.LUT(img_eq, table)
    img_rgb = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB)
    return img_rgb