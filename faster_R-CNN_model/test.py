import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import matplotlib.patches as patches

# ==== Label mapping ====
conversion_label = {1:'bicycle', 2:'bus', 3:'car', 4:'motorbike', 5:'person'}

# ==== Dataset class for test (no labels) ====
class TrafficTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_files = sorted(os.listdir(root))
        self.width, self.height = 224, 224

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_files[idx])
        img = read_image(img_path)
        img = torchvision.transforms.Resize(224)(img)
        img = tv_tensors.Image(img)
        if self.transforms:
            img = self.transforms(img)
        return img, self.img_files[idx]

    def __len__(self):
        return len(self.img_files)

# ==== Transform ====
def get_transform():
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# ==== Load test dataset ====
test_root = 'D:/traffic/traffic_data/traffic_data/test/images'  # folder ảnh test
output_root = 'D:/traffic/kanggle/kanggle/output'  # folder lưu kết quả
os.makedirs(output_root, exist_ok=True)

dataset_test = TrafficTestDataset(test_root, get_transform())

# ==== Load model ====
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 6
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("pytorch_model.pt", map_location=device))
model.to(device)
model.eval()

# ==== NMS function ====
def apply_nms(prediction, threshold=0.3):
    keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], threshold)
    final_prediction = prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    return final_prediction

# ==== Visualization and save image ====
def tensorToPIL(img):
    return torchvision.transforms.ToPILImage()(img).convert('RGB')

def plot_img_bbox_and_save(img, prediction, save_path):
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    if isinstance(img, torch.Tensor):
        img = tensorToPIL(img)
    a.imshow(img)
    boxes = prediction['boxes'].tolist()
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()
    for i, box in enumerate(boxes):
        x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
        a.add_patch(rect)
        a.text(x, y, f"{conversion_label[labels[i]]}: {scores[i]:.2f}", color='red', fontsize=8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path} (Total boxes: {len(boxes)})")

# ==== Run test on all images and save ====
for idx in range(len(dataset_test)):
    img, img_name = dataset_test[idx]
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
    nms_pred = apply_nms(prediction, 0.3)
    save_path = os.path.join(output_root, img_name)
    plot_img_bbox_and_save(img, nms_pred, save_path)
