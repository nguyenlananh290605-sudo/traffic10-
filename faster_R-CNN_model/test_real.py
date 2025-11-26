import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
import matplotlib.patches as patches

# ==== Label mapping ====
conversion_label = {1:'bicycle', 2:'bus', 3:'car', 4:'motorbike', 5:'person'}

# ==== Transform ====
def get_transform():
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# ==== Load image ====
img_path = r"D:\traffic\kanggle\test1.jpg"  # Đường dẫn ảnh bạn muốn test
output_folder = r"kanggle"
os.makedirs(output_folder, exist_ok=True)
img_name = os.path.basename(img_path)

img = read_image(img_path)
img = torchvision.transforms.Resize(224)(img)
img = tv_tensors.Image(img)
img = get_transform()(img)

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

# ==== Visualization and save ====
def tensorToPIL(img):
    return torchvision.transforms.ToPILImage()(img).convert('RGB')

def plot_img_bbox_and_save(img, prediction, save_path):
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(6,6)
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
    print(f"Saved result to: {save_path}, Total boxes: {len(boxes)}")

# ==== Run inference ====
with torch.no_grad():
    prediction = model([img.to(device)])[0]
nms_pred = apply_nms(prediction, 0.3)
save_path = os.path.join(output_folder, img_name)
plot_img_bbox_and_save(img, nms_pred, save_path)
