import io
import random
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

app = FastAPI()

from torchvision.models.detection import fasterrcnn_resnet50_fpn

NUM_CLASSES = 6
model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES)

state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

CLASS_NAMES = ['background', 'bicycle', 'bus', 'car', 'motorbike', 'person']

# Giữ nguyên nhưng KHÔNG dùng nữa cho màu box/text
COLORS = {}
def get_class_color(class_name):
    if class_name not in COLORS:
        COLORS[class_name] = (
            random.randint(80, 200),
            random.randint(80, 200),
            random.randint(80, 200),
        )
    return COLORS[class_name]


@app.post("/detect")
async def detect_vehicle(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    img_tensor = F.to_tensor(image.convert("RGB"))

    with torch.no_grad():
        output = model([img_tensor])[0]

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    boxes = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]

    for box, label, conf in zip(boxes, labels, scores):

        if conf < 0.5:
            continue

        conf_text = f"{conf:.2f}"
        class_name = CLASS_NAMES[label.item()]
        label_text = f"{class_name} {conf_text}"

        x1, y1, x2, y2 = box.tolist()

        # ==============================
        # ✔ MÀU MỚI THEO YÊU CẦU
        color_box = (0, 255, 0)     # Xanh lá cây
        color_text = (200, 0, 0)    # Đỏ
        # ==============================

        # BOX
        draw.rectangle([(x1, y1), (x2, y2)], outline=color_box + (255,), width=3)

        textbox = draw.textbbox((0, 0), label_text, font=font)
        tw = textbox[2] - textbox[0]
        th = textbox[3] - textbox[1]

        # Nền label (màu xanh lá nhạt)
        draw.rectangle(
            [(x1, y1 - th - 6), (x1 + tw + 6, y1)],
            fill=color_box + (130,)   # xanh lá mờ
        )

        # CHỮ đỏ
        draw.text((x1 + 3, y1 - th - 3), label_text, font=font, fill=color_text)

    final_image = Image.alpha_composite(image, overlay).convert("RGB")

    img_bytes = io.BytesIO()
    final_image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/jpeg")
