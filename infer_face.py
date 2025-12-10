# infer_face.py

from pathlib import Path
import cv2
import torch
from torchvision import transforms

from train_model import build_model, get_transforms
from utility.face_dataloader import EmoticFaceDataset

CSV_PATH = "data/emotic_faces_128/labels_coarse.csv"
MODEL_NAME = "efficientnet_lite0"  # your chosen best model
MODEL_PATH = f"models/emotion_{MODEL_NAME}_best.pt"

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)

def load_label_mapping():
    ds = EmoticFaceDataset(CSV_PATH, label_column="coarse_label", transform=None)
    return ds.idx_to_label

def preprocess_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # CHW

    tf = get_transforms(MODEL_NAME, is_train=False)
    tensor = tf(tensor)
    return tensor.unsqueeze(0)  # add batch dim

def main():
    idx_to_label = load_label_mapping()

    # Build + load model
    num_classes = len(idx_to_label)
    model, _ = build_model(MODEL_NAME, num_classes)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Example image
    image_path = "some_face_image.jpg"
    x = preprocess_image(image_path).to(device)

    with torch.no_grad():
        out = model(x)
        pred_idx = out.argmax(1).item()

    print("Predicted:", idx_to_label[pred_idx])


if __name__ == "__main__":
    main()