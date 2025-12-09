import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_image_path(sample, root="emotic"):
    """
    Returns the full path to the image for a given EMOTIC sample.
    """
    folder = sample["folder"][0]
    filename = sample["filename"][0]
    return os.path.join(root, folder, filename)

def load_image(image_path):
    """
    Loads an image in RGB format. Raises if missing.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def get_person(sample, index=0):
    """
    Returns the person_data object for person #index in the sample.
    """
    return sample["person"][0][index]

def get_body_bbox(person_data):
    """
    Returns (x1, y1, x2, y2) as ints.
    Handles EMOTIC bbox nesting automatically.
    """
    bbox = np.array(person_data["body_bbox"][0]).reshape(-1)  # flatten
    x1, y1, x2, y2 = map(int, bbox[:4])
    return x1, y1, x2, y2

def crop_bbox(img, bbox):
    """
    Crops an image using (x1, y1, x2, y2).
    Clamps coordinates to image boundaries.
    """
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    return img[y1:y2, x1:x2]


def visualize(img=None, crop=None, mode="both", title="EMOTIC Visualization"):
    """
    Visualize EMOTIC image data.
    """

    if mode == "full":
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title("Full Image")
        plt.axis("off")
        plt.suptitle(title)
        plt.show()
        return

    if mode == "crop":
        plt.figure(figsize=(6, 6))
        plt.imshow(crop)
        plt.title("Body Crop")
        plt.axis("off")
        plt.suptitle(title)
        plt.show()
        return

    # mode == "both"
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Full Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(crop)
    plt.title("Body Crop")
    plt.axis("off")

    plt.suptitle(title)
    plt.show()