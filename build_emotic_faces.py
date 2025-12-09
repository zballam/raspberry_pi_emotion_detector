# build_emotic_faces.py
from scipy.io import loadmat

from utility.face_pipeline import (
    create_face_detector,
    build_face_dataset,
)


def main():
    # 1) Load EMOTIC annotations
    annotations_path = "emotic/Annotations/annotations.mat"
    data = loadmat(annotations_path)
    train_data = data["train"][0]

    print("Loaded EMOTIC train split:", len(train_data), "samples")

    # 2) Create face detector
    detector = create_face_detector(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    # 3) Build face dataset
    output_root = "data/emotic_faces_128"
    build_face_dataset(
        data=train_data,
        output_root=output_root,
        detector=detector,
        root="emotic",
        face_size=128,
        max_samples=4000,      # or None for all
        min_face_score=0.5,
    )


if __name__ == "__main__":
    main()