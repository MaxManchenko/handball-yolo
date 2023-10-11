"""The module provides the pipeline to extract keypoints from videos and write them to CSV and AVI files."""

from pathlib import Path

from src.data.keypoints_factories import csv_keypoints_factory, video_keypoints_factory
from src.load_config import load_config
from src.models.initialize_models import initialize_yolo_model


def main():
    config = load_config()
    path_to_model = config["models"]["pose"]
    path_to_data_root = Path(config["data"]["root"])
    # path_to_video_folder = path_to_data_root / config["data"]["actions"]

    path_to_video_folder = path_to_data_root / config["data"]["debug_actions"]

    path_to_csv_keypoits_folder = path_to_data_root / config["data"]["auto_layout"]
    classes = config["classes"]
    keypoints_pairs = config["keypoints"]["coco_pairs"]

    model = initialize_yolo_model(path_to_model)

    # csv_keypoints_factory(
    #     model, path_to_video_folder, path_to_csv_keypoits_folder, classes
    # )
    video_keypoints_factory(
        path_to_video_folder, path_to_csv_keypoits_folder, classes, keypoints_pairs
    )


if __name__ == "__main__":
    main()
