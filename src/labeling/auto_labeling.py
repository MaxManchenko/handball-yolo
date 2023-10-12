"""The module provides the pipeline to extract keypoints from videos and write them to CSV files."""

from pathlib import Path

from src.data.keypoints_factories import csv_keypoints_factory, video_keypoints_factory
from src.load_config import load_config
from src.models.initialize_models import initialize_yolo_model


from src.load_config import load_config

config = load_config()

bucket_name = config["S3"]["bucket_name"]
bucket_path_to_download = config["S3"]["actions"]
bucket_path_to_upload = config["S3"]["csv_path_out"]


def main():
    config = load_config()
    path_to_model = config["models"]["pose_gpu"]
    path_to_data_root = Path(config["data_EC2"]["root"])

    path_to_video_folder = path_to_data_root / config["data"]["debug_actions"]

    path_to_csv_keypoits_folder = path_to_data_root / config["data"]["auto_labeling"]
    classes = config["classes"]
    keypoints_pairs = config["keypoints"]["coco_pairs"]

    model = initialize_yolo_model(path_to_model)

    csv_keypoints_factory(
        model, path_to_video_folder, path_to_csv_keypoits_folder, classes
    )
    # video_keypoints_factory(
    #     path_to_video_folder,
    #     path_to_csv_keypoits_folder,
    #     classes,
    #     keypoints_pairs,
    #     auto_labeling=True,
    # )


if __name__ == "__main__":
    main()
