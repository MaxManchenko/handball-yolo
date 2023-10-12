"""The module provides the pipeline to extract keypoints from videos and write them to CSV files."""

import pathlib
from typing import Dict, Optional

from src.aws.data_exchange import download_data_from_S3, upload_data_to_s3
from src.data.keypoints_factories import csv_keypoints_factory
from src.models.initialize_models import initialize_yolo_model
from src.utils.get_config_params import get_config_params_for_autolabeling


def process_video(
    path_to_model: str,
    path_to_local_video_folder: pathlib.Path,
    path_to_local_csv_folder: pathlib.Path,
    classes: Dict[str, str],
    bucket_name: Optional[str] = None,
    bucket_path_to_download: Optional[str] = None,
    bucket_path_to_upload: Optional[str] = None,
) -> None:
    if bucket_path_to_download and bucket_name:
        download_data_from_S3(
            bucket_name, bucket_path_to_download, path_to_local_csv_folder
        )

    model = initialize_yolo_model(path_to_model)
    csv_keypoints_factory(
        model, path_to_local_video_folder, path_to_local_csv_folder, classes
    )

    if bucket_path_to_upload and bucket_name:
        upload_data_to_s3(path_to_local_csv_folder, bucket_name, bucket_path_to_upload)


def main():
    config_params = get_config_params_for_autolabeling()

    process_video(
        path_to_model=config_params["path_to model"],
        path_to_local_video_folder=config_params["path_to_local_video_folder"],
        path_to_local_csv_folder=config_params["path_to_local_csv_folder"],
        classes=config_params["classes"],
        bucket_name=config_params["bucket_name"],
        bucket_path_to_download=config_params["bucket_path_to_download"],
        bucket_path_to_upload=config_params["bucket_path_to_upload"],
    )


if __name__ == "__main__":
    main()