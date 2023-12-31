from pathlib import Path

from src.load_config import load_config


def get_config_params_for_autolabeling_on_AWS():
    """Load configuration and extract necessary parameters."""
    config = load_config()

    params = {
        "path_to_model": config["models"]["pose_gpu"],
        "path_to_local_data_root": Path(config["data_EC2"]["root"]),
        "classes": config["classes"],
        "bucket_name": config["S3"]["bucket_name"],
        "bucket_path_to_download": config["S3"]["actions"],
        "bucket_path_to_upload": config["S3"]["auto_labeling"],
        "artifact_location": config["S3"]["artifact_location"],
    }

    params["path_to_local_video_folder"] = (
        params["path_to_local_data_root"] / config["data_EC2"]["actions"]
    )
    params["path_to_local_csv_folder"] = (
        params["path_to_local_data_root"] / config["data_EC2"]["auto_labeling"]
    )

    return params


def get_config_params_for_autolabeling_locally():
    """Load configuration and extract necessary parameters."""
    config = load_config()

    params = {
        "path_to_model": config["models"]["pose"],
        "path_to_local_data_root": Path(config["data"]["root"]),
        "classes": config["classes"],
        "bucket_name": config["S3"]["bucket_name"],
        "bucket_path_to_download": config["S3"]["actions"],
        "bucket_path_to_upload": config["S3"]["auto_labeling"],
        "artifact_location": config["S3"]["artifact_location"],
    }

    params["path_to_local_video_folder"] = (
        params["path_to_local_data_root"] / config["data"]["actions"]
    )
    params["path_to_local_csv_folder"] = (
        params["path_to_local_data_root"] / config["data"]["auto_labeling"]
    )

    return params


def get_config_params_for_autolabeling_debug_mode():
    """Load configuration and extract necessary parameters."""
    config = load_config()

    params = {
        "path_to_model": config["models"]["pose"],
        "path_to_local_data_root": Path(config["data"]["root"]),
        "classes": config["classes"],
        "bucket_name": config["S3"]["bucket_name"],
        "bucket_path_to_download": config["S3"]["actions"],
        "bucket_path_to_upload": config["S3"]["auto_labeling"],
        "artifact_location": config["S3"]["artifact_location"],
    }

    params["path_to_local_video_folder"] = (
        params["path_to_local_data_root"] / config["data"]["debug_actions"]
    )
    params["path_to_local_csv_folder"] = (
        params["path_to_local_data_root"] / config["data"]["debug_auto_labeling"]
    )

    return params
