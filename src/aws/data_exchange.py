import os
import pathlib
from typing import Optional

import boto3

from src.utils.loggers import setup_logger


def download_data_from_S3(
    bucket_name: str, s3_folder: str, local_path: pathlib.Path
) -> None:
    """
    Downloads a S3 directory (and its subdirectories) to a local machine.

    Args:
        bucket_name (str): Name of the S3 bucket to upload to.
        s3_folder (str): Folder path in the S3 bucket.
        local_path (pathlib.path): Local directory path to download.

    Returns:
        None
    """
    logger = setup_logger("S3_downloader")

    s3 = boto3.client("s3")
    bucket = s3.bucket(bucket_name)

    num_files_downloaded = 0
    total_size = 0

    for obj in bucket.objects.filter(Prefix=s3_folder):
        local_file_path = os.path.join(local_path, obj.key)
        local_directory = os.path.dirname(local_file_path)
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        bucket.download_file(obj.key, local_file_path)

        num_files_downloaded += 1
        total_size += obj.size

    total_size_mb = total_size / 1024
    logger.info(
        f"Downloaded {num_files_downloaded} with a total size of {total_size_mb:.2f} MB"
    )


def upload_data_to_s3(
    local_path: pathlib.Path, bucket_name: str, s3_folder: Optional[str] = None
) -> None:
    """
    Uploads a local directory (and its subdirectories) to an S3 bucket.

    Args:
        local_path (str): Local directory path to upload.
        bucket_name (str): Name of the S3 bucket to upload to.
        s3_folder (Optional[str], optional): Folder path in the S3 bucket. If provided,
                                             the local directory's contents will be uploaded
                                             into this folder. Defaults to the root of the bucket.

    Returns:
        None
    """
    logger = setup_logger("S3_uploader")

    s3 = boto3.client("s3")

    for subdir, _, files in os.walk(local_path):
        for file in files:
            full_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(full_path, local_path)

            # If there's a specified folder in the bucket, append the relative path to it
            s3_path = (
                os.path.join(s3_folder, relative_path) if s3_folder else relative_path
            )

            try:
                s3.upload_file(full_path, bucket_name, s3_path)
                logger.info(f"Uploaded {full_path} to {bucket_name}/{s3_path}")
            except Exception as e:
                logger.info(f"Failed to upload {full_path}. Reason: {e}")
