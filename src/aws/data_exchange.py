import os
import pathlib
from typing import Optional

import boto3
from dotenv import load_dotenv

from src.utils.loggers import setup_logger

load_dotenv("./.env")


def download_data_from_S3(
    bucket_name: str,
    s3_folder: str,
    local_path: pathlib.Path,
    log_file: Optional[str] = None,
) -> None:
    """
    Downloads a S3 directory (and its subdirectories) to a local machine.

    Args:
        bucket_name (str): Name of the S3 bucket to upload to.
        s3_folder (str): Folder path in the S3 bucket.
        local_path (pathlib.path): Local directory path to download.
        log_file (Optional[str], optional): Path to the log file. If specified,
                                            logs will also be written to this file.
                                            Defaults to None.
    Returns:
        None
    """
    logger = setup_logger(name="S3_downloader", level="INFO", log_file=log_file)

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    num_files_downloaded = 0
    total_size = 0

    for obj in bucket.objects.filter(Prefix=s3_folder):
        if obj.key.endswith("/"):
            continue

        _, *rest_of_key = obj.key.split("/")

        local_file_path = os.path.join(local_path, *rest_of_key)
        local_directory = os.path.dirname(local_file_path)
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        s3.meta.client.download_file(bucket_name, obj.key, local_file_path)

        num_files_downloaded += 1
        total_size += obj.size

    total_size_mb = total_size / (1024 * 1024)
    logger.info(
        f"Downloaded {num_files_downloaded} files with a total size of {total_size_mb:.2f} MB in the folder {local_path}"
    )


def upload_data_to_s3(
    local_path: pathlib.Path,
    bucket_name: str,
    s3_folder: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Uploads a local directory (and its subdirectories) to an S3 bucket.

    Args:
        local_path (str): Local directory path to upload.
        bucket_name (str): Name of the S3 bucket to upload to.
        s3_folder (Optional[str], optional): Folder path in the S3 bucket. If provided,
                                             the local directory's contents will be uploaded
                                             into this folder. Defaults to the root of the bucket.
        log_file (Optional[str], optional):  Path to the log file. If specified,
                                             logs will also be written to this file.
                                             Defaults to None.

    Returns:
        None
    """
    logger = setup_logger(name="S3_uploader", level="INFO", log_file=log_file)

    s3 = boto3.client("s3")

    num_files_uploaded = 0

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
                num_files_uploaded += 1
            except Exception as e:
                logger.info(f"Failed to upload {full_path}. Reason: {e}")
    logger.info(f"Uploaded {num_files_uploaded} files to S3:{bucket_name}/{s3_folder}")
