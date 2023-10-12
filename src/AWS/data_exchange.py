import logging
import os
from src.utils.loggers import setup_logger
import boto3




def download_data_from_S3(bucket_name, bucket_path, local_path):
    logger = setup_logger("S3_downloader")
    
    s3 = boto3.client("s3")
    bucket = s3.bucket(bucket_name)

    num_files_downloaded = 0
    total_size = 0

    for obj in bucket.objects.filter(Prefix=bucket_path):
        local_file_path = os.path.join(local_path, obj.key)
        local_directory = os.path.dirname(local_file_path)
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        bucket.download_file(obj.key, local_file_path)

        num_files_downloaded += 1
        total_size += obj.size

    total_size_mb = total_size / 1024
    logger.info(f"Downloaded {num_files_downloaded} with a total size of {total_size_mb:.2f} MB)

def upload_data_to_S3(local_path, bucket_name, bucket_path):
    s3.upload_file(local_path, bucket_name, bucket_path)
