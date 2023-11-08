"""The module provides the functions to extract key points from videos and write them to CSV and AVI files."""

import glob
import pathlib
from pathlib import Path
from typing import Dict, List

from ultralytics.utils.torch_utils import select_device

from src.data.keypoints_handler import (
    KeyPointsCSVWriter,
    KeyPointsOnlyVideoWriter,
    KeyPointsVideoWriter,
)


def csv_keypoints_factory(
    model,
    path_to_video_folder: pathlib.Path,
    path_to_csv_keypoits_folder: pathlib.Path,
    classes: Dict[str, str],
    device: str = "cpu",
) -> None:
    """Exctarct keypoins from videos and write them to CSV files.

    Args:
        model (ultralytics.models.yolo.model.YOLO): A model for keypoints extraction.
        path_to_video_folder (pathlib.Path):
            Path to the folder with video files.
        path_to_csv_keypoits_folder (pathlib.Path):
            Path to the folder where the CSV files will be stored after video processing.
        classes (Dict[str, str]):
            The classes (ex. "crossing", defence", "shot", and etc.)
            to correctly iterate over video folders.
        device (str): Compute device ('cpu' or 'cuda'). Default is 'cpu'.
    """
    device = select_device(device)
    model = model.to(device)

    for class_ in classes.values():
        mp4_videos = glob.glob("*.mp4", root_dir=path_to_video_folder / class_)
        avi_videos = glob.glob("*.avi", root_dir=path_to_video_folder / class_)
        videos = mp4_videos + avi_videos
        for video in videos:
            path_to_video_file_in = path_to_video_folder / class_ / video
            csv_file_name = Path(video).stem + ".csv"
            path_to_csv_file_out = path_to_csv_keypoits_folder / class_ / csv_file_name

            results = model(
                source=path_to_video_file_in, conf=0.30, show=False, stream=True
            )
            kp_csv_writer = KeyPointsCSVWriter(results)
            kp_csv_writer.write_keypoints_to_csv(path_to_csv_file_out)


def video_keypoints_factory(
    path_to_video_folder: pathlib.Path,
    path_to_csv_keypoits_folder: pathlib.Path,
    classes: Dict[str, str],
    keypoints_pairs: List[List[int]],
    auto_labeling: bool = False,
) -> None:
    """Writes key points from CSV to AVI files.

    Args:
        path_to_video_folder (pathlib.Path):
            Path to the folder (with subfolders as classes) with video files
            (e.g. 'data/raw/scenes')
        path_to_csv_keypoits_folder (pathlib.Path):
            Path to the folder (with subfolders as classes) with CSV files.
            (e.g. 'data/processed/scenes')
        classes (Dict[str, str]):
            The classes (ex. "crossing", "defence", "shot", and etc.)
            to correctly iterate over video folders.
        keypoints_pairs (List[List[int]]):
            The COCO keypoint classes ("nose", "left_eye", "right_eye", and etc.)
    """
    for class_ in classes.values():
        csvs = glob.glob("*.csv", root_dir=path_to_csv_keypoits_folder / class_)
        for csv in csvs:
            path_to_csv_file = path_to_csv_keypoits_folder / class_ / csv

            # Try .mp4 extension first
            path_to_video_file_in = (
                path_to_video_folder / class_ / (Path(csv).stem + ".mp4")
            )
            # If .mp4 doesn't exist, try .avi extension
            if not path_to_video_file_in.exists():
                path_to_video_file_in = (
                    path_to_video_folder / class_ / (Path(csv).stem + ".avi")
                )

            path_to_video_file_out = (
                path_to_csv_keypoits_folder / class_ / (Path(csv).stem + ".avi")
            )

            if auto_labeling:
                kp_video_writer = KeyPointsOnlyVideoWriter(keypoints_pairs)
            else:
                kp_video_writer = KeyPointsVideoWriter(keypoints_pairs)
            kp_video_writer.write_video_with_keypoints(
                path_to_video_file_in,
                path_to_video_file_out,
                path_to_csv_file,
            )
