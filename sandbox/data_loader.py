import glob
import pathlib
import random
from pathlib import Path
from typing import Dict, List

import yaml
from ultralytics import YOLO

from src.data.keypoints_handler import KeyPointsCSVWriter, KeyPointsVideoWriter


with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

path_to_model = config["models"]["pose"]
path_to_data_root = Path(config["data"]["root"])
path_to_video_folder = path_to_data_root / config["data"]["scenes"]
path_to_csv_keypoits_folder = path_to_data_root / config["data"]["csv_kpoints"]
classes = config["classes"]


model = YOLO(path_to_model)
keypoints_pairs = config["keypoints"]["coco_pairs"]


def csv_keypoints_factory(
    path_to_video_folder: pathlib.Path,
    classes: Dict[str, str],
) -> None:
    for class_ in classes.values():
        videos = glob.glob("*.mp4", root_dir=path_to_video_folder / class_)
        for video in random.sample(videos, 3):
            path_to_video_file_in = path_to_video_folder / class_ / video
            csv_file_name = Path(video).stem + ".csv"
            path_to_csv_file_out = path_to_csv_keypoits_folder / class_ / csv_file_name

            results = model(
                source=path_to_video_file_in, conf=0.45, show=False, stream=True
            )
            kp_csv_writer = KeyPointsCSVWriter(results)
            kp_csv_writer.write_keypoints_to_csv(path_to_csv_file_out)
        # break


csv_keypoints_factory(path_to_video_folder, classes)


def keypoints_video_writer_factory(
    path_to_csv_keypoits_folder: pathlib.Path,
    classes: Dict[str, str],
    keypoints_pairs: List[List[int]],
) -> None:
    for class_ in classes.values():
        csvs = glob.glob("*.csv", root_dir=path_to_csv_keypoits_folder / class_)
        for csv in csvs:
            path_to_csv_file = path_to_csv_keypoits_folder / class_ / csv
            path_to_video_file_in = (
                path_to_video_folder / class_ / (Path(csv).stem + ".mp4")
            )
            path_to_video_file_out = (
                path_to_csv_keypoits_folder / class_ / (Path(csv).stem + ".avi")
            )

            kp_video_writer = KeyPointsVideoWriter(keypoints_pairs)
            kp_video_writer.write_video_with_keypoints(
                path_to_video_file_in,
                path_to_video_file_out,
                path_to_csv_file,
            )
        # break


keypoints_video_writer_factory(path_to_csv_keypoits_folder, classes, keypoints_pairs)
