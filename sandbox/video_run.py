import csv
import logging
import os

import cv2

from src.data.video_handler import _get_video_params, _video_writer


class KeyPointsVideoWriter:
    """Apply keypoints to a video. The video were processed before, keypoints were extracted stored in a CSV file."""

    def __init__(self, keypoints_pairs):
        self.keypoints_pairs = keypoints_pairs
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)

            file_handler = logging.FileHandler("loggs/video_writer.log")
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_formatter = logging.Formatter("%(levelname)s - %(message)s")
            stream_handler.setFormatter(stream_formatter)
            logger.addHandler(stream_handler)

        return logger

    def read_keypoints_from_csv(self, csv_path_in) -> dict:
        """Read the keypoints from a CSV file.
        The CSV file must have the fillowing structure: "Frame", "Person", "Keypoint", "X", "Y", "Prob".
        """
        keypoints_dict = {}
        try:
            with open(csv_path_in, mode="r", newline="", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # skip header
                for row in csv_reader:
                    try:
                        frame_index, person_index, keypoint_index, x, y, prob = list(
                            map(int, row[:5])
                        ) + [float(row[-1])]
                    except ValueError as err:
                        raise ValueError(
                            f"Error processing row {row} in CSV: {err}"
                        ) from err

                    frame_keypoints = keypoints_dict.get(frame_index, {})
                    person_keypoints = frame_keypoints.get(person_index, [])
                    person_keypoints.append((keypoint_index, x, y, prob))
                    frame_keypoints[person_index] = person_keypoints
                    keypoints_dict[frame_index] = frame_keypoints

        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"The specified CSV file {csv_path_in} was not found."
            ) from err
        return keypoints_dict

    def write_video_with_keypoints(
        self, video_path_in, video_path_out, csv_path_in
    ) -> None:
        """Writes frames with pose estimations to an .avi video file."""
        try:
            keypoints_dict = self.read_keypoints_from_csv(csv_path_in)
            fps, width, height = _get_video_params(video_path_in)
            print(f"fps={fps}")
            avi_writer = _video_writer(video_path_out, fps, width, height)
            cap = cv2.VideoCapture(str(video_path_in))
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Break the loop if we reach the end of the video

                frame_keypoints = keypoints_dict.get(frame_index, {})
                for person_keypoints in frame_keypoints.values():
                    # Draw points
                    for _, x, y, _ in person_keypoints:
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    for i, j in self.keypoints_pairs:
                        # Ensure the indices are within bounds
                        if i < len(person_keypoints) and j < len(person_keypoints):
                            _, x1, y1, _ = person_keypoints[i]
                            _, x2, y2, _ = person_keypoints[j]
                            cv2.line(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2,
                            )

                        else:
                            error_message = (
                                f"Skipping line for out-of-bounds indices: {i}, {j}"
                            )
                            self.logger.error(error_message)

                avi_writer.write(frame)
                frame_index += 1

            if os.path.getsize(video_path_out) == 0:
                error_message = f"The output video file {video_path_out} is empty"
                self.logger.error(error_message)
            else:
                success_message = f"Success for the video {video_path_out}"
                self.logger.info(success_message)

        except Exception as exc:
            error_message = f"Error processing video {video_path_in}: {exc}"
            self.logger.error(error_message)
        finally:
            cv2.destroyAllWindows()
            avi_writer.release()
            cap.release()


video_path_in = "./data/debug/actions/crossing/crossing_KS_1_act1.avi"
video_path_out = "./data/auto_layout/crossing/crossing_KS_1_act1.avi"
csv_path_in = "./data/auto_layout/crossing/crossing_KS_1_act1.csv"

pairs = [
    # (0, 1),  # nose to left_eye
    # (0, 2),  # nose to right_eye
    # (1, 3),  # left_eye to left_ear
    # (2, 4),  # right_eye to right_ear
    (5, 6),  # left_shoulder to right_shoulder
    (5, 7),  # left_shoulder to left_elbow
    (6, 8),  # right_shoulder to right_elbow
    (7, 9),  # left_elbow to left_wrist
    (8, 10),  # right_elbow to right_wrist
    (5, 11),  # left_shoulder to left_hip
    (6, 12),  # right_shoulder to right_hip
    (11, 12),  # left_hip to right_hip
    (11, 13),  # left_hip to left_knee
    (12, 14),  # right_hip to right_knee
    (13, 15),  # left_knee to left_ankle
    (14, 16),  # right_knee to right_ankle
]

writer = KeyPointsVideoWriter(pairs)

writer.write_video_with_keypoints(video_path_in, video_path_out, csv_path_in)
