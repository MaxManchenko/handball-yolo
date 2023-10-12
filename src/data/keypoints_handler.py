"""The module provides the classes to handle persons' keypoints."""
import csv
import logging
import os

import cv2
import numpy as np

from src.data.video_handler import _get_video_params, _video_writer
from src.utils.loggers import setup_logger


class KeyPointsCSVWriter:
    """Writes keypoints coordinates to a CSV file in the following order: "Frame", "Person", "Keypoint", "X", "Y", "Prob"."""

    def __init__(self, results: str):
        self.results = results
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            logger.setLevel(logging.WARNING)
            handler = logging.FileHandler("loggs/csv_writer.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def extract_keypoints_from_frames(self) -> list:
        """Extracts keypoints from a frame for each person detected."""
        keypoints_list = []
        for frame_number, frame_data in enumerate(self.results):
            # Ensure that frame_data has the 'keypoints' attribute and that it is not None
            if not hasattr(frame_data, "keypoints") or frame_data.keypoints is None:
                raise AttributeError(
                    f"Frame data at index {frame_number} lacks keypoints attribute or it is None."
                )
            # Continue to next iteration if no keypoints are present in the current frame
            if not frame_data.keypoints.data.numel():
                continue

            frame_keypoints = []  # List to hold all person keypoints for this frame
            for person in frame_data.keypoints.data:
                if person.shape[0] == 0:
                    continue  # Skip empty person data

                person_keypoints = []  # List to hold this person's keypoints
                for point in person:
                    try:
                        x, y = map(int, point[:2])
                        prob = float(point[2])
                    except ValueError as err:
                        error_message = (
                            f"Error processing keypoints at frame {frame_number}"
                        )
                        self.logger(error_message)
                        raise ValueError(
                            f"Error processing keypoints at frame {frame_number}"
                        ) from err

                    person_keypoints.append((x, y, prob))
                frame_keypoints.append((person_keypoints))
            keypoints_list.append((frame_number, frame_keypoints))
        return keypoints_list

    def write_keypoints_to_csv(self, csv_path_out) -> None:
        """Writes keypoints to a CSV file."""
        keypoints_list = self.extract_keypoints_from_frames()
        if not keypoints_list:
            video_file, _ = os.path.splitext(csv_path_out)
            warning_message = f"No keypoints extracted from the'{video_file}'"
            self.logger.warning(warning_message)
        else:
            try:
                with open(csv_path_out, mode="w", newline="", encoding="utf-8") as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(
                        ["Frame", "Person", "Keypoint", "X", "Y", "Prob"]
                    )
                    for frame_number, person_keypoints_list in keypoints_list:
                        for person_index, person_keypoints in enumerate(
                            person_keypoints_list
                        ):
                            for keypoint_index, keypoint in enumerate(person_keypoints):
                                csv_writer.writerow(
                                    [
                                        frame_number,
                                        person_index,
                                        keypoint_index,
                                        *keypoint,
                                    ]
                                )
            except IOError as err:
                raise IOError(f"Error writing to {csv_path_out}: {err}") from err


class KeyPointsVideoWriter:
    """Apply keypoints to a video. The video were processed before, keypoints were extracted stored in a CSV file."""

    def __init__(self, keypoints_pairs):
        self.keypoints_pairs = keypoints_pairs
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
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

    def write_keypoints_on_frame(self, frame, frame_keypoints) -> np.ndarray:
        """Overlay keypoints onto a video frame.

        Args:
            frame (np.ndarray): Video frame on which keypoints are to be drawn.
            frame_keypoints (dict): Dictionary containing the keypoints for the frame.

        Returns:
            np.ndarray: The frame with keypoints drawn on it.
        """
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
                    error_message = f"Skipping line for out-of-bounds indices: {i}, {j}"
                    self.logger.error(error_message)
        return frame

    def should_write_frame(self, frame_keypoints) -> bool:
        """Determine whether a frame should be written to the output video.
        This base method always returns True, meaning all frames are written.

        Args:
            frame (np.ndarray): Video frame being processed.
            frame_keypoints (dict): Dictionary containing the keypoints for the frame.

        Returns:
            bool: True if the frame should be written, False otherwise.
        """
        return True

    def write_video_with_keypoints(
        self, video_path_in, video_path_out, csv_path_in
    ) -> None:
        """Writes frames with pose estimations to an AVI video file."""
        try:
            keypoints_dict = self.read_keypoints_from_csv(csv_path_in)
            fps, width, height = _get_video_params(video_path_in)
            avi_writer = _video_writer(video_path_out, fps, width, height)
            cap = cv2.VideoCapture(str(video_path_in))
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Break the loop if we reach the end of the video

                frame_keypoints = keypoints_dict.get(frame_index, {})
                if self.should_write_frame(frame_keypoints):
                    frame_with_keypoints = self.write_keypoints_on_frame(
                        frame, frame_keypoints
                    )
                    avi_writer.write(frame_with_keypoints)
                frame_index += 1

        except Exception as exc:
            error_message = f"Error processing video {video_path_in}: {exc}"
            self.logger.error(error_message)
        finally:
            cv2.destroyAllWindows()
            avi_writer.release()
            cap.release()

        if os.path.getsize(video_path_out) == 0:
            error_message = f"The output video file {video_path_out} is empty"
            self.logger.error(error_message)
        else:
            success_message = f"Success for the video {video_path_out}"
            self.logger.info(success_message)


class KeyPointsOnlyVideoWriter(KeyPointsVideoWriter):
    """Writes only the video frames containing keypoints to an AVI video file."""

    def should_write_frame(self, frame_keypoints):
        """
        Args:
            frame_keypoints (dict): Dictionary containing the keypoints for the frame.
        Returns:
            bool: True if the frame contains keypoints and should be written, False otherwise.
        """
        return any(frame_keypoints.values())
