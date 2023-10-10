import os

import cv2

from src.data.video_handler import _get_video_params, _video_writer


class VideoBoundingBoxProcessor:
    """A class to process a video file and draw bounding boxes on each frame
    based on data from a CSV file.

    Attributes:
        video_path_in (str): The path to the video file.
        csv_path_in (str): The path to the CSV file containing bounding box data.
        video_path_out (str): The path where the processed video will be saved.
        cap (cv2.VideoCapture): A cv2 VideoCapture object.
        out (cv2.VideoWriter): A cv2 VideoWriter object.
        frames_data (dict): A dictionary containing bounding box data per frame.
    """

    def __init__(self, video_path_in, csv_path_in, video_path_out):
        self.video_path_in = video_path_in
        self.csv_path_in = csv_path_in
        self.video_path_out = video_path_out
        self.cap = None
        self.frames_data = None

    def load_video(self):
        """Loads the video file."""
        self.cap = cv2.VideoCapture(self.video_path_in)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to find video: {self.video_path_in}")

    def load_csv_data(self):
        """Loads bounding box data from the CSV file."""
        with open(self.csv_path_in, "r", encoding="utf-8") as file:
            csv_data = file.read()

        frames_data = csv_data.split("Frame ")[1:]
        self.frames_data = {
            int(frame.split("\n")[0]): frame.split("\n")[1:-1] for frame in frames_data
        }

    def process_frames(self):
        """Processes each frame, drawing bounding boxes where specified."""

        output_dir = os.path.dirname(self.video_path_out)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fps, width, height = _get_video_params(self.video_path_in)
        avi_writer = _video_writer(self.video_path_out, fps, width, height)

        frame_number = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_number += 1

            if frame_number in self.frames_data:
                for bbox_line in self.frames_data[frame_number]:
                    x1, y1, x2, y2, score = map(float, bbox_line.split(","))
                    if score > 0.5:
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"{score:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                        )

            avi_writer.write(frame)
        avi_writer.release()

    def release_resources(self):
        """Releases video resources and destroys any OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()

    def process_video(self):
        """Main method to process the video and output the result."""
        self.load_video()
        self.load_csv_data()
        self.process_frames()
        self.release_resources()
