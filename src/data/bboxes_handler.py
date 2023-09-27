import cv2


class VideoBoundingBoxProcessor:
    """A class to process a video file and draw bounding boxes on each frame
    based on data from a CSV file.

    Attributes:
        video_path_in (str): The path to the video file.
        csv_path (str): The path to the CSV file containing bounding box data.
        video_path_out (str): The path where the processed video will be saved.
        cap (cv2.VideoCapture): A cv2 VideoCapture object.
        out (cv2.VideoWriter): A cv2 VideoWriter object.
        frames_data (dict): A dictionary containing bounding box data per frame.
    """

    def __init__(self, video_path_in, csv_path, video_path_out):
        self.video_path_in = video_path_in
        self.csv_path = csv_path
        self.video_path_out = video_path_out
        self.cap = None
        self.out = None
        self.frames_data = None

    def load_video(self):
        """Loads the video file."""
        self.cap = cv2.VideoCapture(self.video_path_in)
        if not self.cap.isOpened():
            raise Exception(f"Failed to open video: {self.video_path_in}")

    def load_csv_data(self):
        """Loads bounding box data from the CSV file."""
        with open(self.csv_path, "r") as file:
            csv_data = file.read()

        frames_data = csv_data.split("Frame ")[1:]
        self.frames_data = {
            int(frame.split("\n")[0]): frame.split("\n")[1:-1] for frame in frames_data
        }

    def initialize_writer(self):
        """Initializes the video writer object."""
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.out = cv2.VideoWriter(self.video_path_out, fourcc, fps, (width, height))

    def process_frames(self):
        """Processes each frame, drawing bounding boxes where specified."""
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

            self.out.write(frame)

    def release_resources(self):
        """Releases video resources and destroys any OpenCV windows."""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def process_video(self):
        """Main method to process the video and output the result."""
        self.load_video()
        self.load_csv_data()
        self.initialize_writer()
        self.process_frames()
        self.release_resources()
