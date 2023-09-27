import cv2


VIDEO = "./data/raw/scenes/crossing/crossing_KS_1.mp4"
MARKUP = "./data/raw/player_detections/crossing/crossing_KS_1.csv"
VIDEO_OUT = "./data/processed/output_2.avi"


def video_loader(file_path_in):
    """Create a video capture object to get video properties"""

    cap = cv2.VideoCapture(file_path_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return fps, width, height


with open(MARKUP, "r") as file:
    csv_data = file.read()

# Split csv data into chunks per frame
frames_data = csv_data.split("Frame ")[1:]
frames_data = {
    int(frame.split("\n")[0]): frame.split("\n")[1:-1] for frame in frames_data
}


def process_video(video_path, csv_path, output_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return  # Exit the function if the video file cannot be opened

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Changed codec to MJPG
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load csv data
    with open(csv_path, "r") as file:
        csv_data = file.read()

    # Split csv data into chunks per frame
    frames_data = csv_data.split("Frame ")[1:]
    frames_data = {
        int(frame.split("\n")[0]): frame.split("\n")[1:-1] for frame in frames_data
    }

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number in frames_data:
            for bbox_line in frames_data[frame_number]:
                x1, y1, x2, y2, score = map(float, bbox_line.split(","))
                if score > 0.5:  # Assume a confidence threshold of 0.5
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"{score:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

        out.write(frame)  # Write the frame with bounding boxes to the output video

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Use the function

process_video(VIDEO, MARKUP, VIDEO_OUT)


from src.data.bboxes_processors import VideoBoundingBoxProcessor

processor = VideoBoundingBoxProcessor(VIDEO, MARKUP, VIDEO_OUT)
processor.process_video()
