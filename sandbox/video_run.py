from ultralytics import YOLO
import cv2


MODEL_PATH = "./model"
VIDEO_IN = "./data/raw/actions/shot/shot_KS_8_act1.avi"
# VIDEO_IN = "./data/raw/shot_KS_72.mp4"
VIDEO_OUT = "./data/processed/output.avi"
FRAME_OUT = "./data/processed"
yolo_model = YOLO(f"{MODEL_PATH}/yolov8n-pose.pt")


def get_video_params(file_path_in):
    """Create a video capture object to get video properties"""

    cap = cv2.VideoCapture(file_path_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return fps, width, height


def write_video_with_bbox(file_path_in, file_path_out, yolo_model, fps, width, height):
    """Writes frames to an .avi video file

    Args:
        file_path_out (str): Path to output video, must end with .avi
    """

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(file_path_out, fourcc, fps, (width, height))

    for frame_data in yolo_model(source=file_path_in, conf=0.1, show=True, stream=True):
        frame = frame_data.orig_img.copy()

        for box in frame_data.boxes.cpu().numpy():
            x1, y1, x2, y2 = box.xyxy.flatten().astype(int)
            label = frame_data.names[box.cls[0].astype(int)]
            score = box.conf

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            font_scale = 0.5
            text_margin = 5
            text = f"{label}: {score.item():.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[
                0
            ]

            text_x = min(x1, width - text_size[0] - text_margin)
            text_y = max(y1, text_size[1] + text_margin)

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                2,
            )
        writer.write(frame)
    writer.release()
    cv2.destroyAllWindows()


fps, width, height = get_video_params(VIDEO_IN)

write_video_with_bbox(
    VIDEO_IN, VIDEO_OUT, yolo_model=yolo_model, fps=fps, width=width, height=height
)


def write_video_as_set_of_bbox_frames(file_path_in, file_path_out, yolo_model):
    for frame_number, frame_data in enumerate(
        yolo_model(source=file_path_in, conf=0.45, show=True, stream=True)
    ):
        frame = frame_data.orig_img.copy()

        for box in frame_data.boxes.cpu().numpy():
            x1, y1, x2, y2 = box.xyxy.flatten().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(f"{file_path_out}/frame_{frame_number}.jpg", frame)

    cv2.destroyAllWindows()


fps, width, height = get_video_params(VIDEO_IN)

write_video_as_set_of_bbox_frames(VIDEO_IN, FRAME_OUT, yolo_model=yolo_model)


def write_video_with_keypoints(
    file_path_in, file_path_out, yolo_model, fps, width, height
):
    """Writes frames with pose estimations to an .avi video file"""

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(file_path_out, fourcc, fps, (width, height))

    for frame_data in yolo_model(
        source=file_path_in, conf=0.25, show=True, stream=True
    ):
        frame = frame_data.orig_img.copy()

        if frame_data.keypoints:
            for person in frame_data.keypoints.data:
                if person.shape[0] == 0:
                    continue
                for point in person:
                    x, y = map(int, point[:2])
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # draw joint

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
                for i, j in pairs:
                    x1, y1 = map(int, person[i][:2])
                    x2, y2 = map(int, person[j][:2])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # draw bone

        writer.write(frame)

    writer.release()
    cv2.destroyAllWindows()


write_video_with_keypoints(VIDEO_IN, VIDEO_OUT, yolo_model, fps, width, height)



from ultralytics import YOLO
from src.data.keypoints_handler import KeyPointsCSVWriter, KeyPointsVideoWriter


MODEL_PATH = "./model"
VIDEO_IN = "./data/raw/actions/shot/shot_KS_8_act1.avi"
# VIDEO_IN = "./data/raw/scenes/shot/shot_KS_10.mp4"
VIDEO_OUT = "./data/processed/output_25.avi"
CSV_OUT = "./data/processed/keypoints.csv"
yolo_model = YOLO(f"{MODEL_PATH}/yolov8n-pose.pt")
model = yolo_model(source=VIDEO_IN, conf=0.45, show=False, stream=True)

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


kp_csv_writer = KeyPointsCSVWriter(model)
kp_csv_list = kp_csv_writer.extract_keypoints_from_frames()

kp_csv_writer.write_keypoints_to_csv(CSV_OUT)



kp_video_writer = KeyPointsVideoWriter(pairs)
kp_video_list = kp_video_writer.read_keypoints_from_csv(VIDEO_IN, VIDEO_OUT, CSV_OUT)


for frame_keypoints in kp_list:
    print(frame_keypoints)
    for person_keypoints in frame_keypoints:
        print(person_keypoints)
        break
    break
        x, y = map(int, person_keypoints[:2])




kp_video_writer.write_video_with_keypoints(VIDEO_IN, VIDEO_OUT, CSV_OUT)

