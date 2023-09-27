import cv2


def _get_video_params(video_path_in) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path_in))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cv2.destroyAllWindows()
    return fps, width, height


def _video_writer(video_path_out, fps: int, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path_out), fourcc, fps, (width, height))
    return writer
