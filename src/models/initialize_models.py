from ultralytics import YOLO


def initialize_yolo_model(path_to_model: str):
    """Initialize and return the YOLO model."""
    try:
        model = YOLO(path_to_model)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize model: {exc}") from exc
