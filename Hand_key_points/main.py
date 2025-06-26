import torch
from hand_point_detector import HandKeypointTracker

if __name__ == "__main__":
    PATH_TO_WEIGHTS = r"D:\CV_homework\hand_det\model_final"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tracker = HandKeypointTracker(
        model_path=PATH_TO_WEIGHTS,
        device=DEVICE
    )
    tracker.run()