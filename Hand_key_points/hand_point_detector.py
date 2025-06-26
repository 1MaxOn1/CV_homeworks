import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import deque
from hand_points import ShallowUNet
from torchvision import transforms

MODEL_NEURONS = 16

class HandKeypointTracker:
    def __init__(self, model_path, device):
        self.n_keypoints = 21
        self.n_img_channels = 3
        self.raw_img_size = 224
        self.model_img_size = 128
        self.dataset_means = [0.3950, 0.4323, 0.2954]
        self.dataset_stds = [0.1966, 0.1734, 0.1836]
        
        self.device = device
        self.keypoint_index_to_track = 7
        self.model = self._load_model(model_path)
        
        self.preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.model_img_size, self.model_img_size), antialias=True),
            transforms.Normalize(mean=self.dataset_means, std=self.dataset_stds)
        ])
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.cap = cv2.VideoCapture(0)
        self.trajectory_points = deque(maxlen=64)
        self.smoothed_point = None
        self.smoothing_factor = 0.2

    def _load_model(self, model_path):
        model = ShallowUNet(in_channel=self.n_img_channels, out_channel=self.n_keypoints).to(self.device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"ОШИБКА: Файл с весами не найден по пути '{model_path}'")
            exit()
        model.eval()
        return model

    def _preprocess_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess_transform(image_rgb)
        return tensor.unsqueeze(0).to(self.device)

    def _get_keypoint_from_output(self, model_output, crop_coords, crop_shape):
        heatmap = model_output[0, self.keypoint_index_to_track, :, :].cpu().numpy()
        heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)

        y_pred, x_pred = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        x_min, y_min = crop_coords
        crop_h, crop_w, _ = crop_shape
        
        abs_x_in_crop = (x_pred / self.model_img_size) * crop_w
        abs_y_in_crop = (y_pred / self.model_img_size) * crop_h
        
        final_x = int(x_min + abs_x_in_crop)
        final_y = int(y_min + abs_y_in_crop)
        
        return (final_x, final_y)

    def _draw_trajectory(self, frame):
        for i in range(1, len(self.trajectory_points)):
            if self.trajectory_points[i - 1] is None or self.trajectory_points[i] is None:
                continue
            thickness = int(np.sqrt(len(self.trajectory_points) / float(i + 1)) * 2.5)
            cv2.line(frame, self.trajectory_points[i - 1], self.trajectory_points[i], (0, 255, 0), thickness)

    def run(self):
        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть веб-камеру.")
            return

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                frame_height, frame_width, _ = frame.shape
                x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
                y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]

                x_min_bbox, x_max_bbox = min(x_coords), max(x_coords)
                y_min_bbox, y_max_bbox = min(y_coords), max(y_coords)
                bbox_width = x_max_bbox - x_min_bbox
                bbox_height = y_max_bbox - y_min_bbox
                center_x = x_min_bbox + bbox_width / 2
                center_y = y_min_bbox + bbox_height / 2

                padding_factor = 1.5
                crop_size = int(max(bbox_width, bbox_height) * padding_factor)

                x_min = max(0, int(center_x - crop_size / 2))
                y_min = max(0, int(center_y - crop_size / 2))
                x_max = min(frame_width, int(center_x + crop_size / 2))
                y_max = min(frame_height, int(center_y + crop_size / 2))

                if x_min >= x_max or y_min >= y_max:
                    continue

                hand_crop_bgr = frame[y_min:y_max, x_min:x_max]

                if hand_crop_bgr.size > 0:
                    input_tensor = self._preprocess_image(hand_crop_bgr)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                    
                    raw_keypoint_coords = self._get_keypoint_from_output(
                        output, 
                        crop_coords=(x_min, y_min), 
                        crop_shape=hand_crop_bgr.shape
                    )

                    if self.smoothed_point is None:
                        self.smoothed_point = raw_keypoint_coords
                    else:
                        sx = int(self.smoothing_factor * raw_keypoint_coords[0] + (1 - self.smoothing_factor) * self.smoothed_point[0])
                        sy = int(self.smoothing_factor * raw_keypoint_coords[1] + (1 - self.smoothing_factor) * self.smoothed_point[1])
                        self.smoothed_point = (sx, sy)
                    
                    self.trajectory_points.appendleft(self.smoothed_point)
                    cv2.circle(frame, self.smoothed_point, 5, (0, 0, 255), -1)
                    mp_point = hand_landmarks.landmark[8]
                    mp_x = int(mp_point.x * frame_width)
                    mp_y = int(mp_point.y * frame_height)
                    cv2.circle(frame, (mp_x, mp_y), 10, (255, 0, 0), 2)
                else:
                    self.trajectory_points.appendleft(None)

            else:
                self.trajectory_points.clear()
                self.smoothed_point = None

            self._draw_trajectory(frame)
            cv2.imshow('Hand Keypoint Tracking', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()