import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

class PoseEstimation:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

    def predict(self, frame):
        # Convert to RGB as MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = self.mp_pose.process(frame_rgb)
        keypoints = []
        # Draw pose landmarks on the frame
        if results.pose_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                keypoints.append((x, y))

        # Convert the frame back to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr, np.array(keypoints)