import cv2
import numpy as np
import torch
import torchvision.transforms as T

from .config_3d import (
    NUM_FRAMES_3D,
    IMAGE_SIZE_3D,
    CROP_X1, CROP_X2, CROP_Y1, CROP_Y2,
)

train_frame_transform_3d = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE_3D, IMAGE_SIZE_3D)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
])

eval_frame_transform_3d = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE_3D, IMAGE_SIZE_3D)),
    T.ToTensor(),
])


def _optional_crop(frame: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape

    x1 = int(CROP_X1 * w)
    x2 = int(CROP_X2 * w)
    y1 = int(CROP_Y1 * h)
    y2 = int(CROP_Y2 * h)

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return frame[y1:y2, x1:x2, :]


def load_and_sample_video_3d(
    path: str,
    num_frames: int = NUM_FRAMES_3D,
    is_train: bool = True,
) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []

    if not cap.isOpened():
        raise RuntimeError(f"[3D] Could not open video: {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _optional_crop(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"[3D] No frames found in video: {path}")

    idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled = [frames[i] for i in idxs]

    transform = train_frame_transform_3d if is_train else eval_frame_transform_3d

    out_frames = []
    for f in sampled:
        f = transform(f)  
        out_frames.append(f)

    video_tensor = torch.stack(out_frames, dim=0)  
    return video_tensor
