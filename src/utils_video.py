# src/utils_video.py
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from .config import NUM_FRAMES, IMAGE_SIZE

# -------------- transforms --------------
train_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
])

eval_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])

# -------------- crop settings --------------
# Final choice: horizontal crop [0.2W, 0.8W], full height [0H, 1.0H]
CROP_X1 = 0.2
CROP_X2 = 0.8
CROP_Y1 = 0.0
CROP_Y2 = 1.0


def _crop_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply normalized crop to a single frame.
    frame: H x W x 3 (RGB, uint8)
    """
    h, w = frame.shape[:2]
    x1 = int(CROP_X1 * w)
    x2 = int(CROP_X2 * w)
    y1 = int(CROP_Y1 * h)
    y2 = int(CROP_Y2 * h)

    # safety clamps
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return frame[y1:y2, x1:x2]


def load_and_sample_video(path: str,
                          num_frames: int = NUM_FRAMES,
                          is_train: bool = True) -> torch.Tensor:
    """
    Read mp4 with OpenCV, crop, sample num_frames evenly.
    Returns tensor [T, C, H, W] (float32, 0–1).
    """
    cap = cv2.VideoCapture(str(path))
    frames = []

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = _crop_frame(frame_rgb)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in video: {path}")

    # sample indices
    idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled = [frames[i] for i in idxs]

    transform = train_frame_transform if is_train else eval_frame_transform

    out_frames = []
    for f in sampled:
        f = transform(f)  # [C,H,W]
        out_frames.append(f)

    video_tensor = torch.stack(out_frames, dim=0)  # [T,C,H,W]
    return video_tensor
