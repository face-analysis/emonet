from typing import List, Dict
from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn
from skimage import io
from face_alignment.detection.sfd.sfd_detector import SFDDetector

from emonet.models import EmoNet

import cv2


def load_video(video_path: Path) -> List[np.ndarray]:
    """
    Loads a video using OpenCV.
    """
    video_capture = cv2.VideoCapture(video_path)

    list_frames_rgb = []

    # Reads all the frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        list_frames_rgb.append(image_rgb)

    return list_frames_rgb


def load_emonet(n_expression: int, device: str):
    """
    Loads the emotion recognition model.
    """

    # Loading the model
    state_dict_path = Path(__file__).parent.joinpath(
        "pretrained", f"emonet_{n_expression}.pth"
    )

    print(f"Loading the emonet model from {state_dict_path}.")
    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    return net


def run_emonet(
    emonet: torch.nn.Module, frame_rgb: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Runs the emotion recognition model on a single frame.
    """
    # Resize image to (256,256)
    image_rgb = cv2.resize(frame_rgb, (image_size, image_size))

    # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
    image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0

    with torch.no_grad():
        output = emonet(image_tensor.unsqueeze(0))

    return output


def plot_valence_arousal(
    valence: float, arousal: float, circumplex_size=512
) -> np.ndarray:
    """
    Assumes valence and arousal in range [-1;1].
    """
    circumplex_path = Path(__file__).parent / "images/circumplex.png"

    circumplex_image = cv2.imread(circumplex_path)
    circumplex_image = cv2.resize(circumplex_image, (circumplex_size, circumplex_size))

    # Position in range [0,circumplex_size/2] - arousal axis goes up, so need to take the opposite
    position = (
        (valence + 1.0) / 2.0 * circumplex_size,
        (1.0 - arousal) / 2.0 * circumplex_size,
    )

    cv2.circle(
        circumplex_image, (int(position[0]), int(position[1])), 16, (0, 0, 255), -1
    )

    return circumplex_image


def make_visualization(
    frame_rgb: np.ndarray,
    face_crop_rgb: np.ndarray,
    face_bbox: torch.Tensor,
    emotion_prediction: Dict[str, torch.Tensor],
    font_scale=2,
) -> np.ndarray:
    """
    Composes the final visualization with detected face, landmarks, discrete and continuous emotions.
    """
    # Visualize the detected face
    cv2.rectangle(
        frame_rgb,
        (face_bbox[0], face_bbox[1]),
        (face_bbox[2], face_bbox[3]),
        (255, 0, 0),
        8,
    )

    # Add the discrete emotion next to it
    predicted_emotion_class_idx = (
        torch.argmax(nn.functional.softmax(emotion_prediction["expression"], dim=1))
        .cpu()
        .item()
    )
    frame_rgb = cv2.putText(
        frame_rgb,
        emotion_classes[predicted_emotion_class_idx],
        ((face_bbox[0] + face_bbox[2]) // 2, face_bbox[1] + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Landmarks visualization
    # Resize to the original face_crop image size
    heatmap = torch.nn.functional.interpolate(
        emotion_prediction["heatmap"],
        (face_crop_rgb.shape[0], face_crop_rgb.shape[1]),
        mode="bilinear",
    )

    landmark_visualization = face_crop_rgb.copy()
    for landmark_idx in range(heatmap[0].shape[0]):
        # Detect the position of each landmark and draw a circle there
        landmark_position = (
            heatmap[0, landmark_idx, :, :] == torch.max(heatmap[0, landmark_idx, :, :])
        ).nonzero()
        cv2.circle(
            landmark_visualization,
            (
                int(landmark_position[0][1].cpu().item()),
                int(landmark_position[0][0].cpu().item()),
            ),
            4,
            (255, 255, 255),
            -1,
        )

    # Valence and arousal visualization
    circumplex_bgr = plot_valence_arousal(
        emotion_prediction["valence"].clamp(-1.0, 1.0),
        emotion_prediction["arousal"].clamp(-1.0, 1.0),
        frame_rgb.shape[0],
    )

    # Compose the final visualization
    visualization = np.zeros(
        (frame_rgb.shape[0], frame_rgb.shape[1] + frame_rgb.shape[0] // 2, 3),
        dtype=np.uint8,
    )

    # Resize the circumplex and face crop to match the frame size
    circumplex_bgr = cv2.resize(
        circumplex_bgr, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    landmark_visualization = cv2.resize(
        landmark_visualization, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    visualization[:, : frame_rgb.shape[1], :] = frame_rgb[:, :, ::-1].astype(np.uint8)
    visualization[
        : frame_rgb.shape[0] // 2, frame_rgb.shape[1] :, :
    ] = landmark_visualization[:, :, ::-1].astype(
        np.uint8
    )  # OpenCV needs BGR
    visualization[frame_rgb.shape[0] // 2 :, frame_rgb.shape[1] :, :] = (
        circumplex_bgr.astype(np.uint8)
    )

    return visualization


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nclasses",
        type=int,
        default=8,
        choices=[5, 8],
        help="Number of emotional classes to test the model on. Please use 5 or 8.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="video.mp4",
        help="Path to a video.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.mp4",
        help="Path where the output video is saved.",
    )

    args = parser.parse_args()

    # Parameters of the experiments
    n_expression = args.nclasses
    device = "cuda:0"
    image_size = 256
    emotion_classes = {
        0: "Neutral",
        1: "Happy",
        2: "Sad",
        3: "Surprise",
        4: "Fear",
        5: "Disgust",
        6: "Anger",
        7: "Contempt",
    }

    print(f"Loading emonet")
    emonet = load_emonet(n_expression, device)

    print(f"Loading face detector")
    sfd_detector = SFDDetector(device)

    print(f"Loading video")
    video_path = Path(__file__).parent / args.video_path
    list_frames_rgb = load_video(video_path)

    visualization_frames = []

    for i, frame in enumerate(list_frames_rgb):

        # Run face detector
        with torch.no_grad():
            # Face detector requires BGR frame
            detected_faces = sfd_detector.detect_from_image(frame[:, :, ::-1])

        # If at least a face has been detected, run emotion recognition on the first face
        if len(detected_faces)>0:
            # Only take the first detected face
            bbox = np.array(detected_faces[0]).astype(np.int32)

            face_crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
            emotion_prediction = run_emonet(emonet, face_crop.copy())

            visualization_bgr = make_visualization(
                frame.copy(), face_crop.copy(), bbox, emotion_prediction
            )
            visualization_frames.append(visualization_bgr)
        else:
            # Visualization without emotion
            visualization = np.zeros(
                (frame.shape[0], frame.shape[1] + frame.shape[0] // 2, 3),
                dtype=np.uint8,
            )
            visualization[:, : frame.shape[1], :] = frame[:, :, ::-1].astype(np.uint8)

            visualization_frames.append(visualization)

        if i % 100 == 0:
            print(f"Ran prediction on {i}/{len(list_frames_rgb)} frames")

    # Write the result as a video
    if visualization_frames:
        save_path = Path(__file__).parent / args.output_path

        out = cv2.VideoWriter(
            save_path,
            -1,
            24.0,
            (visualization_frames[0].shape[1], visualization_frames[0].shape[0]),
        )

        for frame in visualization_frames:
            out.write(frame)
