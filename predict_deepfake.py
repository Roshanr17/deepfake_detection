import argparse
from pathlib import Path

import torch

from train_deepfake_detector import build_model, read_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a video is real or fake.")
    parser.add_argument("video", help="Path to a video file.")
    parser.add_argument(
        "--model",
        default="models/deepfake_detector_best.pt",
        help="Path to a trained checkpoint.",
    )
    parser.add_argument("--frames-per-video", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", ["real", "fake"])

    frames_per_video = args.frames_per_video or config.get("frames_per_video", 8)
    image_size = args.image_size or config.get("image_size", 128)

    model_name = checkpoint.get("model_name", config.get("model", "small_cnn"))
    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    video = read_video_frames(
        Path(args.video),
        frames_per_video,
        image_size,
        train=False,
        face_crop=config.get("face_crop", True),
    )
    with torch.no_grad():
        logits = model(video.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_index = int(probabilities.argmax().item())
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.3f}")
    for class_name, probability in zip(class_names, probabilities.tolist()):
        print(f"{class_name}: {probability:.3f}")


if __name__ == "__main__":
    main()
