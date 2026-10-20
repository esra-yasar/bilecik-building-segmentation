"""Run building segmentation on satellite images and report building coverage.

Usage:
    python predict.py --source assets/test_images
    python predict.py --source path/to/image.jpg --conf 0.4
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Building segmentation inference")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="assets/test_images", help="image file or folder")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,               # annotated images -> runs/segment/predict*
        project="runs/segment",
        name="predict",
        retina_masks=True,       # masks at original image resolution
    )

    print(f"\n{'image':<20}{'buildings':>10}{'coverage %':>12}")
    for r in results:
        name = Path(r.path).name
        if r.masks is None:
            print(f"{name:<20}{0:>10}{0.0:>12.2f}")
            continue
        masks = r.masks.data  # (N, H, W)
        union = masks.amax(dim=0)
        coverage = 100.0 * union.float().mean().item()
        print(f"{name:<20}{len(masks):>10}{coverage:>12.2f}")


if __name__ == "__main__":
    main()
