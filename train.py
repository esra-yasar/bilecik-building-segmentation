"""Train YOLOv8s-seg for building segmentation on Bilecik satellite imagery.

Hyperparameters are identical to the original Colab run
(see notebooks/02_final_yolov8s_seg.ipynb and the train_args stored in weights/best.pt).
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8s-seg.pt")  # COCO-pretrained small segmentation model
    model.train(
        data="data/data.yaml",
        epochs=300,
        patience=0,          # early stopping disabled: run all epochs
        batch=16,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,         # cosine learning-rate schedule
        # on-the-fly augmentation (nothing is copied to disk)
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        scale=0.4,
        degrees=15.0,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        fliplr=0.5,
        close_mosaic=10,
        seed=0,
        project="runs",
        name="yolov8s_seg_buildings",
    )


if __name__ == "__main__":
    main()
