# Defect Detection - Toshiba & Galicia Datasets

This project performs semantic segmentation to detect defects on metallic surfaces using U-Net architectures (ConvNeXt, ResNet, EfficientNet).

## Structure

```
organized_experiments/
├── inference/                  # Inference scripts
│   ├── interactive_inference.py  # Interactive visualization tool with metrics
│   └── rt_inference.py           # Real-time inference (webcam/video)
├── UnetConvNeXtTiny/           # Training code and models for U-Net + ConvNeXt Tiny
├── Unet++Resnet34/             # Training code for U-Net++ + ResNet34
└── Unet++EfficienNetB4/        # Training code for U-Net++ + EfficientNet B4
```

## Setup

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Data**:
    Ensure the datasets are located in `../../datasets/final_dataset` relative to this folder, or adjust the paths in the scripts.

## Usage

### Interactive Inference

Run the interactive tool to visualize predictions and ground truth, and calculate metrics.

```bash
python inference/interactive_inference.py
```

*   **Controls**:
    *   `n`: Next image
    *   `p`: Previous image
    *   `m`: Calculate global metrics (Precision, Recall, IoU)
    *   `q`: Quit

### Real-Time Inference

Run inference on a video file or webcam.

```bash
python inference/rt_inference.py [video_path_or_camera_index]
```

Example: `python inference/rt_inference.py 0` (for webcam)
