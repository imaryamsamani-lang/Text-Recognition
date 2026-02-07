## Handwritten Text Detection with DBNet

A fine-tuned implementation of DBNet (Differentiable Binarization Network) optimized for detecting handwritten text in scanned documents, notes, and forms. This model addresses challenges specific to handwriting including curvature, irregular layouts, and varied writing styles.

🔗 Original DBNet repository:
[DBNet](https://github.com/WenmuZhou/DBNet.pytorch)

## Overview
Optical text detection in handwritten documents presents unique challenges compared to printed text, including inconsistent baselines, character connectivity variations, and diverse writing styles. This implementation adapts the DBNet architecture—originally designed for scene text detection—to specialize in handwritten content through targeted fine-tuning on custom datasets.

The model builds upon the original DBNet framework, with modifications to training protocols and data augmentation strategies optimized for handwritten text characteristics. DBNet formulates text detection as a segmentation problem with a differentiable binarization module. This approach allows the network to learn adaptive thresholds during training, enabling precise text region segmentation without post-processing parameter tuning.

## Key Features

Handwriting-optimized detection – Fine-tuned specifically for handwritten text across diverse writing styles

Curved text support – Detects irregular text baselines common in natural handwriting

Rotation robustness – Handles document rotations and skewed scanning

Polygon-based detection – Generates precise polygonal bounding regions around text instances

Dense text capability – Effectively separates adjacent handwritten lines and words

Practical inference speed – Suitable for batch processing of scanned documents

## Installation

### Repository Structure
```bash
├── net/                       # DBNet neural network architectures
├── utils/                     # Utility functions (data loading, losses, metrics)
├── Dataset/
│   ├── train/
│   │   ├── images/            # Training images (.jpg, .png)
│   │   └── labels/            # Training annotations (.txt, ICDAR format)
│   ├── val/
│   │   ├── images/            # Validation images
│   │   └── labels/            # Validation annotations
│   └── test/
│       ├── images/            # Test images
│       └── labels/            # Test annotations
│
│   ├── train.txt              # Training image list (filenames)
│   ├── val.txt                # Validation image list
│   └── test.txt               # Test image list
│
├── weights/                   # Saved model checkpoints (.pth)
├── CV_DBNet.ipynb             # End-to-end training & evaluation notebook
├── inference.py               # Single-image inference script
├── test.py                    # Quantitative evaluation on test set
└── requirements.txt           # Python dependencies
```

### Prerequisites

Python 3.8+

CUDA-compatible GPU (recommended for training)

PyTorch (≥1.8.0)

### Environmental Setup

```bash
# Clone repository
git clone https://github.com/imaryamsamani-lang/Text-Recognition.git
cd Text-Recognition

# Create and activate conda environment
conda create -n dbnet python=3.8 -y
conda activate dbnet

# Install dependencies
pip install -r requirements.txt
```

### PyTorch Installation
Install the appropriate PyTorch version for your system:

```bash
# Example for CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Data Preparation

### Annotation Format
The model expects polygon-level annotations for text regions. Each image should have a corresponding .txt file with the following structure:

```txt
x1, y1, x2, y2, x3, y3, x4, y4, annotation

# example
627,69,629,14,113,24,109,76, ###
121,132,451,125,453,61,119,69, ###
130,179,487,171,483,125,128,121, ###
```

### Dataset Organization

```text
Dataset/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── labels/     # Corresponding txt annotations
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation annotations
└── test/
    ├── images/     # Test images
    └── labels/     # Test annotations
```

## Model Weights
Download the pre-trained weights fine-tuned for handwritten text detection: [db_best.pt](https://drive.google.com/file/d/1GWDVhdM54axJXbb-Buommhr3FlpDecPL/view?usp=sharing)

Place the downloaded weights file in the weights/ directory.

## Usage

### Inference

Process individual images for text detection:

```bash
python inference.py --image path/to/image.jpg --weights weights/db_best.pt
```

Optional arguments:

--output_dir: Directory to save visualization results

--threshold: Confidence threshold for detection (default: 0.3)

--visualize: Generate visualization overlay

### Evaluation

Evaluate model performance on a test set:

```bash
python test.py --test_dir Dataset/test --weights weights/db_best.pt
```

The evaluation script computes precision, recall, and F-score metrics.

### Training

For custom fine-tuning, use the comprehensive training notebook:

```bash
jupyter notebook CV_DBNet.ipynb
```

Key training configurations:

Backbone: ResNet-18 or ResNet-50

Input size: Adjustable based on document resolution

Optimizer: Adam with cosine annealing schedule

Batch size: Configurable based on GPU memory

Loss: Combined probability map and threshold map losses

## Performance
The fine-tuned model achieves the following performance on handwritten text detection:

| Metric | Score |
|--------|-------|
| Precision | 0.99 |
| Recall | 0.73 |
| F-score | 0.84 |

Note: Performance may vary based on handwriting style, image quality, and document complexity.

## Results Visualization

Example detections demonstrating model capability on diverse handwritten content:

![Diagram](results/1.png)
![Diagram](results/2.png)
![Diagram](results/3.png)
![Diagram](results/4.png)
