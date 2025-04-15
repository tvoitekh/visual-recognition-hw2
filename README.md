# NYCU Computer Vision 2025 Spring HW2

StudentID: 111550203  
Name: 提姆西

## Introduction

This project implements a digit recognition system using a modified Faster R-CNN architecture. The system tackles a dual challenge:
1. Detecting individual digits in images with their bounding boxes and classes
2. Recognizing the complete number by combining detected digits in their spatial order

The core approach enhances the standard Faster R-CNN model by incorporating a CSP (Cross Stage Partial Network) backbone with EFM (Enhanced Feature Module), optimizing anchor sizes and aspect ratios specifically for digit detection, and implementing advanced training techniques such as mixed precision training and adaptive learning rate scheduling.

## How to install

1. Clone this repository:
```bash
git clone https://github.com/tvoitekh/visual-recognition-hw2.git
cd visual-recognition-hw2
```

2. Make sure you have the data directory structure:
```
./nycu-hw2-data
├── train/
├── valid/
├── test/
├── train.json
└── valid.json
```

3. Run the model:
   - For training:
     ```python
     # Set do_training = True in train_digit_detector.py
     python train_digit_detector.py
     ```
   - For inference:
     ```python
     # Set do_training = False in train_digit_detector.py
     python train_digit_detector.py
     ```

## How to install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Performance snapshot

<img width="1151" alt="image" src="https://github.com/user-attachments/assets/a8d2822f-d1bc-442b-858d-4143b9971eb1" />


The model achieves a validation mAP of approximately 47.30 on the digit detection task. 

Key performance features:
- Custom CSPWithEFM backbone outperforms standard ResNet50-FPN
- Optimized anchor sizes [32, 64, 96, 128, 160] and aspect ratios [0.33, 0.5, 0.67]
- Mixed precision training with OneCycleLR scheduler
- Advanced feature fusion through ExactFusionModel
- Channel attention mechanism for adaptive feature refinement

## Code Linting

The following commands have been run as well as manual modifications performed:

```bash
autopep8 --in-place --aggressive --max-line-length 79 train_digit_detector.py
```

```bash
black --line-length 79 train_digit_detector.py
```

<img width="791" alt="image" src="https://github.com/user-attachments/assets/a77978c7-baea-41c0-ae83-31fc86e143c1" />

As can be seen no warnings or errors are present. This verifies that the code had been successfully linted as required.
