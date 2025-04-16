import time
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from collections import OrderedDict
import timm
from PIL import Image, ImageDraw
import glob
from torch.utils.data import Dataset
import numpy as np
import random
import torchvision.transforms.functional as FT
import json
import torch.cuda.amp as amp
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import box_iou

os.makedirs("visualizations", exist_ok=True)


def plot_training_curves(train_losses, val_maps, lr_history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, "b-")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.yscale("log")
    min_loss = min(train_losses) * 0.9
    max_loss = max(train_losses) * 1.1
    plt.ylim(min_loss, max_loss)

    plt.subplot(1, 3, 2)
    plt.plot(val_maps, "r-")
    plt.title("Validation mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True)
    min_map = max(0, min(val_maps) * 0.9)
    max_map = min(1.0, max(val_maps) * 1.1)
    plt.ylim(min_map, max_map)

    plt.subplot(1, 3, 3)
    plt.plot(lr_history, "g-")
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True)
    if max(lr_history) / (min(lr_history) + 1e-10) > 10:
        plt.yscale("log")
    min_lr = min(lr_history) * 0.9
    max_lr = max(lr_history) * 1.1
    plt.ylim(min_lr, max_lr)

    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)

    plt.savefig("visualizations/training_curves.png")
    plt.close()
    print("Training curves saved to visualizations/training_curves.png")


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


def plot_confusion_matrix(model, valid_dataloader, device):
    model.eval()
    confusion = np.zeros((10, 10), dtype=np.int32)

    with torch.no_grad():
        for images, targets in tqdm(valid_dataloader,
                                    desc="Creating confusion matrix"):
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)

                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]
                mask = pred_scores > 0.5
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]

                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)

                    matched_pred_indices = set()

                    for gt_idx in range(len(gt_boxes)):
                        if len(matched_pred_indices) == len(pred_boxes):
                            continue

                        unmatched_mask = torch.tensor(
                            [
                                i not in matched_pred_indices
                                for i in range(len(pred_boxes))
                            ],
                            device=device,
                        )

                        if not torch.any(unmatched_mask):
                            break

                        gt_ious = ious[unmatched_mask, gt_idx]

                        if len(gt_ious) == 0:
                            continue

                        max_iou, rel_idx = torch.max(gt_ious, dim=0)

                        unmatched_indices = torch.where(unmatched_mask)[0]
                        max_idx = unmatched_indices[rel_idx].item()

                        if max_iou > 0.5:
                            gt_digit = gt_labels[gt_idx].item() - 1
                            pred_digit = pred_labels[max_idx].item() - 1

                            if 0 <= gt_digit < 10 and 0 <= pred_digit < 10:
                                confusion[gt_digit, pred_digit] += 1

                            matched_pred_indices.add(max_idx)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("visualizations/confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to visualizations/confusion_matrix.png")
    return confusion


def denormalize_image(image, mean, std):
    img_copy = image.clone()

    for c in range(3):
        img_copy[c] = img_copy[c] * std[c] + mean[c]

    img_copy = torch.clamp(img_copy, 0, 1)

    return img_copy


def visualize_detections(model, dataset, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    # Define ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Calculate the indices for the last 5 images
    dataset_size = len(dataset)
    start_index = max(0, dataset_size - num_samples)

    for i in range(num_samples):
        # Get one of the last images from the dataset
        idx = start_index + i
        image, target = dataset[idx]

        # Denormalize the image first
        denormalized_image = denormalize_image(image, mean, std)
        # Convert to PIL image for left panel (ground truth)
        original_img = FT.to_pil_image(denormalized_image)

        # Draw ground truth boxes
        gt_img = original_img.copy()
        draw = ImageDraw.Draw(gt_img)
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()

        for box, label in zip(gt_boxes, gt_labels):
            draw.rectangle(box.tolist(), outline="red", width=2)
            draw.text((box[0], box[1]), str(label - 1), fill="red")

        axs[i, 0].imshow(gt_img)
        axs[i, 0].set_title(f"Ground Truth (Image {idx})")
        axs[i, 0].axis("off")

        # Run prediction
        with torch.no_grad():
            pred = model([image.to(device)])[0]

        # Draw predictions on a copy of the original image
        pred_img = original_img.copy()
        draw = ImageDraw.Draw(pred_img)

        # Filter predictions by confidence
        good_preds = pred["scores"] > 0.5
        boxes = pred["boxes"][good_preds].cpu().numpy()
        labels = pred["labels"][good_preds].cpu().numpy()
        scores = pred["scores"][good_preds].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            draw.rectangle(box.tolist(), outline="blue", width=2)
            draw.text((box[0], box[1]), f"{label-1}:{score:.2f}", fill="blue")

        axs[i, 1].imshow(pred_img)
        axs[i, 1].set_title(f"Predictions (Image {idx})")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("visualizations/sample_detections_last5.png")
    plt.close()

    print(
        f"""Sample detections from last {num_samples}
        images saved to visualizations/sample_detections_last5.png"""
    )


def visualize_feature_maps(model, dataset, num_samples=2):
    model.eval()

    # Create hooks to capture feature maps
    feature_maps = {}
    hooks = []

    # Identify feature map layers to visualize
    # For FPN/EFM this would be the output of each level
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()

        return hook

    for name, layer in model.named_modules():
        if "fpn" in name and isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    for i in range(num_samples):
        # Get a sample image
        image, target = dataset[i]

        # Run a forward pass to capture feature maps
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            model(image_tensor)

        # Original image for reference
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        denormalized_image = denormalize_image(image, mean, std)

        original_img = FT.to_pil_image(denormalized_image)

        # Plot the feature maps
        plt.figure(figsize=(15, 8))

        # First, show the original image
        plt.subplot(2, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis("off")

        # Then show feature maps
        count = 2
        # Show only first 5 feature maps
        for name, feature_map in sorted(feature_maps.items())[:5]:
            if count > 6:  # Limit to 5 feature maps + original image
                break

            # Get the feature map
            fmap = feature_map[0]  # First batch item

            # Average across channels for visualization
            fmap_avg = torch.mean(fmap, dim=0)

            # Normalize for better visualization
            fmap_min = fmap_avg.min()
            fmap_max = fmap_avg.max()
            fmap_norm = (fmap_avg - fmap_min) / (fmap_max - fmap_min + 1e-10)

            # Plot
            plt.subplot(2, 3, count)
            plt.imshow(fmap_norm.cpu().numpy(), cmap="viridis")
            plt.title(f"Feature Map {count-1}")
            plt.axis("off")
            count += 1

        plt.tight_layout()
        plt.savefig(f"visualizations/feature_maps_sample_{i+1}.png")
        plt.close()

        # Clear feature maps for next sample
        feature_maps.clear()

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    print("Feature maps saved to visualizations/feature_maps_sample_X.png")


def plot_bbox_size_distribution(dataset):
    # Collect box dimensions
    widths = []
    heights = []
    aspect_ratios = []

    for _, target in tqdm(dataset, desc="Analyzing bounding boxes"):
        boxes = target["boxes"]

        if len(boxes) > 0:
            # Calculate width and height for each box
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]

            # Store dimensions
            widths.extend(width.tolist())
            heights.extend(height.tolist())

            # Calculate aspect ratios (width/height)
            ratios = width / (height + 1e-10)  # Avoid division by zero
            aspect_ratios.extend(ratios.tolist())

    # Create the figure for multiple plots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot width distribution
    axs[0].hist(widths, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axs[0].set_title("Width Distribution", fontsize=14)
    axs[0].set_xlabel("Width (pixels)", fontsize=12)
    axs[0].set_ylabel("Count", fontsize=12)
    axs[0].grid(True, linestyle="--", alpha=0.7)

    # Plot height distribution
    axs[1].hist(
        heights,
        bins=30,
        color="lightgreen",
        edgecolor="black",
        alpha=0.7)
    axs[1].set_title("Height Distribution", fontsize=14)
    axs[1].set_xlabel("Height (pixels)", fontsize=12)
    axs[1].set_ylabel("Count", fontsize=12)
    axs[1].grid(True, linestyle="--", alpha=0.7)

    # Plot aspect ratio distribution
    axs[2].hist(
        aspect_ratios,
        bins=30,
        color="salmon",
        edgecolor="black",
        alpha=0.7)
    axs[2].set_title("Aspect Ratio Distribution", fontsize=14)
    axs[2].set_xlabel("Aspect Ratio (width/height)", fontsize=12)
    axs[2].set_ylabel("Count", fontsize=12)
    axs[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("visualizations/bbox_size_distribution.png")
    plt.close()
    print(
        """Bounding box size distribution saved to
        visualizations/bbox_size_distribution.png"""
    )


# Function to visualize model performance on different digit sizes
def analyze_detection_by_size(model, dataset, device):
    model.eval()

    # Initialize size bins and their performance metrics
    size_bins = [(0, 20), (20, 40), (40, 60), (60, 100), (100, float("inf"))]
    size_metrics = {bin_range: {"total": 0, "detected": 0}
                    for bin_range in size_bins}

    for i in range(len(dataset)):
        image, target = dataset[i]
        gt_boxes = target["boxes"]

        # Calculate sizes (area) of ground truth boxes
        gt_sizes = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
            (gt_boxes[:, 3] - gt_boxes[:, 1])

        # Run prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)

        pred_boxes = output[0]["boxes"].cpu()
        scores = output[0]["scores"].cpu()

        # Filter by confidence threshold
        mask = scores > 0.5
        pred_boxes = pred_boxes[mask]

        # Compute IoU between predicted and ground truth boxes
        for gt_idx, (gt_box, gt_size) in enumerate(zip(gt_boxes, gt_sizes)):
            # Find which size bin this ground truth box belongs to
            for bin_range in size_bins:
                min_size, max_size = bin_range
                if min_size <= gt_size < max_size:
                    size_metrics[bin_range]["total"] += 1

                    # Check if any predicted box matches this ground truth
                    detected = False
                    for pred_box in pred_boxes:
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > 0.5:  # IoU threshold
                            detected = True
                            break

                    if detected:
                        size_metrics[bin_range]["detected"] += 1

    # Calculate detection rates and prepare for plotting
    bin_labels = [f"{min_}-{max_}" for min_, max_ in size_bins]
    detection_rates = []

    for bin_range in size_bins:
        metrics = size_metrics[bin_range]
        rate = metrics["detected"] / \
            metrics["total"] if metrics["total"] > 0 else 0
        detection_rates.append(rate * 100)  # Convert to percentage

    plt.figure(figsize=(10, 6))
    plt.bar(bin_labels, detection_rates, color="lightgreen")
    plt.xlabel("Object Size (area in pixels)")
    plt.ylabel("Detection Rate (%)")
    plt.title("Detection Rate by Object Size")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("visualizations/detection_by_size.png")
    plt.close()
    print("Detection rate by size saved visualizations/detection_by_size.png")


# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Training configuration
min_size = 400
max_size = 600
batch_size = 16
epochs = 15
workers = 4
weight_decay = 5e-4
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Utility functions


def find_intersection(set_1, set_2):
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)
    )  # (n1, n2, 2)
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)
    )  # (n1, n2, 2)
    intersection_dims = torch.clamp(
        upper_bounds - lower_bounds,
        min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * \
        (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * \
        (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = (
        areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    )  # (n1, n2)

    return intersection / union  # (n1, n2)


def random_crop(image, boxes, labels):
    original_h = image.size(1)
    original_w = image.size(2)

    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # More aggressive cropping with wider range of overlaps
        min_overlap = random.choice([0.6, 0.7, 0.8, 0.9, None])

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

        # Try up to 50 times for this choice of minimum overlap
        max_trials = 50
        for _ in range(max_trials):
            # More varied scaling for better augmentation
            min_scale = 0.6 if random.random() > 0.7 else 0.75
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)
            # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater
            # than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (
                (bb_centers[:, 0] > left)
                * (bb_centers[:, 0] < right)
                * (bb_centers[:, 1] > top)
                * (bb_centers[:, 1] < bottom)
            )

            # If not a single bounding box has its center in the crop, try
            # again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(
                new_boxes[:, :2], crop[:2]
            )  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            # crop[2:] is [right, bottom]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def photometric_distort(image):
    new_image = image

    distortions = [
        FT.adjust_brightness,
        FT.adjust_contrast,
        FT.adjust_saturation,
        FT.adjust_hue,
    ]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == "adjust_hue":
                adjust_factor = random.uniform(-18 / 255.0, 18 / 255.0)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def get_optimized_anchor_sizes():
    # Digits typically have aspect ratios closer to 1:2 (height:width)
    # Smaller anchor sizes to better detect small digits
    anchor_sizes = ((32,), (64,), (96,), (128,), (160,))
    aspect_ratios = ((0.33, 0.5, 0.67),) * len(anchor_sizes)
    return anchor_sizes, aspect_ratios


def transform(image, boxes=[], labels=[], is_train=False):
    new_image = image
    new_boxes = boxes
    new_labels = labels

    # Skip the following operations for evaluation/testing
    if is_train:
        # Improved augmentation strategy with more aggressive transformations

        # Apply custom color jittering with higher probability
        if random.random() < 0.8:
            # Apply multiple distortions in sequence for more varied data
            distortions = [
                lambda img: FT.adjust_brightness(
                    img, random.uniform(0.7, 1.3)),
                lambda img: FT.adjust_contrast(img, random.uniform(0.7, 1.3)),
                lambda img: FT.adjust_saturation(
                    img, random.uniform(0.7, 1.3)),
                lambda img: FT.adjust_hue(img, random.uniform(-0.1, 0.1)),
            ]
            random.shuffle(distortions)

            # Apply 2-3 random distortions
            num_distortions = random.randint(1, 3)
            for distort in distortions[:num_distortions]:
                new_image = distort(new_image)

        # Add gaussian noise occasionally
        if random.random() < 0.2:
            new_image = np.array(new_image)
            noise = np.random.normal(0, 5, new_image.shape)
            new_image = np.clip(new_image + noise, 0, 255).astype(np.uint8)
            new_image = Image.fromarray(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Random crop with higher probability for better localization training
        if random.random() < 0.5:
            new_image, new_boxes, new_labels = random_crop(
                new_image, new_boxes, new_labels
            )

        # Add slight rotation occasionally (helpful for digit recognition)
        if random.random() < 0.2 and len(new_boxes) > 0:
            # Small angle to avoid severe distortion
            angle = random.uniform(-10, 10)
            # Convert to PIL for rotation
            pil_image = FT.to_pil_image(new_image)
            rotated_image = pil_image.rotate(
                angle, resample=Image.BILINEAR, expand=False
            )
            new_image = FT.to_tensor(rotated_image)

        # Random horizontal flip with careful consideration for digits
        if random.random() < 0.3:
            # Only flip if we don't have digits that would change meaning (6/9)
            can_flip = True
            for label in new_labels:
                if label == 6 or label == 9:
                    can_flip = False
                    break

            if can_flip:
                width = new_image.shape[2]
                new_image = FT.hflip(new_image)
                if len(new_boxes) > 0:
                    # Flip boxes coordinates
                    flipped_boxes = new_boxes.clone()
                    flipped_boxes[:, 0] = width - new_boxes[:, 2]
                    flipped_boxes[:, 2] = width - new_boxes[:, 0]
                    new_boxes = flipped_boxes

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Random grayscale to improve robustness to color variations
        if random.random() < 0.2:
            new_image = FT.to_grayscale(new_image, num_output_channels=3)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Normalize the image with ImageNet stats
        # This helps with transfer learning from pretrained models
        if random.random() < 0.8:  # Apply normalization most of the time
            new_image = FT.normalize(
                new_image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    else:
        # For validation/testing, just convert to tensor and normalize
        new_image = FT.to_tensor(new_image)
        # Consistently normalize validation images
        new_image = FT.normalize(
            new_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    return new_image, new_boxes, new_labels


def mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, device):
    _mAP = 0.0
    for thres in range(50, 100, 5):
        _mAP += calculate_AP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            thres / 100.0,
            device,
        )

    return _mAP / 10


# {background, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def calculate_AP(
    det_boxes,
    det_labels,
    det_scores,
    true_boxes,
    true_labels,
    threshold,
    device
):
    assert (
        len(det_boxes)
        == len(det_labels)
        == len(det_scores)
        == len(true_boxes)
        == len(true_labels)
    )
    n_classes = 11

    # Store all (true) objects in a single continuous tensor while keeping
    # track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects)
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track
    # of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert (
        det_images.size(0)
        == det_boxes.size(0)
        == det_labels.size(0)
        == det_scores.size(0)
    )

    # Calculate APs for each class (except background), class 10 is background
    average_precisions = torch.zeros(
        (n_classes - 1), dtype=torch.float
    )  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        n_easy_class_objects = true_class_images.size(0)

        if n_easy_class_objects == 0:
            continue

        true_class_boxes_detected = torch.zeros(
            (true_class_images.size(0)), dtype=torch.uint8
        ).to(
            device
        )  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(
            det_class_scores, dim=0, descending=True
        )  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections),
                                     dtype=torch.float).to(
            device
        )
        false_positives = torch.zeros((n_class_detections),
                                      dtype=torch.float).to(
            device
        )
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            object_boxes = true_class_boxes[true_class_images == this_image]
            # If no such object in this image, then the detection is a false
            # positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(
                overlaps.squeeze(0), dim=0)

            all_indices = torch.LongTensor(
                range(true_class_boxes.size(0))).to(device)
            image_mask = true_class_images == this_image
            original_ind = all_indices[image_mask][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's
            # a match
            if max_overlap.item() > threshold:
                # If this object has already not been detected, it's a true
                # positive
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    # this object has now been detected/accounted for
                    true_class_boxes_detected[original_ind] = 1
                # Otherwise, it's a false positive (since this object is
                # already accounted for)
                else:
                    false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the
            # actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the
        # order of decreasing scores
        cumul_true_positives = torch.cumsum(
            true_positives, dim=0
        )  # (n_class_detections)
        cumul_false_positives = torch.cumsum(
            false_positives, dim=0
        )  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
            cumul_true_positives + cumul_false_positives + 1e-10
        )  # (n_class_detections)
        cumul_recall = (
            cumul_true_positives / n_easy_class_objects
        )  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to
        # recalls above the threshold 't'
        recall_thresholds = torch.arange(
            start=0, end=1.1, step=0.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)),
                                 dtype=torch.float).to(
            device
        )  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.0
        # c is in [1, n_classes - 1]
        average_precisions[c - 1] = precisions.mean()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    return mean_average_precision


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


class DigitDataset(Dataset):

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        return images, targets

    def __len__(self):
        return len(self.images)

    def __init__(self, data_folder, annotation_file, is_train, split):
        self.data_folder = data_folder
        self.is_train = is_train

        # Load COCO format annotations if file is provided
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                coco_data = json.load(f)

            # Convert COCO annotations to a more convenient format
            self.annot_dict = {}

            # Create image_id to filename mapping
            id_to_filename = {}
            for image in coco_data["images"]:
                id_to_filename[image["id"]] = image["file_name"]

            # Create a reverse mapping from filename to id from the COCO data
            self.filename_to_id = {v: k for k, v in id_to_filename.items()}

            # Process annotations
            for ann in coco_data["annotations"]:
                image_id = ann["image_id"]
                filename = id_to_filename[image_id]

                # COCO format is [x_min, y_min, width, height]
                # Convert to [x_min, y_min, x_max, y_max]
                bbox = ann["bbox"]
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height

                # Category ID starts from 1
                category_id = ann["category_id"]

                if filename not in self.annot_dict:
                    self.annot_dict[filename] = {"labels": [], "boxes": []}

                self.annot_dict[filename]["labels"].append(category_id)
                self.annot_dict[filename]["boxes"].append(
                    [x_min, y_min, x_max, y_max])
        else:
            self.annot_dict = {}
            self.filename_to_id = {}

        # Get list of image files and sort them naturally
        self.images = sorted(
            glob.glob(os.path.join(data_folder, "*.png")), key=natural_sort_key
        )

        # Create a mapping file for reference
        if not os.path.exists("id_mappings"):
            os.makedirs("id_mappings")

        mapping_filename = (
            f"id_mappings/{'train' if is_train else 'valid'}_id_mappings.txt"
        )
        with open(mapping_filename, "w") as f:
            f.write("Filename,Original ID,Index in Dataset\n")
            for idx, img_path in enumerate(self.images):
                img_name = os.path.basename(img_path)
                # Use COCO ID if available, otherwise fall back to
                # filename-based ID
                original_id = self.filename_to_id.get(
                    img_name, int(os.path.splitext(img_name)[0])
                )
                f.write(f"{img_name},{original_id},{idx}\n")

    def __getitem__(self, i):
        # Read image
        image_path = self.images[i]
        image = Image.open(image_path)
        image = image.convert("RGB")
        img_name = os.path.basename(image_path)

        # Read objects in this image (bounding boxes, labels)
        if img_name in self.annot_dict:
            labels = self.annot_dict[img_name]["labels"]
            boxes = self.annot_dict[img_name]["boxes"]
            boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
            labels = torch.LongTensor(labels)  # (n_objects)
        else:
            # Handle case where image has no annotations
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)

        # Apply transformations
        image, boxes, labels = transform(
            image, boxes, labels, is_train=self.is_train)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["img_name"] = img_name

        # Use COCO ID if available, otherwise fall back to filename-based ID
        target["image_id"] = self.filename_to_id.get(
            img_name, int(os.path.splitext(img_name)[0])
        )

        return image, target


def get_test_img(idx):
    img_filename = "test/{}.png".format(idx)

    image = Image.open(img_filename)
    image = image.convert("RGB")

    image, _, _ = transform(image)

    return image


def get_speed_test_img(idx):
    img_filename = "for_speed_test/{}.png".format(idx)

    image = Image.open(img_filename)
    image = image.convert("RGB")

    image, _, _ = transform(image)

    return image


# Model definition
def _conv2d(in_channels, out_channels, size=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            size,
            stride=1,
            padding=size // 2),
        torch.nn.BatchNorm2d(out_channels),
    )


class ExactFusionModel(torch.nn.Module):
    def __init__(
        self,
        in_channels_list,
        out_channels,
        transition=128,
        withproduction=True,
        extra_blocks=None,
    ):
        if len(in_channels_list) < 4:
            raise ("length of in_channels_list must be longer than 3")
        super(ExactFusionModel, self).__init__()
        self.in_channels_list = in_channels_list
        self.same_blocks = nn.ModuleList()
        self.prod_blocks = nn.ModuleList()
        self.upto_blocks = nn.ModuleList()
        self.extra_blocks = extra_blocks

        b_index = len(in_channels_list) - 1
        up_channel = (
            self.in_channels_list[b_index] +
            self.in_channels_list[b_index - 1] // 2
        )
        self.efm_channels = [up_channel]

        b_index -= 1
        while b_index > 0:
            channels = (
                self.in_channels_list[b_index] // 2
                + self.in_channels_list[b_index - 1] // 2
                + transition
            )
            up_channel = channels
            self.efm_channels.insert(0, channels)
            b_index -= 1

        for i, in_channel in enumerate(self.efm_channels):
            self.same_blocks.append(_conv2d(in_channel, out_channels, 3))
            self.prod_blocks.append(
                _conv2d(out_channels, out_channels, 3)
                if withproduction
                else torch.nn.Identity()
            )
            self.upto_blocks.append(_conv2d(in_channel, transition, 1))

        for m in self.children():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                torch.nn.init.constant_(m.bias, 0)

    # input must be dict, from bottom to top
    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())

        xb_index = len(x) - 1
        shape = x[xb_index].shape[-2:]
        csp_x = [
            torch.cat(
                [
                    torch.nn.functional.interpolate(
                        x[xb_index - 1][:,
                                        self.in_channels_list[xb_index-1]//2:],
                        size=shape,
                        mode="nearest",
                    ),
                    x[xb_index],
                ],
                1,
            )
        ]
        xb_index -= 1

        while xb_index > 0:
            shape = x[xb_index].shape[-2:]
            csp_x.insert(
                0,
                torch.cat(
                    [
                        torch.nn.functional.interpolate(
                            x[xb_index - 1][
                                :, self.in_channels_list[xb_index - 1] // 2:
                            ],
                            size=shape,
                            mode="nearest",
                        ),
                        x[xb_index][:, : self.in_channels_list[xb_index] // 2],
                        torch.nn.functional.interpolate(
                            self.upto_blocks[xb_index](csp_x[0]),
                            size=shape,
                            mode="nearest",
                        ),
                    ],
                    1,
                ),
            )
            xb_index -= 1

        bottom_feature = self.same_blocks[0](csp_x[0])
        result = [self.prod_blocks[0](bottom_feature)]
        for csp, same_block, prod_block in zip(
            csp_x[1:], self.same_blocks[1:], self.prod_blocks[1:]
        ):
            shape = csp.shape[-2:]
            feature = same_block(csp)
            feature = feature + torch.nn.functional.interpolate(
                bottom_feature, size=shape, mode="nearest"
            )
            bottom_feature = feature
            result.append(prod_block(feature))

        if self.extra_blocks is not None:
            result, names = self.extra_blocks(result, x, names)

        out = OrderedDict((k, v) for k, v in zip(names[1:], result))

        return out


class CSPIntermediateLayer(nn.ModuleDict):
    def __init__(self, model: nn.Module) -> None:
        layers = OrderedDict()
        layers["stem"] = model.stem
        layers["0"] = model.stages[0]
        layers["1"] = model.stages[1]
        layers["2"] = model.stages[2]
        layers["3"] = model.stages[3]

        super(CSPIntermediateLayer, self).__init__(layers)
        self.return_layers = ["0", "1", "2", "3"]

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[name] = x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels //
                reduction_ratio,
                1,
                bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels //
                reduction_ratio,
                in_channels,
                1,
                bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CSPWithEFM(nn.Module):
    def __init__(self):
        super(CSPWithEFM, self).__init__()
        ops = torchvision.ops
        extra_blocks = ops.feature_pyramid_network.LastLevelMaxPool()
        backbone = timm.models.cspresnet50(pretrained=True)

        # Unfreeze more layers for better adaptation
        layers_to_train = [
            "stem",
            "stages.0",
            "stages.1",
            "stages.2",
            "stages.3"]

        for name, parameter in backbone.named_parameters():
            if any([name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)

        self.body = CSPIntermediateLayer(backbone)

        # Add attention mechanism for better feature fusion
        self.fpn = ExactFusionModel(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=256,
            transition=192,  # Increased transition channels
            extra_blocks=extra_blocks,
        )

        # Add channel attention modules for better feature refinement
        # Create attention modules for the FPN output keys, not the backbone
        # keys
        self.attention = nn.ModuleDict(
            {
                "0": ChannelAttention(256),
                "1": ChannelAttention(256),
                "2": ChannelAttention(256),
                "3": ChannelAttention(256),
                "pool": ChannelAttention(256),
            }
        )

        self.out_channels = 256

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)

        # Apply attention to each feature level using the correct keys
        for k in x.keys():
            # Only apply attention if we have an attention module for this key
            if k in self.attention:
                x[k] = self.attention[k](x[k]) * x[k]

        return x


# Main training function
def train_digit_detector(train_dir, valid_dir, train_json, valid_json):
    print("Preparing datasets...")
    # Use the provided paths for train and validation data
    train_dataset = DigitDataset(
        data_folder=train_dir,
        annotation_file=train_json,
        is_train=True,
        split=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True,
    )
    valid_dataset = DigitDataset(
        data_folder=valid_dir,
        annotation_file=valid_json,
        is_train=False,
        split=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        num_workers=workers,
    )

    # Define model
    print("Creating model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        pretrained=True, min_size=min_size, max_size=max_size
    )

    # Replace the backbone with improved CSP backbone
    backbone = CSPWithEFM()

    # Create a dummy input to get feature map dimensions
    dummy_input = torch.zeros(1, 3, min_size, min_size)
    features = backbone(dummy_input)
    num_feature_maps = len(features)

    model.backbone = backbone

    anchor_sizes = tuple((s,)
                         for s in [32, 64, 96, 128, 160][:num_feature_maps])
    aspect_ratios = ((0.33, 0.5, 0.67),) * num_feature_maps

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator

    # Update RPN head
    in_channels = model.backbone.out_channels  # Should be 256
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    from torchvision.models.detection.rpn import RPNHead

    model.rpn.head = RPNHead(in_channels, num_anchors)

    # Set number of classes
    num_classes = 11  # 10 digits (0-9) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to device
    model = model.to(device)

    # Initialize parameters with proper weights
    for name, param in model.named_parameters():
        if "bbox_pred" in name or "cls_score" in name:
            if "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.normal_(param, mean=0, std=0.01)

    # Optimizer with improved parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=0.001,  # Starting learning rate
        weight_decay=weight_decay,
        amsgrad=True,  # Enable AMSGrad variant
    )

    # Calculate steps per epoch for OneCycleLR
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs

    # Use OneCycleLR scheduler with proper parameters
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,  # Slightly higher max learning rate
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy="cos",
    )

    # Mixed precision scaler
    scaler = amp.GradScaler()

    # Tracking metrics
    best_mAP = 0.0
    train_losses = []
    val_maps = []
    lr_history = []

    # Training loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        # Training phase
        model.train()
        epoch_loss = 0

        train_iter = tqdm(
            train_dataloader,
            desc=f"Training epoch {epoch}/{epochs}",
            unit="batch"
        )
        for i, (images, targets) in enumerate(train_iter):
            # Move data to device
            images = list(image.to(device) for image in images)
            targets = [
                {"boxes": t["boxes"].to(device),
                 "labels": t["labels"].to(device)}
                for t in targets
            ]

            # Forward pass with mixed precision
            with amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            epoch_loss += losses.item()

            # Backward pass and optimization with gradient scaling
            optimizer.zero_grad()
            scaler.scale(losses).backward()

            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            scaler.step(optimizer)
            scaler.update()

            # Step the scheduler after each batch - THIS IS THE FIX
            scheduler.step()

            # Update progress bar with current loss and learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            train_iter.set_postfix(loss=losses.item(), lr=current_lr)

            # Record current learning rate periodically
            if i % 20 == 0:
                lr_history.append(current_lr)

        # Calculate average loss for this epoch
        avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        # Print current learning rate at the end of the epoch
        print(f"Current learning rate: {current_lr:.7f}")

        # Validation phase
        model.eval()
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()

        with torch.no_grad():
            val_start = time.time()
            # Add tqdm progress bar for validation
            val_iter = tqdm(valid_dataloader, desc="Validating", unit="image")
            for i, (images, targets) in enumerate(val_iter):
                images = list(image.to(device) for image in images)

                output = model(images)

                # Store batch results for mAP calculation
                boxes = [t["boxes"].to(device) for t in targets]
                labels = [t["labels"].to(device) for t in targets]

                det_boxes.extend([o["boxes"] for o in output])
                det_labels.extend([o["labels"] for o in output])
                det_scores.extend([o["scores"] for o in output])
                true_boxes.extend(boxes)
                true_labels.extend(labels)

            # Calculate mAP
            current_mAP = mAP(
                det_boxes, det_labels, det_scores,
                true_boxes, true_labels, device
            )
            val_maps.append(current_mAP)
            val_end = time.time()
            print(
                "Validation mAP: {:.4f}, Time: {:.3f}s".format(
                    current_mAP, val_end - val_start
                )
            )

            # Save best model
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                torch.save(model.state_dict(), "best_faster_rcnn_model.pth")
                print(f"New best model saved with mAP: {best_mAP:.4f}")

        # Plot training curves after each epoch
        plot_training_curves(train_losses, val_maps, lr_history)

    # Save final model
    torch.save(model.state_dict(), "final_faster_rcnn_model.pth")
    torch.save(model, "full_faster_rcnn_model.pth")

    print(f"Training completed! Best mAP: {best_mAP:.4f}")

    # Generate visualizations with the trained model
    print("Generating confusion matrix...")
    plot_confusion_matrix(model, valid_dataloader, device)

    print("Generating detection visualizations...")
    visualize_detections(model, valid_dataset, num_samples=5)

    print("Generating feature maps...")
    visualize_feature_maps(model, valid_dataset, num_samples=2)

    print("All visualizations completed!")

    return best_mAP


def predict_on_test_images(model, test_folder, output_json, output_csv):
    model.eval()

    test_images = sorted(
        glob.glob(os.path.join(test_folder, "*.png")), key=natural_sort_key
    )

    if not os.path.exists("id_mappings"):
        os.makedirs("id_mappings")

    mapping_file = "id_mappings/test_id_mappings.txt"
    with open(mapping_file, "w") as f:
        f.write("Filename,Extracted ID,Index in List\n")
        for idx, img_path in enumerate(test_images):
            img_name = os.path.basename(img_path)
            image_id = int(os.path.splitext(img_name)[0])
            f.write(f"{img_name},{image_id},{idx}\n")
    print(f"Test mapping saved to {mapping_file}")

    # Results for both tasks
    detection_results = []
    recognition_results = []

    with torch.no_grad():
        for img_path in tqdm(
                test_images, desc="Processing images", unit="image"):
            # Get image ID from filename
            img_name = os.path.basename(img_path)
            image_id = int(os.path.splitext(img_name)[0])

            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            image_tensor, _, _ = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Run inference
            output = model(image_tensor)

            # Process detections for Task 1
            boxes = output[0]["boxes"].cpu().numpy()
            scores = output[0]["scores"].cpu().numpy()
            labels = output[0]["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5:
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min

                    if label == 0:
                        continue

                    detection_results.append(
                        {
                            "image_id": image_id,
                            "bbox": [
                                float(x_min),
                                float(y_min),
                                float(width),
                                float(height),
                            ],
                            "score": float(score),
                            "category_id": int(label),
                        }
                    )

            # Process for Task 2 (whole number recognition)
            mask = output[0]["scores"] > 0.5
            filtered_boxes = output[0]["boxes"][mask].cpu()
            filtered_labels = output[0]["labels"][mask].cpu()

            if len(filtered_boxes) > 0:
                # Sort by x-coordinate
                sorted_indices = filtered_boxes[:, 0].argsort()
                sorted_labels = filtered_labels[sorted_indices]

                whole_number = "".join([str(lst.item() - 1)
                                       for lst in sorted_labels])
            else:
                whole_number = "-1"

            recognition_results.append(
                {"image_id": image_id, "pred_label": whole_number}
            )

    with open(output_json, "w") as f:
        json.dump(detection_results, f)

    with open(output_csv, "w") as f:
        f.write("image_id,pred_label\n")
        for result in recognition_results:
            f.write(f"{result['image_id']},{result['pred_label']}\n")

    print(f"Detection results saved to {output_json}")
    print(f"Recognition results saved to {output_csv}")


def main():
    data_root = "./nycu-hw2-data"
    train_folder = os.path.join(data_root, "train")
    valid_folder = os.path.join(data_root, "valid")
    test_folder = os.path.join(data_root, "test")
    train_annotations = os.path.join(data_root, "train.json")
    valid_annotations = os.path.join(data_root, "valid.json")

    # Switch between training and testing
    do_training = True

    if do_training:
        train_digit_detector(
            train_folder, valid_folder, train_annotations, valid_annotations
        )
    else:
        model_path = "best_faster_rcnn_model.pth"

        backbone = CSPWithEFM()

        dummy_input = torch.zeros(1, 3, min_size, min_size)
        features = backbone(dummy_input)
        num_feat_maps = len(features)

        anchor_sizes = tuple((s,)
                             for s in [32, 64, 96, 128, 160][:num_feat_maps])
        aspect_ratios = ((0.33, 0.5, 0.67),) * num_feat_maps

        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        # Load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            pretrained=False, min_size=min_size, max_size=max_size
        )

        # Set the backbone and anchor generator
        model.backbone = backbone
        model.rpn.anchor_generator = anchor_generator

        in_channels = model.backbone.out_channels
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        from torchvision.models.detection.rpn import RPNHead

        model.rpn.head = RPNHead(in_channels, num_anchors)

        num_classes = 11
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        predict_on_test_images(
            model,
            test_folder,
            "pred.json",
            "pred.csv")


if __name__ == "__main__":
    main()
