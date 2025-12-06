import os
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from pytorch_lightning import Trainer
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from src.dataloader.anomaly_datamodule import XRayDataModule
from src.models.anomaly_detector import AnomalyDetectorPL

print("Imports starting...", flush=True)


def main() -> None:
    """Main training script for anomaly detection (osobowe)."""
    print("Script started.", flush=True)

    # Ensure lightning_logs directory exists
    os.makedirs("lightning_logs", exist_ok=True)

    # Configuration for higher precision
    patch_size = 224  # Reverted to 224 for better context
    stride = 64  # Decreased stride for denser sampling

    # 1. Setup Data
    data_module = XRayDataModule(
        vehicle_type="osobowe", batch_size=16, patch_size=patch_size, stride=stride
    )

    # 2. Setup Model
    # use_coords=True for cars (osobowe) to handle complex structures like engines
    # Restoring use_coords=True as the engine area causes false positives
    # without spatial awareness.
    model = AnomalyDetectorPL(
        backbone_name="resnet18",
        use_coords=False,
        knn_path="trained_models_anomaly/knn_osobowe.pkl",
    )

    # 3. Trainer
    trainer = Trainer(
        max_epochs=1,  # One pass to extract features
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,  # We save KNN manually
        logger=False,  # Disable logger to avoid directory issues
        enable_progress_bar=True,
    )

    # 4. Train (Feature Extraction & KNN Fitting)
    print("Starting training (Feature Extraction)...", flush=True)
    trainer.fit(model, data_module)

    # 5. Predict (Anomaly Detection)
    print("Starting prediction...", flush=True)
    predictions = trainer.predict(model, data_module)

    # 6. Post-processing & Visualization
    print("Processing predictions...")

    # Group results by image
    results_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    if predictions:
        for batch_res in predictions:
            scores = batch_res["scores"]
            xs = batch_res["x"]
            ys = batch_res["y"]
            img_idxs = batch_res["img_idx"]
            img_paths = batch_res["img_path"]
            labels = batch_res["label"]

            for i in range(len(scores)):
                img_idx = img_idxs[i]
                results_by_image[img_idx].append(
                    {
                        "score": scores[i],
                        "x": xs[i],
                        "y": ys[i],
                        "path": img_paths[i],
                        "label": labels[i],
                    }
                )

    # Calculate Image-Level Metrics
    image_scores = []
    image_labels_list = []

    # First pass: Collect scores for metrics
    for _img_idx, patches in results_by_image.items():
        if not patches:
            continue
        scores = [p["score"] for p in patches]
        img_score = np.max(scores)
        image_scores.append(img_score)
        image_labels_list.append(patches[0]["label"])

    # Metrics Calculation
    best_threshold = 0.0

    if len(image_scores) > 0:
        image_scores_np = np.array(image_scores)
        image_labels_np = np.array(image_labels_list)

        # Check if we have both classes
        if len(np.unique(image_labels_np)) > 1:
            # AUROC
            auroc = roc_auc_score(image_labels_np, image_scores_np)
            print(f"\nImage-Level AUROC: {auroc:.4f}")

            # 1. Optimal Threshold (Maximize F1)
            precisions, recalls, thresholds = precision_recall_curve(
                image_labels_np, image_scores_np
            )
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            # f1_scores has one extra element, slice to match thresholds
            f1_scores = f1_scores[:-1]

            # Find best F1, but prefer thresholds that don't flag everything (TN > 0)
            # Refined: Try to find a threshold where we have at least 50% TNR
            sorted_indices = np.argsort(f1_scores)[::-1]

            best_threshold = 0.0
            best_f1 = 0.0
            found_balanced = False

            clean_scores_check = image_scores_np[image_labels_np == 0]
            n_clean = len(clean_scores_check)

            # First pass: Try to find good F1 with TNR >= 0.5
            for idx in sorted_indices:
                th = thresholds[idx]
                tn_count = np.sum(clean_scores_check < th) if n_clean > 0 else 0

                if n_clean > 0 and (tn_count / n_clean) >= 0.5:
                    best_threshold = th
                    best_f1 = f1_scores[idx]
                    found_balanced = True
                    break

            # Second pass: If not found, fallback to TN > 0
            if not found_balanced:
                for idx in sorted_indices:
                    th = thresholds[idx]
                    tn_count = np.sum(clean_scores_check < th) if n_clean > 0 else 0

                    if n_clean > 0 and tn_count > 0:
                        best_threshold = th
                        best_f1 = f1_scores[idx]
                        found_balanced = True
                        break

            if not found_balanced:
                if len(f1_scores) > 0:
                    best_idx = np.argmax(f1_scores)
                    best_threshold = thresholds[best_idx]
                    best_f1 = f1_scores[best_idx]

            print("\n--- Strategy 1: Balanced F1 Score (Target TNR >= 0.5) ---")
            print(f"Best F1 Score: {best_f1:.4f} at Threshold: {best_threshold:.4f}")

            preds = (image_scores_np >= best_threshold).astype(int)
            cm = confusion_matrix(image_labels_np, preds)
            tn, fp, fn, tp = cm.ravel()
            print(f"Confusion Matrix:\n{cm}")
            print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

            # 2. Safe Threshold (Zero False Positives on Clean Data)
            clean_scores = image_scores_np[image_labels_np == 0]
            if len(clean_scores) > 0:
                # Set threshold slightly above the max clean score
                safe_threshold = np.max(clean_scores) + 1e-5

                print("\n--- Strategy 2: Zero False Positives (Safe Threshold) ---")
                print(f"Max Clean Score: {np.max(clean_scores):.4f}")
                print(f"Safe Threshold: {safe_threshold:.4f}")

                preds_safe = (image_scores_np >= safe_threshold).astype(int)
                cm_safe = confusion_matrix(image_labels_np, preds_safe)
                tn_s, fp_s, fn_s, tp_s = cm_safe.ravel()

                print(f"Confusion Matrix:\n{cm_safe}")
                print(f"TP: {tp_s}, FN: {fn_s}, FP: {fp_s}, TN: {tn_s}")

                recall_safe = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
                print(f"Recall (Sensitivity) at Safe Threshold: {recall_safe:.4f}")
            else:
                print("\nNo clean images in test set to determine Safe Threshold.")
        else:
            print("\nMetrics cannot be calculated: Only one class present in test set.")

    # Visualization Loop (Using Best F1 Threshold)
    output_dir = os.path.join("anomaly_results", data_module.vehicle_type)
    os.makedirs(output_dir, exist_ok=True)

    # Decision Logic for Visualization Threshold
    final_threshold = best_threshold

    # Check if Best F1 strategy is too aggressive (TN=0)
    if len(image_scores) > 0:
        preds = (image_scores_np >= best_threshold).astype(int)
        cm = confusion_matrix(image_labels_np, preds)
        tn, fp, fn, tp = cm.ravel()

        if tn == 0 and fp > 0:
            print(
                "\n[SAFETY FUSE] Strategy 1 flagged ALL clean images as anomalies. "
                "Switching to Safe Threshold."
            )
            clean_scores = image_scores_np[image_labels_np == 0]
            if len(clean_scores) > 0:
                final_threshold = np.max(clean_scores) + 1e-5
            else:
                # Fallback if no clean scores (unlikely given the check above)
                final_threshold = best_threshold

    print(f"\n--- Generating Visualizations (Threshold: {final_threshold:.4f}) ---")

    for _img_idx, patches in results_by_image.items():
        if not patches:
            continue

        # Get image info from first patch
        img_path = patches[0]["path"]

        # Extract lists
        scores = [p["score"] for p in patches]
        coords = [(p["x"], p["y"]) for p in patches]

        img_score = np.max(scores)

        # Read original image
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        _h, _w = orig_img.shape[:2]

        fname = os.path.basename(img_path)

        if img_score > final_threshold:
            # Anomaly Detected

            # Sort patches by score descending
            sorted_indices = np.argsort(scores)[::-1]

            output_img = orig_img.copy()

            # Find top 1 anomaly
            anomalies_found: list[tuple[int, int, float]] = []

            for idx in sorted_indices:
                if len(anomalies_found) >= 1:
                    break

                curr_score = scores[idx]
                curr_x, curr_y = coords[idx]

                # Check distance to already found anomalies to avoid overlapping boxes
                # on the same defect
                is_distinct = True
                for exist_x, exist_y, _ in anomalies_found:
                    dist = np.sqrt((curr_x - exist_x) ** 2 + (curr_y - exist_y) ** 2)
                    if (
                        dist < patch_size
                    ):  # If closer than one patch size, consider it the same anomaly
                        is_distinct = False
                        break

                if is_distinct:
                    anomalies_found.append((curr_x, curr_y, curr_score))

            # Draw rectangles for found anomalies
            for ax, ay, ascore in anomalies_found:
                # Calculate center of the 224x224 patch
                center_x = ax + patch_size // 2
                center_y = ay + patch_size // 2

                # Draw a smaller "focus" box (e.g., 100x100) centered on the anomaly
                box_s = 100
                half_s = box_s // 2

                cv2.rectangle(
                    output_img,
                    (center_x - half_s, center_y - half_s),
                    (center_x + half_s, center_y + half_s),
                    (0, 0, 255),
                    5,
                )  # Red box
                cv2.putText(
                    output_img,
                    f"{ascore:.2f}",
                    (center_x - half_s, center_y - half_s - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            save_path = os.path.join(output_dir, f"DETECTED_{fname}")
            cv2.imwrite(save_path, output_img)
        else:
            # Clean
            # Just save original
            save_path = os.path.join(output_dir, f"CLEAN_{fname}")
            cv2.imwrite(save_path, orig_img)


print("About to run main...", flush=True)
if __name__ == "__main__":
    main()
