print("Imports starting...", flush=True)
import torch
from pytorch_lightning import Trainer
from src.dataloader.anomaly_datamodule import XRayDataModule
from src.models.anomaly_detector import AnomalyDetectorPL
from src.utils.image_utils import stitch_heatmap
import cv2
import numpy as np
import os
from collections import defaultdict

from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix

def main():
    print("Script started.", flush=True)
    
    # Configuration for higher precision
    PATCH_SIZE = 224 # Reverted to 224 for better context
    STRIDE = 64      # Increased stride for faster processing (still good overlap)
    
    # 1. Setup Data
    data_module = XRayDataModule(
        vehicle_type='dostawcze', 
        batch_size=32,
        patch_size=PATCH_SIZE,
        stride=STRIDE
    )
    
    # 2. Setup Model
    model = AnomalyDetectorPL(backbone_name='resnet18')
    
    # 3. Trainer
    trainer = Trainer(
        max_epochs=1, # One pass to extract features
        accelerator='auto',
        devices=1,
        enable_checkpointing=False, # We save KNN manually
        logger=True, # Enable logger to see progress bar
        enable_progress_bar=True
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
    results_by_image = defaultdict(list)
    
    for batch_res in predictions:
        scores = batch_res['scores']
        xs = batch_res['x']
        ys = batch_res['y']
        img_idxs = batch_res['img_idx']
        img_paths = batch_res['img_path']
        labels = batch_res['label']
        
        for i in range(len(scores)):
            img_idx = img_idxs[i]
            results_by_image[img_idx].append({
                'score': scores[i],
                'x': xs[i],
                'y': ys[i],
                'path': img_paths[i],
                'label': labels[i]
            })
            
    # Calculate Image-Level Metrics
    image_scores = []
    image_labels = []
    
    # First pass: Collect scores for metrics
    for img_idx, patches in results_by_image.items():
        if not patches:
            continue
        scores = [p['score'] for p in patches]
        img_score = np.max(scores)
        image_scores.append(img_score)
        image_labels.append(patches[0]['label'])

    # Metrics Calculation
    best_threshold = 0.0
    
    if len(image_scores) > 0:
        image_scores = np.array(image_scores)
        image_labels = np.array(image_labels)
        
        # Check if we have both classes
        if len(np.unique(image_labels)) > 1:
            # AUROC
            auroc = roc_auc_score(image_labels, image_scores)
            print(f"\nImage-Level AUROC: {auroc:.4f}")
            
            # 1. Optimal Threshold (Maximize F1)
            precisions, recalls, thresholds = precision_recall_curve(image_labels, image_scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            print(f"\n--- Strategy 1: Maximize F1 Score ---")
            print(f"Best F1 Score: {best_f1:.4f} at Threshold: {best_threshold:.4f}")
            
            preds = (image_scores >= best_threshold).astype(int)
            cm = confusion_matrix(image_labels, preds)
            tn, fp, fn, tp = cm.ravel()
            print(f"Confusion Matrix:\n{cm}")
            print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
            
            # 2. Safe Threshold (Zero False Positives on Clean Data)
            clean_scores = image_scores[image_labels == 0]
            if len(clean_scores) > 0:
                # Set threshold slightly above the max clean score
                safe_threshold = np.max(clean_scores) + 1e-5
                
                print(f"\n--- Strategy 2: Zero False Positives (Safe Threshold) ---")
                print(f"Max Clean Score: {np.max(clean_scores):.4f}")
                print(f"Safe Threshold: {safe_threshold:.4f}")
                
                preds_safe = (image_scores >= safe_threshold).astype(int)
                cm_safe = confusion_matrix(image_labels, preds_safe)
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
    output_dir = "anomaly_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Generating Visualizations (Threshold: {best_threshold:.4f}) ---")
    
    for img_idx, patches in results_by_image.items():
        if not patches:
            continue
            
        # Get image info from first patch
        img_path = patches[0]['path']
        
        # Extract lists
        scores = [p['score'] for p in patches]
        coords = [(p['x'], p['y']) for p in patches]
        
        img_score = np.max(scores)
        
        # Read original image
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]
        
        fname = os.path.basename(img_path)
        
        if img_score > best_threshold:
            # Anomaly Detected
            
            # Find the single most anomalous patch
            max_score_idx = np.argmax(scores)
            max_x, max_y = coords[max_score_idx]
            
            # Draw rectangle around the most anomalous patch on the original image
            output_img = orig_img.copy()
            
            # Calculate center of the 224x224 patch
            center_x = max_x + PATCH_SIZE // 2
            center_y = max_y + PATCH_SIZE // 2
            
            # Draw a smaller "focus" box (e.g., 100x100) centered on the anomaly
            box_s = 100
            half_s = box_s // 2
            
            cv2.rectangle(output_img, (center_x - half_s, center_y - half_s), (center_x + half_s, center_y + half_s), (0, 0, 255), 5) # Red box
            cv2.putText(output_img, f"ANOMALY: {scores[max_score_idx]:.2f}", (center_x - half_s, center_y - half_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            save_path = os.path.join(output_dir, f"DETECTED_{fname}")
            cv2.imwrite(save_path, output_img)
        else:
            # Clean
            # Just save original
            save_path = os.path.join(output_dir, f"CLEAN_{fname}")
            cv2.imwrite(save_path, orig_img)

print("About to run main...", flush=True)
if __name__ == '__main__':
    main()
