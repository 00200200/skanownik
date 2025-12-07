import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from src.models.vehicle_classifier import VehicleClassifier
from src.models.anomaly_detector import AnomalyDetectorPL
from src.dataloader.anomaly_datamodule import XRayDataset

# Configuration
VEHICLE_CLASSIFIER_CKPT = "trained_models/vehicle-classifier-epoch=19-val_loss=0.19.ckpt"

def load_anomaly_config(vehicle_type):
    config_path = f"trained_models_anomaly/config_{vehicle_type}.json"
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using defaults.")
        # Fallback defaults
        return {
            "knn_path": f"trained_models_anomaly/knn_{vehicle_type}.pkl",
            "patch_size": 224,
            "stride": 96 if vehicle_type == "dostawcze" else 64,
            "use_coords": False,
            "threshold": 10.0
        }
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Add KNN path which is not in the JSON usually
    config["knn_path"] = f"trained_models_anomaly/knn_{vehicle_type}.pkl"
    
    # Prefer safe_threshold if available, else threshold
    if "threshold" in config:
        print(f"Using Threshold: {config['threshold']}")
        config["threshold"] = config["threshold"]
    else:
        print(f"Using Safe Threshold: {config['safe_threshold']}")
        
    return config

def load_vehicle_classifier(ckpt_path):
    print(f"Loading vehicle classifier from {ckpt_path}...")
    # Load to CPU first to avoid issues, then move if needed
    model = VehicleClassifier.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.freeze()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Vehicle Classifier moved to GPU.")
    
    return model

def classify_vehicle(model, image_path):
    # Preprocessing matching VehicleClassifier training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_tensor = transform(img).unsqueeze(0) # Add batch dim
    
    # Move input to same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    # 0: osobowe, 1: dostawcze
    label = "dostawcze" if pred_idx == 1 else "osobowe"
    confidence = probs[0][pred_idx].item()
    
    print(f"Vehicle classified as: {label} (Confidence: {confidence:.4f})")
    return label

def detect_anomalies(vehicle_type, image_path, output_dir):
    config = load_anomaly_config(vehicle_type)
        
    print(f"Running anomaly detection for {vehicle_type}...")
    
    # 1. Load Model
    model = AnomalyDetectorPL(
        backbone_name="resnet18",
        use_coords=config["use_coords"],
        knn_path=config["knn_path"]
    )
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Anomaly Detector moved to GPU.")
    
    # 2. Prepare Data (Patches)
    # We use XRayDataset in 'predict' mode to get patches and metadata
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config["patch_size"], config["patch_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = XRayDataset(
        image_paths=[Path(image_path)],
        transform=transform,
        patch_size=config["patch_size"],
        stride=config["stride"],
        mode="predict"
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 3. Predict
    all_scores = []
    all_coords = []
    
    print("Extracting features and querying KNN...")
    
    device = next(model.parameters()).device
    
    for batch in dataloader:
        # Move batch tensors to device
        batch['patch'] = batch['patch'].to(device)
        batch['x'] = batch['x'].to(device)
        batch['y'] = batch['y'].to(device)
        batch['orig_h'] = batch['orig_h'].to(device)
        batch['orig_w'] = batch['orig_w'].to(device)
        # img_idx, img_path, label might not need to be on GPU or are not tensors
        # But AnomalyDetectorPL.predict_step expects them in the dict
        
        with torch.no_grad():
            results = model.predict_step(batch, 0)
        
        batch_scores = results['scores']
        batch_xs = results['x']
        batch_ys = results['y']
        
        for i in range(len(batch_scores)):
            all_scores.append(batch_scores[i])
            all_coords.append((batch_xs[i], batch_ys[i]))
            
    if not all_scores:
        print("No patches generated.")
        return

    max_score = np.max(all_scores)
    print(f"Max Anomaly Score: {max_score:.4f}")
    
    # 4. Visualize
    orig_img = cv2.imread(str(image_path))
    fname = os.path.basename(image_path)
    
    # Use a hardcoded threshold or the one from config
    # Ideally this should be dynamic or saved from training
    threshold = config["threshold"] 
    
    # Simple heuristic: if max score is very high, it's an anomaly
    # For visualization, we show top anomalies
    
    output_img = orig_img.copy()
    
    if max_score > threshold:
        print(f"Anomaly DETECTED (Score {max_score:.4f} > {threshold})")
        label_text = "DETECTED"
        color = (0, 0, 255) # Red
        
        # Draw boxes
        sorted_indices = np.argsort(all_scores)[::-1]
        anomalies_found = []
        max_anomalies = 3
        
        for idx in sorted_indices:
            if len(anomalies_found) >= max_anomalies:
                break
                
            curr_score = all_scores[idx]
            if curr_score < threshold:
                continue

            # Filter out weak anomalies compared to the strongest one found
            if len(anomalies_found) > 0:
                best_score = anomalies_found[0][2]
                # Secondary anomalies must be:
                # 1. Very close to the best score (delta < 1.0) - stricter
                # 2. Absolutely strong (score > 13.0)
                if curr_score < (best_score - 1.0) or curr_score < 13.0:
                    continue
                
            curr_x, curr_y = all_coords[idx]
            
            # Distinct check
            is_distinct = True
            for (ex, ey, _) in anomalies_found:
                dist = np.sqrt((curr_x - ex)**2 + (curr_y - ey)**2)
                if dist < config["patch_size"] / 3: # Allow significant overlap
                    is_distinct = False
                    break
            
            if is_distinct:
                anomalies_found.append((curr_x, curr_y, curr_score))
                
        for (ax, ay, ascore) in anomalies_found:
            center_x = int(ax + config["patch_size"] // 2)
            center_y = int(ay + config["patch_size"] // 2)
            box_s = 100
            half_s = box_s // 2
            
            cv2.rectangle(output_img, (center_x - half_s, center_y - half_s), (center_x + half_s, center_y + half_s), color, 5)
            cv2.putText(output_img, f"{ascore:.2f}", (center_x - half_s, center_y - half_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
    else:
        print(f"Image is CLEAN (Score {max_score:.4f} <= {threshold})")
        label_text = "CLEAN"
        color = (0, 255, 0) # Green
        
    # Add label to image
    cv2.putText(output_img, f"{label_text} ({vehicle_type})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"RESULT_{label_text}_{fname}")
    cv2.imwrite(save_path, output_img)
    print(f"Result saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Full Inference Pipeline")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: File not found: {args.image_path}")
        return

    # 1. Classify Vehicle
    classifier = load_vehicle_classifier(VEHICLE_CLASSIFIER_CKPT)
    vehicle_type = classify_vehicle(classifier, args.image_path)
    
    # 2. Detect Anomalies
    detect_anomalies(vehicle_type, args.image_path, args.output_dir)

if __name__ == "__main__":
    main()
