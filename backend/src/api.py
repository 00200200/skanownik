import os
import cv2
import numpy as np
import torch
import base64
import shutil
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torch.utils.data import DataLoader

from src.models.vehicle_classifier import VehicleClassifier
from src.models.anomaly_detector import AnomalyDetectorPL
from src.dataloader.anomaly_datamodule import XRayDataset
from src.inference_pipeline import load_vehicle_classifier, classify_vehicle, load_anomaly_config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VEHICLE_CLASSIFIER_CKPT = "trained_models/vehicle-classifier-epoch=19-val_loss=0.19.ckpt"
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Global models
classifier_model = None

@app.on_event("startup")
async def startup_event():
    global classifier_model
    if os.path.exists(VEHICLE_CLASSIFIER_CKPT):
        classifier_model = load_vehicle_classifier(VEHICLE_CLASSIFIER_CKPT)
    else:
        print(f"Warning: Classifier checkpoint not found at {VEHICLE_CLASSIFIER_CKPT}")

def detect_anomalies_internal(vehicle_type, image_path):
    config = load_anomaly_config(vehicle_type)
    
    print(f"Running anomaly detection for {vehicle_type}...")
    
    # 1. Load Model
    model = AnomalyDetectorPL(
        backbone_name="resnet18",
        use_coords=config["use_coords"],
        knn_path=config["knn_path"]
    )
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 2. Prepare Data
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
    
    device = next(model.parameters()).device
    
    for batch in dataloader:
        batch['patch'] = batch['patch'].to(device)
        batch['x'] = batch['x'].to(device)
        batch['y'] = batch['y'].to(device)
        
        with torch.no_grad():
            results = model.predict_step(batch, 0)
        
        batch_scores = results['scores']
        batch_xs = results['x']
        batch_ys = results['y']
        
        for i in range(len(batch_scores)):
            all_scores.append(batch_scores[i])
            all_coords.append((batch_xs[i], batch_ys[i]))
            
    if not all_scores:
        return 0.0, False, None

    max_score = np.max(all_scores)
    threshold = config["threshold"]
    
    # 4. Visualize
    orig_img = cv2.imread(str(image_path))
    output_img = orig_img.copy()
    
    is_anomaly = False
    label_text = "CLEAN"
    color = (0, 255, 0) # Green
    
    if max_score > threshold:
        is_anomaly = True
        label_text = "DETECTED"
        color = (0, 0, 255) # Red
        
        sorted_indices = np.argsort(all_scores)[::-1]
        anomalies_found = []
        
        for idx in sorted_indices:
            if len(anomalies_found) >= 1:
                break
                
            curr_score = all_scores[idx]
            if curr_score < threshold:
                continue
                
            curr_x, curr_y = all_coords[idx]
            
            is_distinct = True
            for (ex, ey, _) in anomalies_found:
                dist = np.sqrt((curr_x - ex)**2 + (curr_y - ey)**2)
                if dist < config["patch_size"]:
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
            
    cv2.putText(output_img, f"{label_text} ({vehicle_type})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    return float(max_score), is_anomaly, output_img

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Sanitize filename to prevent directory traversal and ensure it's just a filename
    filename = os.path.basename(file.filename)
    temp_path = TEMP_DIR / filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Classify
        if classifier_model is None:
             return {"error": "Model not loaded"}
             
        vehicle_type = classify_vehicle(classifier_model, temp_path)
        
        # 2. Detect
        max_score, is_anomaly, result_img = detect_anomalies_internal(vehicle_type, temp_path)
        
        # 3. Encode image
        if result_img is not None:
            _, buffer = cv2.imencode('.jpg', result_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            img_base64 = None
            
        return {
            "filename": file.filename,
            "vehicle_type": vehicle_type,
            "anomaly_score": max_score,
            "is_anomaly": is_anomaly,
            "image_base64": img_base64
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if temp_path.exists():
            os.remove(temp_path)
