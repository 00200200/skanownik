import os
import cv2
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from src.utils.image_utils import tile_image

def generate_frankenstein_car(vehicle_type="osobowe", output_file="frankenstein_car.png"):
    print(f"Generating Frankenstein car for {vehicle_type}...")
    
    data_dir = Path(f"data/czyste/{vehicle_type}")
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    all_files = list(data_dir.glob("**/*.bmp"))
    all_files = [f for f in all_files if "czarno" in f.name]
    
    if not all_files:
        print("No images found.")
        return

def get_content_bbox(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Content is what is NOT background (white or black)
    # Background: > 230 (white) or < 40 (black)
    # Content: between 40 and 230
    mask = (gray > 40) & (gray < 230)
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return 0, 0, img.shape[1], img.shape[0]
    return cv2.boundingRect(coords)

def generate_frankenstein_car(vehicle_type="osobowe", output_file="frankenstein_car.png"):
    print(f"Generating Frankenstein car for {vehicle_type}...")
    
    data_dir = Path(f"data/czyste/{vehicle_type}")
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    all_files = list(data_dir.glob("**/*.bmp"))
    all_files = [f for f in all_files if "czarno" in f.name]
    
    if not all_files:
        print("No images found.")
        return

    # 1. Select and Crop Template
    # Find a template that has actual content
    random.shuffle(all_files)
    template_img = None
    template_bbox = None
    
    for p in all_files:
        img = cv2.imread(str(p))
        if img is None: continue
        
        x, y, w, h = get_content_bbox(img)
        # Ensure it's not empty or tiny
        if w > 100 and h > 100:
            template_img = img[y:y+h, x:x+w]
            template_bbox = (x, y, w, h)
            print(f"Selected template from {p.name}, cropped to {w}x{h}")
            break
            
    if template_img is None:
        print("Could not find a valid template.")
        return
        
    h_template, w_template = template_img.shape[:2]

    # 2. Prepare Bank (Crop and Resize)
    resized_bank = []
    subset_files = all_files[:50] # Use 50 random images
    
    for p in subset_files:
        img = cv2.imread(str(p))
        if img is None: continue
        
        # Crop to content first!
        x, y, w, h = get_content_bbox(img)
        if w > 50 and h > 50:
            img_cropped = img[y:y+h, x:x+w]
            # Resize to match template exactly
            img_resized = cv2.resize(img_cropped, (w_template, h_template))
            resized_bank.append(img_resized)
            
    if not resized_bank:
        print("No bank images loaded.")
        return

    # 3. Build Mosaic
    # Initialize with white background (255) to match X-ray style
    mosaic = np.full((h_template, w_template, 3), 255, dtype=np.uint8)
    
    patch_size = 112 # Smaller patch size for better resolution/detail
    stride = patch_size
    
    # Iterate over the grid
    for y in range(0, h_template, stride):
        for x in range(0, w_template, stride):
            # Pick a random image from the bank
            source_img = random.choice(resized_bank)
            
            # Extract patch
            # Handle boundary conditions
            h_p = min(patch_size, h_template - y)
            w_p = min(patch_size, w_template - x)
            
            if h_p <= 0 or w_p <= 0:
                continue
                
            patch = source_img[y:y+h_p, x:x+w_p]
            mosaic[y:y+h_p, x:x+w_p] = patch
             
    cv2.imwrite(output_file, mosaic)
    print(f"Saved mosaic to {output_file}")

if __name__ == "__main__":
    generate_frankenstein_car("osobowe", "mosaic_osobowe.png")
    generate_frankenstein_car("dostawcze", "mosaic_dostawcze.png")
