import os
import cv2
import numpy as np
import random
from pathlib import Path
from src.utils.image_utils import tile_image

def visualize_prototype(vehicle_type="osobowe", num_patches=100, output_file="prototype_visualization.png"):
    print(f"Generating prototype visualization for {vehicle_type}...")
    
    data_dir = Path(f"data/czyste/{vehicle_type}")
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    # Find all BMP files
    all_files = list(data_dir.glob("**/*.bmp"))
    # Filter for 'czarno' if needed, similar to datamodule
    all_files = [f for f in all_files if "czarno" in f.name]
    
    if not all_files:
        print("No images found.")
        return

    print(f"Found {len(all_files)} clean images.")
    
    # Collect patches
    collected_patches = []
    
    # Shuffle files to get variety
    random.shuffle(all_files)
    
    for img_path in all_files:
        if len(collected_patches) >= num_patches:
            break
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Tile image
        patches_with_coords = tile_image(
            img,
            patch_size=224,
            stride=224, # No overlap for visualization to get more variety
            filter_background=True
        )
        
        for patch, _, _ in patches_with_coords:
            collected_patches.append(patch)
            if len(collected_patches) >= num_patches:
                break
    
    if not collected_patches:
        print("No patches extracted.")
        return

    print(f"Collected {len(collected_patches)} patches.")

    # Create a grid
    # Calculate grid size (approx square)
    grid_side = int(np.ceil(np.sqrt(len(collected_patches))))
    patch_h, patch_w = collected_patches[0].shape[:2]
    
    grid_h = grid_side * patch_h
    grid_w = grid_side * patch_w
    
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, patch in enumerate(collected_patches):
        r = idx // grid_side
        c = idx % grid_side
        
        y = r * patch_h
        x = c * patch_w
        
        grid_img[y:y+patch_h, x:x+patch_w] = patch
        
    cv2.imwrite(output_file, grid_img)
    print(f"Prototype visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_prototype(vehicle_type="osobowe", num_patches=144, output_file="prototype_osobowe.png")
    visualize_prototype(vehicle_type="dostawcze", num_patches=144, output_file="prototype_dostawcze.png")
