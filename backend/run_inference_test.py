import os
import random
import subprocess
from pathlib import Path

def main():
    # Define paths
    base_dir = Path("data")
    clean_dir = base_dir / "czyste"
    dirty_dir = base_dir / "brudne"
    
    # Collect all image files
    clean_images = []
    dirty_images = []
    
    print("Collecting images...")
    
    # Walk through clean directory
    for root, _, files in os.walk(clean_dir):
        for file in files:
            if "czarno" in file and file.endswith(".bmp"):
                clean_images.append(os.path.join(root, file))
                
    # Walk through dirty directory
    for root, _, files in os.walk(dirty_dir):
        for file in files:
            if "czarno" in file and file.endswith(".bmp"):
                dirty_images.append(os.path.join(root, file))
                
    print(f"Found {len(clean_images)} clean images and {len(dirty_images)} dirty images.")
    
    # Select random samples
    random.seed(42) # For reproducibility
    
    # selected_clean = random.sample(clean_images, min(10, len(clean_images)))
    selected_dirty = random.sample(dirty_images, min(10, len(dirty_images)))
    
    all_selected = selected_dirty # selected_clean + selected_dirty
    
    print(f"Selected {len(all_selected)} images for inference test.")
    
    # Run inference pipeline for each image
    output_dir = "inference_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in all_selected:
        print(f"\nProcessing: {img_path}")
        try:
            # Call the inference pipeline script
            # We use 'uv run' as per previous context
            cmd = ["uv", "run", "src/inference_pipeline.py", img_path, "--output_dir", output_dir]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {img_path}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print(f"\nTest completed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
