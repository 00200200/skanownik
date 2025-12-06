import shutil
from pathlib import Path

import cv2


def manual_classify():
    base_path = Path(__file__).parent.parent / "data" / "brudne"

    convertive_dir = base_path / ".convertive"
    truck_dir = base_path / ".truck"

    convertive_dir.mkdir(exist_ok=True)
    truck_dir.mkdir(exist_ok=True)

    # Get all directories that are not the target directories
    folders = []
    for item in base_path.iterdir():
        if item.is_dir():
            if item.name.startswith("."):  # Skip .convertive, .truck
                continue
            folders.append(item)

    folders.sort(key=lambda x: x.name)

    print(f"Found {len(folders)} folders to classify.")
    print("Controls:")
    print("  'c' -> Move to .convertive")
    print("  't' -> Move to .truck")
    print("  'q' -> Quit")
    print("  Any other key -> Skip")

    for folder in folders:
        # Find the 'czarno.bmp' file or any bmp
        bmp_files = list(folder.glob("*czarno.bmp"))
        if not bmp_files:
            bmp_files = list(folder.glob("*.bmp"))

        if not bmp_files:
            print(f"No images found in {folder.name}, skipping.")
            continue

        # Prefer the one with 'czarno' if available, else take the first one
        image_path = bmp_files[0]

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image {image_path}, skipping.")
            continue

        # Resize for better viewing if too large
        height, width = img.shape[:2]
        max_height = 800
        if height > max_height:
            scale = max_height / height
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        cv2.imshow("Classifier", img)
        print(f"Classifying {folder.name}...")

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print("Quitting.")
            break
        elif key == ord("c"):
            target = convertive_dir / folder.name
            print("Moving to .convertive...")
            shutil.move(str(folder), str(target))
        elif key == ord("t"):
            target = truck_dir / folder.name
            print("Moving to .truck...")
            shutil.move(str(folder), str(target))
        else:
            print("Skipping.")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    manual_classify()
