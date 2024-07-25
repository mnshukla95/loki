import cv2
import numpy as np
import os
import sys

# Add the libcom directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'libcom')))

# Import the ImageHarmonization class from image_harmonization.py
from libcom.image_harmonization.image_harmonization import ImageHarmonizationModel


def resize_to_height(image, target_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)

def insert_object_in_video(video_path, masks_folder, new_object_path, output_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    new_object = cv2.imread(new_object_path, cv2.IMREAD_UNCHANGED)
    if new_object is None:
        raise ValueError(f"Could not read new object image from {new_object_path}")
    
    first_mask_path = os.path.join(masks_folder, f"frame_mask_00001.png")
    first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
    y, x = np.where(first_mask > 0)
    top, bottom, left, right = y.min(), y.max(), x.min(), x.max()
    mask_height = bottom - top
    
    new_object_resized = resize_to_height(new_object, mask_height)
    
    pad_left = max(0, (right - left - new_object_resized.shape[1]) // 2)
    
    if new_object_resized.shape[2] == 4:
        object_alpha = new_object_resized[:,:,3] / 255.0
        new_object_resized = new_object_resized[:,:,:3]
    else:
        object_alpha = np.ones(new_object_resized.shape[:2])
    
    frame_count = 1
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Simple alpha blending without color adjustments
        for c in range(3):
            frame[top:bottom, left+pad_left:left+pad_left+new_object_resized.shape[1], c] = \
                frame[top:bottom, left+pad_left:left+pad_left+new_object_resized.shape[1], c] * (1 - object_alpha) + \
                new_object_resized[:,:,c] * object_alpha
        
        out.write(frame.astype(np.uint8))
        frame_count += 1
    
    video.release()
    out.release()
    cv2.destroyAllWindows()


# Usage
home_dir = os.path.expanduser("~")
current_dir = os.path.join(home_dir, "Downloads", "smoshcomp")  # Update if needed
video_path = os.path.join(current_dir, "smosh720.mp4")
masks_folder = os.path.join(current_dir, "masks")
new_object_path = os.path.join(current_dir, "redbull.png")
output_path = os.path.join(current_dir, "output_video.mp4")

print(f"Video file path: {video_path}")
print(f"Masks folder path: {masks_folder}")
print(f"New object path: {new_object_path}")
print(f"Output video path: {output_path}")

if not os.path.exists(masks_folder):
    print(f"Error: Masks folder not found at {masks_folder}")
else:
    print("\nContents of masks folder:")
    mask_files = os.listdir(masks_folder)
    for file in mask_files[:5]:  # Print first 5 files
        print(file)
    if len(mask_files) > 5:
        print(f"... and {len(mask_files) - 5} more files")

    if not mask_files:
        print("No files found in the masks folder.")
    elif not any(file.startswith("frame_mask_") and file.endswith(".png") for file in mask_files):
        print("No files matching the expected mask format (frame_mask_XXXXX.png) found.")

try:
    # Initialize the harmonization model
    harmonization_model = ImageHarmonizationModel()  # Adjust if initialization requires parameters

    # Harmonize the image
    harmonized_image = harmonize_image(harmonization_model, video_path, new_object_path)

    # Save the harmonized image
    harmonized_image_path = os.path.join(current_dir, "harmonized_redbull.png")
    cv2.imwrite(harmonized_image_path, harmonized_image)

    # Insert harmonized object into video
    insert_object_in_video(video_path, masks_folder, harmonized_image_path, output_path)
    print(f"Video processing complete. Output saved to {output_path}")
except Exception as e:
    print(f"An error occurred during video processing: {str(e)}")