

import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "/home/dmore/code/TFM_David/CNN_Galicia_Toshiba/severstal_backup_ep35.pth"
IMAGE_DIR = "/home/dmore/code/TFM_David/CNN_Galicia_Toshiba/final_dataset/test/images"
OUTPUT_DIR = "/home/dmore/code/TFM_David/CNN_Galicia_Toshiba/inference_results"

# Input size matching training crop (Height, Width)
WINDOW_HEIGHT = 256
WINDOW_WIDTH = 800
STRIDE_HEIGHT = 200  # Overlap of 56
STRIDE_WIDTH = 600   # Overlap of 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimal Thresholds from Analysis
THRESHOLDS = {
    0: 0.45,  # Class 1
    1: 0.70,  # Class 2
    2: 0.70,  # Class 3
    3: 0.65   # Class 4
}

# Colors for visualization (B, G, R)
COLORS = {
    0: (255, 0, 0),    # Class 1: Blue
    1: (0, 255, 0),    # Class 2: Green
    2: (0, 165, 255),  # Class 3: Orange
    3: (0, 0, 255)     # Class 4: Red/Yellowish
}

def get_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = smp.Unet(
        encoder_name="tu-convnext_tiny",
        encoder_weights=None,
        in_channels=3,
        classes=5  # Background + 4 defects
    )
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle both full checkpoint dict and direct state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def get_preprocessing():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def sliding_window_inference(image, model, transform):
    h, w = image.shape[:2]
    
    # Pad image to be divisible by window size (optional, but safer to just pad end)
    pad_h = (WINDOW_HEIGHT - (h % WINDOW_HEIGHT)) % WINDOW_HEIGHT
    pad_w = (WINDOW_WIDTH - (w % WINDOW_WIDTH)) % WINDOW_WIDTH
    
    # Simpler approach: work on a canvas large enough
    canvas_h = h + WINDOW_HEIGHT
    canvas_w = w + WINDOW_WIDTH
    
    padded_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    padded_image[:h, :w] = image
    
    # Accumulators
    probs_map = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32) # 4 defect classes
    count_map = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    
    # Sliding window
    for y in range(0, h, STRIDE_HEIGHT):
        for x in range(0, w, STRIDE_WIDTH):
            y_end = y + WINDOW_HEIGHT
            x_end = x + WINDOW_WIDTH
            
            crop = padded_image[y:y_end, x:x_end]
            
            # Preprocess
            augmented = transform(image=crop)["image"]
            input_tensor = augmented.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(input_tensor)
                # Output shape: [1, 5, 256, 800] -> We want [1, 4, 256, 800] (ignoring background channel 0 if 5 classes)
                # Assuming output is logits. Apply sigmoid.
                probs = torch.sigmoid(output)
                
                # Setup: Train was classes=5 (0=bg, 1,2,3,4=defects). 
                # We care about channels 1,2,3,4
                probs_defects = probs[0, 1:, :, :].cpu().numpy() # Shape (4, 256, 800)
                
                # Transpose to (256, 800, 4)
                probs_defects = np.transpose(probs_defects, (1, 2, 0))
                
            probs_map[y:y_end, x:x_end] += probs_defects
            count_map[y:y_end, x:x_end] += 1
            
    # Normalize by count
    count_map[count_map == 0] = 1 # Avoid div by zero
    final_probs = probs_map / count_map
    
    # Crop back to original size
    return final_probs[:h, :w, :]

def visualize_and_save(image, probs, filename):
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Create mask for each class
    mask_combined = np.zeros((h, w), dtype=np.uint8)
    
    found_defect = False
    
    for cls_idx in range(4): # 0..3 mapping to classes 1..4
        threshold = THRESHOLDS[cls_idx]
        mask = (probs[:, :, cls_idx] > threshold).astype(np.uint8)
        
        if mask.max() > 0:
            found_defect = True
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw on overlay
            color = COLORS[cls_idx]
            cv2.drawContours(overlay, contours, -1, color, 3)
            
            # Add label
            # Just label the first contour for cleanliness
            if contours:
                c = max(contours, key=cv2.contourArea)
                x,y,w_rect,h_rect = cv2.boundingRect(c)
                cv2.putText(overlay, f"Class {cls_idx+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Blend
    alpha = 0.6
    vis = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, vis)
    
    status = "DEFECTS FOUND" if found_defect else "Clean"
    print(f"Saved: {filename} [{status}]")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model = get_model()
    transform = get_preprocessing()
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Convert BGR (cv2) to RGB (albumentations)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Inference
        probs = sliding_window_inference(image_rgb, model, transform)
        
        # Visualize (use original BGR image for opencv drawing)
        visualize_and_save(image, probs, img_name)

if __name__ == "__main__":
    main()
