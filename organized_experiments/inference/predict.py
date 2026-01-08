import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = '/home/dmore/code/TFM_David/CNN_Galicia_Toshiba/Steel-Defect-Detection-AI-Wind-Turbines-Production-Quality-Inspection/organized_experiments/Unet++Resnet34/severstal_best.pth' # Ajustar
TEST_DIR = '/home/dmore/code/TFM_David/CNN_Galicia_Toshiba/datasets/final_dataset/test/images'
OUTPUT_DIR = 'predictions'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path):
    print(f"Cargando modelo desde {path}")
    model = smp.UnetPlusPlus(encoder_name='resnet34', classes=5, in_channels=3)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_transforms():
    return A.Compose([
        A.Resize(256, 1600), # Ajustar al tamaño de entrenamiento o usar sliding window
        A.Normalize(),
        ToTensorV2()
    ])

def predict_folder(model, folder_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    
    transform = get_transforms()
    
    print(f"Procesando {len(images)} imágenes...")
    
    for img_name in tqdm(images):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        aug = transform(image=image)
        input_tensor = aug['image'].unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
            
        # Visualize
        if pred_mask.max() > 0: # Solo guardar si hay defecto
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask, vmin=0, vmax=4)
            plt.title("Predicción")
            plt.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
            plt.savefig(save_path)
            plt.close()

if __name__ == "__main__":
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        predict_folder(model, TEST_DIR)
    else:
        print(f"❌ No se encontró el modelo en {MODEL_PATH}. Ajusta la ruta en el script.")
