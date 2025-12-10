import cv2
import torch
import numpy as np
import time
import segmentation_models_pytorch as smp
import os
# import albumentations as A  <-- REMOVED to avoid headless conflict
# from albumentations.pytorch import ToTensorV2 <-- REMOVED

# ===============================
# CONFIGURACIÓN
# ===============================
# ===============================
# CONFIGURACIÓN
# ===============================
# Directorio base (donde está este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Intentamos buscar el modelo en el directorio actual o en la ruta relativa original
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, '../UnetConvNeXtTiny/models/severstal_backup_ep35.pth')
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    print(f"⚠️ NO SE ENCONTRÓ EL MODELO. Buscando en: {POSSIBLE_PATHS}")
    # Fallback para permitir que el script corra (aunque fallará al predecir si no hay pesos)
    MODEL_PATH = 'severstal_best.pth' 

ENCODER = 'tu-convnext_tiny'               # resnet34, efficientnet-b4, tu-convnext_tiny
CLASSES = ['Fondo', 'Defecto 1', 'Defecto 2', 'Defecto 3', 'Defecto 4']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.5         # Umbral para mostrar máscara

# Colores para cada clase (BGR)
COLORS = {
    1: (0, 255, 255),  # Amarillo
    2: (0, 0, 255),    # Rojo
    3: (255, 0, 0),    # Azul
    4: (0, 255, 0)     # Verde
}

# Constants
WINDOW_HEIGHT = 256
WINDOW_WIDTH = 800
STRIDE_HEIGHT = 200
STRIDE_WIDTH = 600

# Optimal Thresholds from Analysis (0-based indices for classes 1-4)
THRESHOLDS = {
    0: 0.45,  # Class 1
    1: 0.70,  # Class 2
    2: 0.70,  # Class 3
    3: 0.65   # Class 4
}

def load_model():
    print(f"Cargando modelo {ENCODER} en {DEVICE} desde {MODEL_PATH}...")
    # Ajustar según el modelo que quieras usar
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=5
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"⚠️ Error cargando pesos: {e}")
        print("Iniciando con pesos aleatorios para demo...")
    
    model.to(DEVICE)
    model.eval()
    return model

def sliding_window_inference(image, model):
    """
    Realiza inferencia usando sliding window manual (para evitar resize).
    Input image: BGR o RGB (se asume RGB dentro si se convierte antes).
    """
    h, w = image.shape[:2]
    
    # 1. Pad image to be divisible by window size or just large enough
    canvas_h = h + WINDOW_HEIGHT
    canvas_w = w + WINDOW_WIDTH
    
    padded_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    padded_image[:h, :w] = image
    
    # Accumulators
    probs_map = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32) # 4 defect classes
    count_map = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    
    # Precompute normalization stats to avoid doing it inside loop if possible
    # But for simplicity, we do it per crop like the reference
    pass 
    
    # Sliding window
    # WARNING: This double loop is SLOW for Python. 
    # But we want strict adherence to logic.
    
    crops_batch = []
    coords_batch = []
    
    for y in range(0, h, STRIDE_HEIGHT):
        for x in range(0, w, STRIDE_WIDTH):
            y_end = y + WINDOW_HEIGHT
            x_end = x + WINDOW_WIDTH
            crop = padded_image[y:y_end, x:x_end]
            crops_batch.append(crop)
            coords_batch.append((y, y_end, x, x_end))
            
    # Batch processing for slight speedup over fully sequential
    # Preprocess all crops
    # Manual Normalize (ImageNet) + Transpose
    
    # Stack -> (N, H, W, 3)
    crops_np = np.array(crops_batch, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    crops_norm = (crops_np - mean) / std
    
    # N H W C -> N C H W
    crops_tensor = torch.from_numpy(crops_norm.transpose(0, 3, 1, 2)).float().to(DEVICE)
    
    with torch.no_grad():
        output = model(crops_tensor)
        probs_batch = torch.sigmoid(output) # [N, 5, 256, 800]
        
        # Take channels 1-4
        probs_defects_batch = probs_batch[:, 1:, :, :].cpu().numpy() # [N, 4, 256, 800]
        # Transpose to [N, 256, 800, 4]
        probs_defects_batch = probs_defects_batch.transpose(0, 2, 3, 1)
        
    # Reassemble
    for idx, (y_start, y_end, x_start, x_end) in enumerate(coords_batch):
        probs_map[y_start:y_end, x_start:x_end] += probs_defects_batch[idx]
        count_map[y_start:y_end, x_start:x_end] += 1
            
    # Normalize by count
    count_map[count_map == 0] = 1 
    final_probs = probs_map / count_map
    
    # Crop back
    return final_probs[:h, :w, :]

def run_realtime_inference(source=0):
    # Verificación de entorno gráfico
    headless = False
    try:
        cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("Test")
    except cv2.error:
        print("⚠️ ERROR CRÍTICO: OpenCV no tiene soporte para GUI (imshow).")
        print("Parece que tienes instalada la versión 'headless' de OpenCV.")
        print("Para arreglarlo, instala la versión completa:")
        print("   pip install opencv-python")
        print("   (Y desinstala opencv-python-headless si existe)")
        headless = True
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: No se puede abrir la fuente de video {source}")
        print("Prueba cambiando el índice (0, 1, 2...) o usa un archivo de video.")
        return

    model = load_model()
    
    print("Iniciando captura... (Pulsa 'q' para salir)")
    
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error de lectura.")
            break
            
        # Pasar a RGB para compatibilidad interna
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inferencia Sliding Window
        # Devuelve probabilidades (H, W, 4)
        probs = sliding_window_inference(frame_rgb, model)
        
        # Overlay
        overlay = frame.copy()
        found_defects = []
        
        # 0-3 mapping to 1-4
        for cls_idx in range(4):
            threshold = THRESHOLDS[cls_idx]
            cls_mask = (probs[:, :, cls_idx] > threshold).astype(np.uint8)
            
            if cls_mask.max() > 0:
                real_class_id = cls_idx + 1
                color = COLORS[real_class_id]
                
                # Encontrar contornos
                contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                
                # Relleno semitransparente
                overlay[cls_mask == 1] = overlay[cls_mask == 1] * 0.5 + np.array(color) * 0.5
                found_defects.append(CLASSES[real_class_id])
        
        # Texto Informativo
        cv2.putText(overlay, f"Defectos: {', '.join(found_defects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        curr_time = time.time()
        fps = 1 / (max(curr_time - fps_time, 0.001))
        fps_time = curr_time
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Real-Time Defect Detection', overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    # Permitir pasar argumento por línea de comandos
    src = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Si es dígito, convertir a int (índice de cámara)
        if arg.isdigit():
            src = int(arg)
        else:
            src = arg # Archivo de video
            
    run_realtime_inference(src)
