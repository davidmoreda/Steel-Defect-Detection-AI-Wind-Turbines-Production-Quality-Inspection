import cv2
import torch
import numpy as np
import os
import sys
import segmentation_models_pytorch as smp
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===============================
# ===============================
# CONFIGURACIÓN
# ===============================
# Directorio base (donde está este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta a las imágenes (3 niveles arriba desde inference -> organized_experiments -> CNN_Galicia_Toshiba -> datasets)
DEFAULT_IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../../datasets/final_dataset/test/images'))
GT_LABEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../../datasets/final_dataset/test/labels'))

# Rutas posibles del modelo
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, '../UnetConvNeXtTiny/models/severstal_backup_ep35.pth'),
    'severstal_best.pth',
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    print(f"⚠️ NO SE ENCONTRÓ EL MODELO. Se usará nombre por defecto para fallar/advertir después.")
    MODEL_PATH = 'severstal_best.pth'

ENCODER = 'tu-convnext_tiny'
CLASSES = ['Fondo', 'Defecto 1', 'Defecto 2', 'Defecto 3', 'Defecto 4']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COLORS = {
    1: (0, 255, 255),  # Amarillo
    2: (0, 0, 255),    # Rojo
    3: (255, 0, 0),    # Azul
    4: (0, 255, 0)     # Verde
}

# Constants from reference
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
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=5)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Pesos cargados correctamente.")
    except Exception as e:
        print(f"⚠️ Error cargando pesos (Es normal si no has entrenado aún): {e}")
        print("⚠️ USANDO PESOS ALEATORIOS (Solo para demo UI)")
    
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
    # The reference implementation uses a simplified padding or just ensures loop covers it
    # Here we copy logic exactly: simple zero canvas
    
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
                # Output shape: [1, 5, 256, 800] -> We want [1, 4, 256, 800] (ignoring background channel 0)
                probs = torch.sigmoid(output)
                
                # Channels 1,2,3,4
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

def parse_gt_polygons(img_path, h, w):
    """
    Parses YOLO-style polygon labels from .txt file matching the image name.
    """
    filename = os.path.basename(img_path)
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(GT_LABEL_DIR, label_filename)
    
    polygons = [] # List of (class_id, points)
    
    if not os.path.exists(label_path):
        return polygons
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        cls_id = int(parts[0]) + 1 # YOLO is 0-indexed, we use 1-4
        coords = list(map(float, parts[1:]))
        
        # Pairs (x, y)
        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i+1] * h)
            points.append([px, py])
            
        polygons.append((cls_id, np.array(points, dtype=np.int32)))
        
    return polygons

def evaluate_dataset(file_list, model):
    print("\n" + "="*50)
    print(" INICIANDO EVALUACIÓN DE MÉTRICAS (Precision, Recall, IoU)...")
    print(" ESTO PUEDE TARDAR UN POCO...")
    print("="*50)
    
    # Init accumulators
    # Classes 1-4
    tp = {c: 0 for c in range(1, 5)}
    fp = {c: 0 for c in range(1, 5)}
    fn = {c: 0 for c in range(1, 5)}
    
    transform = get_preprocessing()
    
    for idx, filepath in enumerate(file_list):
        print(f"\rProcesando [{idx+1}/{len(file_list)}] {os.path.basename(filepath)}...", end="")
        
        # Load Image
        img = cv2.imread(filepath)
        if img is None: continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Predict
        probs = sliding_window_inference(img_rgb, model, transform) # [H, W, 4]
        
        # Load GT
        gt_polygons = parse_gt_polygons(filepath, h, w)
        
        # Create GT Masks per class
        gt_masks = {c: np.zeros((h, w), dtype=np.uint8) for c in range(1, 5)}
        for cls_id, pts in gt_polygons:
            if cls_id in gt_masks:
                cv2.fillPoly(gt_masks[cls_id], [pts], 1)
                
        # Create Pred Masks per class
        pred_masks = {c: np.zeros((h, w), dtype=np.uint8) for c in range(1, 5)}
        for cls_idx in range(4): # 0-3 -> 1-4
             threshold = THRESHOLDS[cls_idx]
             pred_masks[cls_idx+1] = (probs[:, :, cls_idx] > threshold).astype(np.uint8)
             
        # Compute Stats
        for c in range(1, 5):
            gt = gt_masks[c]
            pred = pred_masks[c]
            
            intersection = (gt & pred).sum()
            tp[c] += intersection
            fp[c] += pred.sum() - intersection
            fn[c] += gt.sum() - intersection
            
    print("\n" + "="*50)
    print(" RESULTADOS FINALES")
    print("="*50)
    print(f"{'CLASE':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'IoU':<10}")
    print("-" * 46)
    
    avg_iou = 0
    valid_classes = 0
    
    for c in range(1, 5):
        TP = tp[c]
        FP = fp[c]
        FN = fn[c]
        
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)
        
        print(f"{CLASSES[c]:<10} | {precision:.4f}     | {recall:.4f}     | {iou:.4f}")
        
        avg_iou += iou
        valid_classes += 1
        
    print("-" * 46)
    print(f"mIoU: {avg_iou / valid_classes:.4f}")
    print("="*50 + "\n")
    print("Evaluación completada. Pulsa cualquier tecla para volver al modo interactivo.")
    cv2.waitKey(0)

def main():
    # 1. Cargar Modelo
    model = load_model()
    
    # 2. Listar Imágenes
    img_dir = DEFAULT_IMAGE_DIR
    if not os.path.exists(img_dir):
        print(f"❌ La carpeta no existe: {img_dir}")
        print("Usando directorio actual como fallback...")
        img_dir = "."
        
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    files = sorted(files)
    if not files:
        print("❌ No se encontraron imágenes en el directorio.")
        return

    index = 0
    total = len(files)
    
    print("\n" + "="*50)
    print(" INTERACTIVE INFERENCE TOOL")
    print("="*50)
    print(f"Directorio: {img_dir}")
    print(f"Imágenes encontradas: {total}")
    print("\nCONTROLES DE VENTANA:")
    print(" [ n ] -> Siguiente imagen")
    print(" [ p ] -> Anterior imagen")
    print(" [ m ] -> CALCULAR MÉTRICAS (Todo el dataset)")
    print(" [ q ] -> Salir")
    print("="*50 + "\n")
    
    # Configurar ventana resizable
    WINDOW_NAME = "Review Tool - Severstal"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 800)

    while True:
        filename = files[index]
        print(f"[{index+1}/{total}] Procesando: {os.path.basename(filename)}...")
        
        # Leer imagen
        img = cv2.imread(filename)
        if img is None:
            print("⚠️ Error leyendo imagen, saltando...")
            index = (index + 1) % total
            continue
            
        # Inferencia
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        transform = get_preprocessing()

        # Inferencia
        probs = sliding_window_inference(img_rgb, model, transform)
        
        # Cargar GT
        gt_polygons = parse_gt_polygons(filename, h, w)
        
        # Visualización
        overlay = img.copy()
        found_defects = []
        
        # Dibujar GT
        for cls_id, pts in gt_polygons:
             cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
        
        # Iterate over classes 0-3 (which map to 1-4)
        for cls_idx in range(4):
            threshold = THRESHOLDS[cls_idx]
            cls_prob = probs[:, :, cls_idx]
            cls_mask = (cls_prob > threshold).astype(np.uint8)
            
            if cls_mask.max() > 0:
                # Map index 0->1, 1->2 etc
                real_class_id = cls_idx + 1
                color = COLORS[real_class_id]
                
                # Contornos
                contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                
                # Relleno
                overlay[cls_mask == 1] = overlay[cls_mask == 1] * 0.6 + np.array(color) * 0.4
                found_defects.append(CLASSES[real_class_id])
                
        # Info Texto
        cv2.putText(overlay, f"Img: {os.path.basename(filename)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"Defects (Pred): {', '.join(found_defects) if found_defects else 'Ninguno'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not found_defects else (0, 0, 255), 2)
        
        gt_classes = set([c for c, _ in gt_polygons])
        gt_names = [CLASSES[c] for c in gt_classes]
        cv2.putText(overlay, f"Defects (GT): {', '.join(gt_names) if gt_names else 'Ninguno'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.putText(overlay, "N: Next | P: Prev | M: Metrics | Q: Quit", (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Mostrar resultado y esperar tecla
        cv2.imshow(WINDOW_NAME, overlay)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("Saliendo...")
            break
        elif key == ord('n'): # Next
            index = (index + 1) % total
        elif key == ord('p'): # Prev
            index = (index - 1 + total) % total
        elif key == ord('m'): # Metrics
            evaluate_dataset(files, model)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
