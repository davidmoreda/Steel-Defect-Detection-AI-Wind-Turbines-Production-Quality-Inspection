import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Configuración
IMAGE_PATH = 'organized_experiments/inference/inference_results/raw_image_toshiba_77.jpg'
TILE_SIZE = 512
OUTPUT_FILE = 'tiling_strategy_demo.png'

# Coordenadas manuales del tile (Si sabes dónde está el defecto, ajusta esto)
# Para este ejemplo, buscamos una zona con variación (un borde o mancha) automáticamente
# o usamos un valor fijo interesante.
# Nota: En (1500, 2500) suele haber detalles en imágenes centradas, pero ajustaremos dinámicamente.
START_Y = 1500
START_X = 2500

def visualize_tiling():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: No se encuentra {IMAGE_PATH}")
        return

    # 1. Cargar Imagen Gigante
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("Error cargando la imagen.")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    print(f"Dimensiones Originales: {w}x{h}")

    # Asegurar que el crop no se salga
    y = min(START_Y, h - TILE_SIZE)
    x = min(START_X, w - TILE_SIZE)

    # 2. Extraer el Tile (Recorte) a Resolución Nativa
    tile = img_rgb[y:y+TILE_SIZE, x:x+TILE_SIZE]

    # 3. Crear Visualización Comparativa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Panel Izquierdo: Imagen Completa (Downsampled por visualización) ---
    ax1.imshow(img_rgb)
    ax1.set_title(f"Imagen Completa: {w}x{h} px\n(Reescalada para visión global)", fontsize=14)
    # Dibujar el rectángulo del tile
    rect = patches.Rectangle((x, y), TILE_SIZE, TILE_SIZE, linewidth=3, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.axis('off')

    # --- Panel Derecho: Tile a Full Resolución ---
    ax2.imshow(tile)
    ax2.set_title(f"Tile (Recorte): {TILE_SIZE}x{TILE_SIZE} px\nResolución Nativa Preservada", fontsize=14)
    
    # Añadir anotaciones para explicar
    plt.suptitle("Estrategia de Tiling: Preservación de Detalles para Defectos Pequeños", fontsize=16, weight='bold', y=0.98)
    
    # Guardar con ajuste de layout manual para evitar solapamientos
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Visualización guardada en: {os.path.abspath(OUTPUT_FILE)}")
    # plt.show() # Comentado para entorno headless

if __name__ == "__main__":
    visualize_tiling()
