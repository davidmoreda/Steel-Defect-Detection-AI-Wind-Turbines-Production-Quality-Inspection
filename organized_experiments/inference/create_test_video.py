import cv2
import numpy as np

def create_dummy_video(filename='test_video.mp4', duration=5, fps=30):
    height, width = 256, 1600 # Similar a Severstal
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    print(f"Generando video de prueba: {filename} ({duration}s)...")
    
    frames = duration * fps
    for i in range(frames):
        # Fondo gris oscuro (similar a acero)
        img = np.full((height, width, 3), 100, dtype=np.uint8)
        
        # Ruido para simular textura
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Dibujar un "defecto" que se mueve
        x_pos = int((i / frames) * width)
        y_pos = height // 2
        
        # Defecto tipo 1 (Amarillo) - Elipse
        cv2.ellipse(img, (x_pos, y_pos), (50, 20), 0, 0, 360, (0, 255, 255), -1)
        
        # Defecto tipo 3 (Azul - Rayajo)
        cv2.line(img, (x_pos, y_pos + 50), (x_pos + 100, y_pos + 60), (255, 0, 0), 3)

        out.write(img)

    out.release()
    print("✅ Video generado correctamente.")

if __name__ == "__main__":
    create_dummy_video()
