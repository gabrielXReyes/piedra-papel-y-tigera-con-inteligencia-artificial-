import cv2
import mediapipe as mp
import numpy as np
import time
import random
from sklearn.neighbors import KNeighborsClassifier

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Colores (BGR, tonos suaves para interfaz minimalista)
COLOR_PIEDRA = (100, 100, 255)  # Rojo suave
COLOR_PAPEL = (100, 255, 100)  # Verde suave
COLOR_TIJERA = (255, 100, 100)  # Azul suave
COLOR_DESCONOCIDO = (200, 200, 200)  # Gris claro
COLOR_VICTORIA = (150, 255, 255)  # Amarillo suave
COLOR_DERROTA = (200, 100, 200)  # Morado suave
COLOR_EMPATE = (150, 150, 150)  # Gris medio
COLOR_TEXT = (30, 30, 30)  # Gris oscuro para texto
COLOR_BG_TEXT = (255, 255, 255, 128)  # Fondo semitransparente

# Estados del juego
ESTADO_ESPERANDO = 0
ESTADO_CONTEO = 1
ESTADO_MOSTRAR_RESULTADO = 2

# Entrenar KNN para clasificación de gestos
def train_knn_classifier():
    X_train = []
    y_train = []
    
    # Piedra: dedos cerrados
    for _ in range(100):
        sample = [
            random.uniform(0.1, 0.3),  # índice: y_punta - y_pip
            random.uniform(0.1, 0.3),  # medio
            random.uniform(0.1, 0.3),  # anular
            random.uniform(0.1, 0.3),  # meñique
            random.uniform(-0.05, 0.05)  # pulgar: x_tip - x_ip
        ]
        X_train.append(sample)
        y_train.append("Piedra")
    
    # Papel: dedos extendidos
    for _ in range(100):
        sample = [
            random.uniform(-0.3, -0.1),
            random.uniform(-0.3, -0.1),
            random.uniform(-0.3, -0.1),
            random.uniform(-0.3, -0.1),
            random.uniform(0.1, 0.3)
        ]
        X_train.append(sample)
        y_train.append("Papel")
    
    # Tijera: índice y medio extendidos
    for _ in range(100):
        sample = [
            random.uniform(-0.3, -0.1),
            random.uniform(-0.3, -0.1),
            random.uniform(0.1, 0.3),
            random.uniform(0.1, 0.3),
            random.uniform(0.1, 0.3)
        ]
        X_train.append(sample)
        y_train.append("Tijera")
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

# Determinar gesto con KNN
def get_gesture(hand_landmarks, knn_classifier):
    landmark_points = []
    for landmark in hand_landmarks.landmark:
        landmark_points.append((landmark.x, landmark.y, landmark.z))
    
    # Puntos clave
    thumb_tip = landmark_points[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmark_points[mp_hands.HandLandmark.THUMB_IP]
    index_tip = landmark_points[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmark_points[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = landmark_points[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmark_points[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = landmark_points[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmark_points[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = landmark_points[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmark_points[mp_hands.HandLandmark.PINKY_PIP]
    
    features = [
        index_tip[1] - index_pip[1],
        middle_tip[1] - middle_pip[1],
        ring_tip[1] - ring_pip[1],
        pinky_tip[1] - pinky_pip[1],
        thumb_tip[0] - thumb_ip[0]
    ]
    
    return knn_classifier.predict([features])[0]

# Determinar ganador
def determine_winner(gesture1, gesture2):
    if gesture1 == "Desconocido" or gesture2 == "Desconocido":
        return "Ninguno"
    if gesture1 == gesture2:
        return "Empate"
    elif (gesture1 == "Piedra" and gesture2 == "Tijera") or \
         (gesture1 == "Papel" and gesture2 == "Piedra") or \
         (gesture1 == "Tijera" and gesture2 == "Papel"):
        return "Victoria"
    return "Derrota"

# Generar jugada de la máquina
def get_computer_move():
    return random.choice(["Piedra", "Papel", "Tijera"])

# Preprocesar imagen con Sobel y Canny
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # Sobel
    sobel_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x*2 + sobel_y*2)
    sobel = cv2.convertScaleAbs(sobel)
    
    # Canny
    edges = cv2.Canny(sobel, 100, 200)
    
    # Filtro bilateral
    filtered = cv2.bilateralFilter(equalized, 9, 75, 75)
    filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    # Ajustar brillo y contraste
    adjusted = cv2.convertScaleAbs(filtered_bgr, alpha=1.2, beta=10)
    
    return adjusted, edges

# Segmentación con Watershed
def segment_hand(image, hand_landmarks):
    markers = np.zeros(image.shape[:2], dtype=np.int32)
    
    # Marcar fondo
    markers[0:10, :] = 1
    markers[-10:, :] = 1
    markers[:, 0:10] = 1
    markers[:, -10:] = 1
    
    # Marcar mano
    if hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(markers, (x, y), 5, 2, -1)
    else:
        # Usar centro de la imagen como marcador aproximado
        cv2.circle(markers, (image.shape[1]//2, image.shape[0]//2), 50, 2, -1)
    
    # Aplicar Watershed
    img_copy = image.copy()
    cv2.watershed(img_copy, markers)
    
    # Crear máscara
    mask = np.zeros_like(image)
    mask[markers == 2] = (255, 255, 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented, mask

# Transformada de Hough para líneas
def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    line_image = np.zeros_like(edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    return line_image

# Cargar imágenes de gestos
def load_gesture_images():
    img_size = 150
    font = cv2.FONT_HERSHEY_DUPLEX
    
    piedra_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    cv2.circle(piedra_img, (img_size//2, img_size//2), img_size//4, COLOR_PIEDRA, -1)
    cv2.putText(piedra_img, "Piedra", (img_size//6, img_size//2+10), font, 0.6, COLOR_TEXT, 1)
    
    papel_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    cv2.rectangle(papel_img, (img_size//4, img_size//4), (3*img_size//4, 3*img_size//4), COLOR_PAPEL, -1)
    cv2.putText(papel_img, "Papel", (img_size//6, img_size//2+10), font, 0.6, COLOR_TEXT, 1)
    
    tijera_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    cv2.line(tijera_img, (img_size//4, img_size//4), (3*img_size//4, 3*img_size//4), COLOR_TIJERA, 5)
    cv2.line(tijera_img, (img_size//4, 3*img_size//4), (3*img_size//4, img_size//4), COLOR_TIJERA, 5)
    cv2.putText(tijera_img, "Tijera", (img_size//6, img_size//2+10), font, 0.6, COLOR_TEXT, 1)
    
    return {
        "Piedra": piedra_img,
        "Papel": papel_img,
        "Tijera": tijera_img,
        "Desconocido": np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
    }

# Dibujar texto con fondo semitransparente
def put_text_with_background(image, text, org, font, font_scale, text_color, thickness, bg_color):
    # Obtener tamaño del texto
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    
    # Crear fondo
    overlay = image.copy()
    cv2.rectangle(overlay, (x-5, y-text_h-5), (x+text_w+5, y+5), bg_color[:3], -1)
    
    # Combinar con transparencia
    alpha = bg_color[3] / 255.0
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
    
    # Dibujar texto
    cv2.putText(image, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)

# Función principal
def main():
    # Verificar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    
    # Inicializar KNN
    knn_classifier = train_knn_classifier()
    
    # Variables de estado
    estado_juego = ESTADO_ESPERANDO
    countdown_start = 0
    resultado_mostrado_inicio = 0
    mi_puntuacion = 0
    computadora_puntuacion = 0
    mi_gesto = "Desconocido"
    gesto_computadora = "Desconocido"
    resultado_ronda = ""
    prev_time = 0
    gesture_images = load_gesture_images()
    font = cv2.FONT_HERSHEY_DUPLEX
    
    print("¡Juego Piedra, Papel o Tijera iniciado!")
    print("Presiona ESPACIO para jugar una ronda")
    print("Presiona Q para salir")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error: No se pudo capturar la imagen.")
            break
        
        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape
        
        # Preprocesar
        processed_image, edges = preprocess_image(image)
        
        # Procesar con MediaPipe primero para obtener landmarks
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Segmentar mano usando landmarks
        segmented_image, mask = segment_hand(processed_image, results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None)
        
        # Detectar líneas
        line_image = detect_lines(edges)
        
        # Crear imagen de procesamiento (ventana secundaria)
        processing_image = np.zeros_like(image)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        line_bgr = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        processing_image = cv2.addWeighted(edges_bgr, 0.4, line_bgr, 0.3, 0)
        processing_image = cv2.addWeighted(processing_image, 0.7, mask_bgr, 0.3, 0)
        
        # Dibujar detecciones
        display_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = get_gesture(hand_landmarks, knn_classifier)
                mi_gesto = gesture
                
                color = COLOR_DESCONOCIDO
                if gesture == "Piedra":
                    color = COLOR_PIEDRA
                elif gesture == "Papel":
                    color = COLOR_PAPEL
                elif gesture == "Tijera":
                    color = COLOR_TIJERA
                
                mp_drawing.draw_landmarks(
                    display_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2),
                    mp_drawing.DrawingSpec(color=color, thickness=1)
                )
                
                put_text_with_background(
                    display_image,
                    f"Tu gesto: {gesture}",
                    (20, 40),
                    font,
                    0.8,
                    COLOR_TEXT,
                    1,
                    COLOR_BG_TEXT
                )
        else:
            mi_gesto = "Desconocido"
            put_text_with_background(
                display_image,
                "No se detecta la mano",
                (20, 40),
                font,
                0.8,
                COLOR_TEXT,
                1,
                COLOR_BG_TEXT
            )
        
        # Info de computadora
        put_text_with_background(
            display_image,
            f"Computadora: {gesto_computadora}",
            (20, 80),
            font,
            0.8,
            COLOR_TEXT,
            1,
            COLOR_BG_TEXT
        )
        
        # Puntuaciones
        put_text_with_background(
            display_image,
            f"Jugador: {mi_puntuacion} | Computadora: {computadora_puntuacion}",
            (image_width-400, 40),
            font,
            0.8,
            COLOR_TEXT,
            1,
            COLOR_BG_TEXT
        )
        
        # Estados
        curr_time = time.time()
        
        if estado_juego == ESTADO_ESPERANDO:
            put_text_with_background(
                display_image,
                "Presiona ESPACIO para jugar",
                (image_width//2-150, image_height//2),
                font,
                0.8,
                COLOR_TEXT,
                1,
                COLOR_BG_TEXT
            )
        
        elif estado_juego == ESTADO_CONTEO:
            tiempo_transcurrido = curr_time - countdown_start
            if tiempo_transcurrido < 3:
                count = 3 - int(tiempo_transcurrido)
                put_text_with_background(
                    display_image,
                    f"{count}",
                    (image_width//2-30, image_height//2),
                    font,
                    3,
                    COLOR_TEXT,
                    3,
                    COLOR_BG_TEXT
                )
            else:
                if tiempo_transcurrido < 3.5:
                    put_text_with_background(
                        display_image,
                        "YA!",
                        (image_width//2-50, image_height//2),
                        font,
                        2,
                        COLOR_VICTORIA,
                        2,
                        COLOR_BG_TEXT
                    )
                else:
                    gesto_computadora = get_computer_move()
                    resultado_ronda = determine_winner(mi_gesto, gesto_computadora)
                    if resultado_ronda == "Victoria":
                        mi_puntuacion += 1
                    elif resultado_ronda == "Derrota":
                        computadora_puntuacion += 1
                    estado_juego = ESTADO_MOSTRAR_RESULTADO
                    resultado_mostrado_inicio = curr_time
        
        elif estado_juego == ESTADO_MOSTRAR_RESULTADO:
            if curr_time - resultado_mostrado_inicio < 3:
                color_resultado = COLOR_EMPATE
                display_text = resultado_ronda
                if resultado_ronda == "Victoria":
                    color_resultado = COLOR_VICTORIA
                elif resultado_ronda == "Derrota":
                    color_resultado = COLOR_DERROTA
                elif resultado_ronda == "Ninguno":
                    display_text = "Gesto no valido"
                
                put_text_with_background(
                    display_image,
                    display_text,
                    (image_width//2-100, image_height//2),
                    font,
                    1.2,
                    color_resultado,
                    2,
                    COLOR_BG_TEXT
                )
                
                if gesto_computadora in gesture_images:
                    computer_gesture_img = gesture_images[gesto_computadora]
                    h, w = computer_gesture_img.shape[:2]
                    display_image[20:20+h, image_width-20-w:image_width-20] = computer_gesture_img
            else:
                estado_juego = ESTADO_ESPERANDO
        
        # FPS
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        put_text_with_background(
            display_image,
            f"FPS: {int(fps)}",
            (image_width-100, image_height-20),
            font,
            0.6,
            COLOR_TEXT,
            1,
            COLOR_BG_TEXT
        )
        
        # Mostrar imágenes
        cv2.imshow('Piedra, Papel o Tijera', display_image)
        cv2.imshow('Procesamiento', cv2.resize(processing_image, (320, 240)))  # Reducir tamaño
        
        # Controles
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            if estado_juego == ESTADO_ESPERANDO:
                estado_juego = ESTADO_CONTEO
                countdown_start = curr_time
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()