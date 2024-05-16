import cv2

# Inicialize a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Captura frame-a-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Converte para escala de cinza para detecção de rosto
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Carrega o classificador de rostos pré-treinado
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detecta rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibe o frame resultante
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quando tudo estiver feito, libere a captura
cap.release()
cv2.destroyAllWindows()
