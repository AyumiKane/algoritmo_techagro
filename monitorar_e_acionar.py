import cv2
import numpy as np
import tensorflow as tf
import time  # Para controle de tempo
# Importar biblioteca de GPIO conforme o hardware utilizado. Exemplo:
# import RPi.GPIO as GPIO

# Carregar o modelo treinado
model = tf.keras.models.load_model('muda_classifier_model.h5')

# Função para acionar a garra
def acionar_garra():
    print("Ação: Garra acionada para remover planta morta.")
    # Código para controlar a garra aqui
    # Exemplo para Raspberry Pi:
    # GPIO.output(pin_garra, GPIO.HIGH)
    # time.sleep(2)
    # GPIO.output(pin_garra, GPIO.LOW)

# Captura de imagem e detecção
cap = cv2.VideoCapture(0)  # Abrir a câmera

while True:
    ret, frame = cap.read()  # Captura a imagem
    if not ret:
        break
    
    # Pré-processamento da imagem
    img_resized = cv2.resize(frame, (128, 128))
    img_resized = np.expand_dims(img_resized / 255.0, axis=0)  # Normalizar e expandir dimensão
    
    # Fazer a previsão
    prediction = model.predict(img_resized)[0][0]
    
    if prediction > 0.5:
        print("Planta morta detectada!")
        acionar_garra()  # Acionar a garra para remoção
    else:
        print("Planta viva detectada.")
    
    # Exibir a imagem na janela
    cv2.imshow("Monitoramento da Horta", frame)
    
    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
