
import cv2
import os
import numpy as np

# Função para carregar imagens e redimensioná-las
def load_and_preprocess_images(image_dir, img_size=(128, 128)):
    images = []
    labels = []
    
    for label_dir in os.listdir(image_dir):
        path = os.path.join(image_dir, label_dir)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path)  # Carrega a imagem
            img = cv2.resize(img, img_size)  # Redimensiona a imagem
            images.append(img)
            labels.append(label_dir)  # Atribui o rótulo (viva/morta)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

from sklearn.preprocessing import LabelEncoder

def encode_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return labels_encoded


from sklearn.model_selection import train_test_split

def split_data(images, labels_encoded):
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

from tensorflow.keras import layers, models

def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Saída binária (viva/morta)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Normalizar as imagens (0-255 para 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Criar e treinar o modelo
input_shape = (128, 128, 3)
model = create_cnn_model(input_shape)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Avaliação
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisão no conjunto de teste: {test_acc}")

# Salvar o modelo
model.save('muda_classifier_model.h5')

# Carregar o modelo salvo
model = tf.keras.models.load_model('muda_classifier_model.h5')

# Captura de imagem com OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Captura o frame da câmera
    if not ret:
        break
    
    # Pré-processar a imagem (redimensionar, normalizar)
    img_resized = cv2.resize(frame, (128, 128))
    img_resized = np.expand_dims(img_resized / 255.0, axis=0)
    
    # Fazer a previsão
    prediction = model.predict(img_resized)
    
    if prediction[0] > 0.5:
        print("Muda morta detectada!")
    else:
        print("Muda viva detectada!")
    
    # Mostrar a imagem capturada
    cv2.imshow("Horta Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

