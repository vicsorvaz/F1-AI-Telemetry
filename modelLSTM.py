import os
import json
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler  # Importa la clase MinMaxScaler

# Ubicación de los archivos JSON
folder = './DATA_OLD/4'

# Listas para los datos
X = []
y = []

# Lee todos los archivos en la carpeta
for filename in os.listdir(folder):
    # Ignora el archivo 1.json y solo carga los archivos .json
    if filename.endswith('.json') and filename != '1.json':
        with open(os.path.join(folder, filename), 'r') as f:
            data = json.load(f)
            X.extend(data['X'])
            y.extend(data['y'])

# Convierte las listas a arrays de NumPy
X = np.array(X)
y = np.array(y)

# Normaliza los datos de X
scaler = MinMaxScaler(feature_range=(0, 1))  # Crea una instancia del escalador
X_scaled = scaler.fit_transform(X)  # Ajusta el escalador a los datos y los transforma

# Convertir los datos a formato de series temporales
n_features = X_scaled.shape[1]  # número de características
n_steps = 10  # número de pasos de tiempo a considerar

X_series = np.zeros((X_scaled.shape[0] - n_steps + 1, n_steps, n_features))
y_series = y[n_steps - 1:]

for i in range(n_steps, X_scaled.shape[0] + 1):
    X_series[i - n_steps] = X_scaled[i - n_steps:i]

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_series, y_series, test_size=0.2, random_state=42)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))

# Segunda capa LSTM y capa de dropout
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

# Tercera capa LSTM y capa de dropout
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Entrena el modelo
model.fit(X_train, y_train, epochs=50, verbose=1)

loss = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {loss}')

# Guarda el modelo entrenado
model.save('lstm_model.h5')
