import os
import json
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.callbacks import EarlyStopping
from joblib import dump, load

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

# Normalizar los datos
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Exporta los escaladores para su uso futuro
dump(scaler_X, 'scaler_X.joblib')
dump(scaler_y, 'scaler_y.joblib')

# Convertir los datos a formato de series temporales
n_features = X.shape[1]  # número de características
n_steps = 10  # número de pasos de tiempo a considerar

X_series = np.zeros((X.shape[0] - n_steps + 1, n_steps, n_features))
y_series = y[n_steps - 1:]

for i in range(n_steps, X.shape[0] + 1):
    X_series[i - n_steps] = X[i - n_steps:i]

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_series, y_series, test_size=0.2, random_state=42)

#Funcion de parada temprana
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0012, patience=10, verbose=1, mode='min', restore_best_weights=True)

# Crear el modelo LSTM con capa de convolución
model = Sequential()

# Primero agregar una capa de convolución
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))

# Luego agregar capas LSTM
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Entrena el modelo
model.fit(X_train, y_train, validation_split=0.2, epochs=75, verbose=1, callbacks=[early_stopping])

loss = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {loss}')

# Guarda el modelo entrenado
model.save('lstm_model.h5')