import os
import json
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

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
            #y.extend([float(time.replace(':', '').replace(' ', '')) for time in data['y']])
            y.extend(data['y'])

# Convierte las listas a arrays de NumPy
X = np.array(X)
y = np.array(y)

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un escalador de características y ajusta los datos de entrenamiento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Escala los datos de prueba con el mismo escalador
X_test = scaler.transform(X_test)

# Entrena el modelo SGD
regr = SGDRegressor(random_state=42)
regr.fit(X_train, y_train)

# Evalúa el rendimiento del modelo
score = regr.score(X_test, y_test)
print(f"R^2 score: {score}")

# Guarda el modelo entrenado y el escalador
joblib.dump(regr, 'sgd_regressor.pkl')
joblib.dump(scaler, 'scaler.pkl')
