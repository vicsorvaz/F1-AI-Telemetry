import json
from telemetry_f1_2021.listener import TelemetryListener
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from keras.models import load_model

listener = TelemetryListener(port=20777, host='localhost')

# Carga el modelo entrenado
regr = joblib.load('random_forest_regressor.pkl')
model_lstm = load_model('lstm_model.h5')
model_lstmconv = load_model('lstm_modelconv.h5')
sgd = joblib.load('sgd_regressor.pkl')
scaler = joblib.load('scaler.pkl')

last_10_steps = []

X = []
y = []

speeds = []
accelerations = []

# Preparar la gráfica
plt.ion()
fig, ax = plt.subplots(figsize=(10, 2))  # Ajusta el tamaño de la figura aquí
line, = ax.plot([], [])
accel_line, = ax.plot([], [])

def update_graph():
    global line, speeds, accelerations, accel_line, ax

    line.set_ydata(speeds)
    line.set_xdata(range(len(speeds)))
    accel_line.set_ydata(accelerations)
    accel_line.set_xdata(range(len(accelerations)))
    ax.relim()
    ax.autoscale_view()
    # Ajusta los límites del eje X para que abarquen todos los datos
    ax.set_xlim(0, len(speeds))
    plt.draw()
    plt.pause(0.01)

# Preparar la ventana de Tkinter
root = tk.Tk()
root.title("Velocidad en tiempo real")

# Establecer la posición de la ventana de Tkinter
root.geometry("+50+50")


gear_label = tk.Label(root, text="0", font=("Helvetica", 75))
speed_label = tk.Label(root, text="0 KM/H", font=("Helvetica", 72), width=8)
rpm_label = tk.Label(root, text="0", font=("Helvetica", 40))
car_position_label = tk.Label(root, text="0", font=("Helvetica", 20))
laptime_label = tk.Label(root, text="0", font=("Helvetica", 12))
prediction_label = tk.Label(root, text="0", font=("Helvetica", 20))
diff_last_label = tk.Label(root, text="0", font=("Helvetica", 30))
diff_best_label = tk.Label(root, text="0", font=("Helvetica", 30))

# Agregar el botón de parada
stop_button = tk.Button(root, text="Stop", command=root.quit)


gear_label.pack()
speed_label.pack()
rpm_label.pack()
car_position_label.pack()
laptime_label.pack()
prediction_label.pack()
diff_last_label.pack()
#diff_best_label.pack()
stop_button.pack()

def save_data_to_file(filename):
    data = {'X': X, 'y': y}
    with open(filename, 'w') as f:
        json.dump(data, f)

def update_gear(gear):
    gear_label.config(text=f"{gear}")
    root.update()

def update_speed(speed):
    speed_label.config(text=f"{speed} KM/H".zfill(3))
    root.update()

def update_rpm(rpm):
    rpm_label.config(text=f"{rpm}", fg="red")
    root.update()

def update_car_position(car_position):
    car_position_label.config(text=f"{car_position}", fg="green")
    root.update()

def update_laptime(laptime):
    laptime_label.config(text=f"{laptime}", fg="green")
    root.update()

def update_prediction(prediction):
    prediction_label.config(text=f"{prediction}", fg="black")
    root.update()

def update_diff_last(future_laptime_ms, last_lap):
    fg_i= "black"
    if future_laptime_ms > last_lap:
        fg_i = "red"
    else:
        fg_i = "green"
    s, ms = divmod(future_laptime_ms, 1000)
    m, s = divmod(s, 60)
    future_laptime = "%02d:%02d:%03d" % (m, s, ms)

    diff_last_label.config(text=f"{future_laptime}", fg = fg_i)
    root.update()

future_laptime=0
def update_diff_best(future_laptime_ms, last_lap):
    fg_i= "black"
    if future_laptime_ms > last_lap:
        fg_i = "red"
    else:
        fg_i = "green"
    s, ms = divmod(future_laptime_ms, 1000)
    m, s = divmod(s, 60)
    future_laptime = "%02d:%02d:%03d" % (m, s, ms)

    diff_best_label.config(text=f"{future_laptime}", fg = fg_i)
    root.update()

# Establecer la posición de la ventana de Matplotlib
fig.canvas.manager.window.geometry("+500+150")

car_position = -1
first_time = True
data_number = 0
last_lap_ms = 0
lap_times = list()

def best_lap(lap_times):
    if not lap_times:
        return 0  # Retorna None si la lista está vacía

    mayor = lap_times[0]  # Asigna el primer elemento como el mayor valor inicialmente

    for elemento in lap_times:
        if elemento > mayor:
            mayor = elemento

    return mayor


while True:
    packet = listener.get()
    data = json.loads(str(packet))
    if data["m_header"]["m_packet_id"] == 6:
        speed = data["m_car_telemetry_data"][0]["m_speed"]
        acceleration = data["m_car_telemetry_data"][0]["m_throttle"]
        rpm = data["m_car_telemetry_data"][0]["m_engine_rpm"]
        gear = data["m_car_telemetry_data"][0]["m_gear"]
        speeds.append(speed)
        accelerations.append(acceleration*333)

        # Si car_position es válido, agregamos los datos a X e y
        if car_position >= 0:
            X.append([car_position, rpm, speed, acceleration, gear])
            y.append(laptime_ms)  # Suponiendo que deseas predecir el tiempo de vuelta
            
            new_data = np.array([car_position, rpm, speed, acceleration, gear])


            # Añade el nuevo conjunto de datos a la lista
            last_10_steps.append(new_data)

            if len(last_10_steps) > 10:
                last_10_steps.pop(0)

            if len(last_10_steps) == 10:
                # Primero, normalizamos los datos de entrada usando el escalador que se usó durante el entrenamiento
                input_sequence_normalized = scaler.transform(last_10_steps)
               

                # Luego, convertimos la lista a un array de numpy y expandimos sus dimensiones para que tenga la forma (1, 10, 5)
                input_sequence = np.expand_dims(input_sequence_normalized, axis=0)

                # Ahora, podemos alimentar esta secuencia a nuestro modelo para hacer una predicción
                prediction_ms = model_lstm.predict(input_sequence)

                # Haz algo con la predicción (por ejemplo, imprimirlo)
                print(prediction_ms)

            
                
                #prediction_ms = model_lstmconv.predict([[car_position, rpm, speed, acceleration, gear]])

                s, ms = divmod(prediction_ms, 1000)
                m, s = divmod(s, 60)

                diff = laptime_ms - prediction_ms
                future_laptime = last_lap_ms + diff
                print("future_laptime" + str(future_laptime))
                update_diff_last(future_laptime, last_lap_ms)

                if lap_times:
                    lap_times = [num for num in lap_times if num != 0]
                    diff_best = min(lap_times) + diff
                    print("Diffbest" + str(diff_best))
                    update_diff_best(diff_best, min(lap_times))

                # Formatear en formato MM:SS:DD
                prediction = "%02d:%02d:%03d" % (m, s, ms)
                print(f"Predicted lap time: {prediction[0]}")
                update_prediction(prediction)
    
        update_gear(gear)
        update_speed(speed)
        update_rpm(rpm)

        update_graph()

        #print(data["m_car_telemetry_data"][0])

    if data["m_header"]["m_packet_id"] == 2:
        car_position = data["m_lap_data"][0]["m_lap_distance"]
        laptime_ms = data["m_lap_data"][0]["m_current_lap_time_in_ms"]
        last_lap = data["m_lap_data"][0]["m_last_lap_time_in_ms"]
        last_lap_ms = last_lap
        s, ms = divmod(laptime_ms, 1000)
        m, s = divmod(s, 60)
        # Formatear en formato MM:SS:DD
        laptime = "%02d:%02d:%03d" % (m, s, ms)
        update_laptime(laptime)
        update_car_position(car_position)
        if(laptime_ms <= 60 and laptime_ms > 1):
            print("=======================================================================================================")
            data_number += 1
            save_data_to_file(str(data_number)+'.json')
            '''if(y[-1] > 20000):
                lap_times.append(y[-1])
            elif(y[-1] < 20000):
                lap_times.append(y[-2])'''
            # Guarda el gráfico de telemetría después de cada vuelta
            plt.savefig(f'telemetry_{data_number}.svg', format='svg')
            plt.clf()
            plt.cla()
            speeds = []
            accelerations = []

            # Preparar la gráfica
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 2))  # Ajusta el tamaño de la figura aquí
            line, = ax.plot([], [])
            accel_line, = ax.plot([], [])
            X.clear()
            y.clear()
        



