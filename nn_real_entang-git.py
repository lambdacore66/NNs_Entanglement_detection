"""
Compilación de una red neuronal con tensorflow para aproximar el rango de Schmidt de un estado cuántico
arbitario de 2 qubits.

@author: Antonio de la M. Sojo Lopez
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard #Para sacar gráficas del proceso

"""
Cargar los datos desde un .csv a un np.array (N, 8) real
"""

txt_test = open('test.csv', 'r');
data_test = np.loadtxt(txt_test, delimiter=",")
###################################################################################
txt_test_sv = open('test_sol.csv', 'r');
test_sv = np.loadtxt(txt_test_sv, delimiter=",")
###################################################################################
txt_train = txt_test = open('train.csv', 'r');
data_train= np.loadtxt(txt_train, delimiter=",")
###################################################################################
txt_train_sv = txt_test = open('train_sol.csv', 'r');
train_sv = np.loadtxt(txt_train_sv, delimiter=",")
##################################################################################

"""
Definimos la estructura de la red. 
"""

structure = tf.keras.Sequential([
    tf.keras.layers.Dense(units=20, activation = tf.nn.relu, input_shape=[8]),  # Input de 4 coeficientes complejos como 4x2 reales.
    tf.keras.layers.Dense(15, activation = tf.nn.relu),   # 2a capa con activación relu
    tf.keras.layers.Dense(10, activation = tf.nn.relu),   # 3a capa con activación relu
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation = tf.nn.relu),    # 4a capa con activación relu
    tf.keras.layers.Dense(2, activation = tf.nn.relu),    # 5a capa con activación relu
    tf.keras.layers.Dense(1, activation = 'sigmoid')      # Salida. 
    ])  

"""
Configuración de la compilación
"""

structure.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.BinaryCrossentropy(), #F. Loss para clasificar. Usa entropía binaria cruzada.
    metrics = ['accuracy'] 
    )

"""
Cargamos el logger
"""

#tensorboard_nn = TensorBoard(log_dir = 'logs/real_entang')
logger = CSVLogger('log.csv', append=True, separator=';')

"""
Entrenamiento
"""
model = structure.fit(data_train, train_sv, batch_size = 50, epochs=1000, steps_per_epoch = 200, callbacks = [logger], validation_data=(data_test, test_sv),)
    

