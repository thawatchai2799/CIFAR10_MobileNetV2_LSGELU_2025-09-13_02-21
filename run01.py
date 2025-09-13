#--- 2025-09-13 11-36 – by Dr. Thawatchai Chomsiri
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Model
import datetime

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Dense

import datetime
import numpy as np
import os
import re
import math
import pickle

  
def lsgelu(x):    # Left-Shifted GELU with 1 range
    return x * 0.5 * (1 + tf.math.erf((x + 1.5) / tf.sqrt(2.0)))

def lsgelu9999(x):    
    S=3.71901648546
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9950(x):    
    S=2.57582930355
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9900(x):    
    S=2.32634787404
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9750(x):    
    S=1.95996398454
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9500(x):    
    S=1.64485362695
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

#-----
def lsgelu9332(x): # LSGELU   
    S=1.5
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

#-----
def lsgelu9250(x):    
    S=1.43953147094
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9000(x):    
    S=1.28155156554
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu8000(x):    
    S=0.841621233573
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu7500(x):    
    S=0.674489750196
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu6666(x):    
    S=0.430727299295
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

#-----
def lsgelu2(x):    # Left-Shifted GELU with 2 range
    return tf.where(
        x >= 0,
        x, 
        x * 0.5 * (1 + tf.math.erf((x + 1.5) / tf.sqrt(2.0)))
    ) 

def lsgelu3(x):    # Left-Shifted GELU with 3 range
    L = -3.00
    return tf.where(
        x >= 0,
        x,
        tf.where(
            x >= L,
            x * 0.5 * (1 + tf.math.erf((x + 1.5) / tf.sqrt(2.0))),
            tf.zeros_like(x)
        )
    ) 

# Function สำหรับสร้างโมเดล
def build_model(activation_fn):
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), strides=2, padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)

    def bottleneck_block(x, filters):
        dw = DepthwiseConv2D((3, 3), padding='same')(x)
        dw = BatchNormalization()(dw)
        dw = Activation(activation_fn)(dw)

        pw = Conv2D(filters, (1, 1), padding='same')(dw)
        pw = BatchNormalization()(pw)
        pw = Activation(activation_fn)(pw)
        return pw

    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

activations_list = {
    #"LSGELU9999": lsgelu9999,
    "LSGELU9950": lsgelu9950,
    "LSGELU9900": lsgelu9900,
    "LSGELU9750": lsgelu9750,
    "LSGELU9500": lsgelu9500,
    "LSGELU9332": lsgelu9332, # LSGELU
    "LSGELU9250": lsgelu9250,
    "LSGELU9000": lsgelu9000,
    "LSGELU8000": lsgelu8000,
    "LSGELU7500": lsgelu7500,
    "LSGELU6666": lsgelu6666,
    
    'GELU': tf.nn.gelu,
    'ELU': tf.nn.elu,
    'ReLU': tf.nn.relu,
    'Swish': tf.nn.swish,     
}

epochs = 101  ################
num_runs = 30 ###############
batch_size = 64 ###############
results = {
    'activation': [],
    'accuracy_per_epoch': []
}
accuracy_summary = {}

for run_idx in range(num_runs):
    print(f"\n--- Run {run_idx:03d} of {num_runs:03d} ---")
    results = {}
    accuracy_results = {}
    loss_results = {}
    
    for act_name, act_fn in activations_list.items():
        print(f"Running: Activation={act_name} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # เทรน
        model = build_model(act_fn)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)
        
        print(f"Running: Activation={act_name} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")   
        results[act_name] = {key: np.array(val) for key, val in history.history.items()}
        print(f"Details in results for round {run_idx:03d}:")
        for act_name, metrics_dict in results.items():
            print(f"\nActivation: {act_name}")
            for metric_name, metric_values in metrics_dict.items():
                print(f"  {metric_name}: {metric_values}")


        np.savez(f"accuracy_{run_idx:03d}_{act_name}.npz", accuracy=np.array(history.history['accuracy']))
        np.savez(f"loss_{run_idx:03d}_{act_name}.npz", loss=np.array(history.history['loss']))
        
        accuracy_results[act_name] = np.array(history.history['accuracy'])
        loss_results[act_name] = np.array(history.history['loss'])

    print(f" ----- Data -------- ")
    results[act_name] = {key: np.array(val) for key, val in history.history.items()}
    print(f"Details in results for round {run_idx:03d}:")
    for act_name, metrics_dict in results.items():
        print(f"\nActivation: {act_name}")
        for metric_name, metric_values in metrics_dict.items():
            print(f"  {metric_name}: {metric_values}")

    print(f" ------------------- ")
    ###np.savez(f"accuracy_{run_idx:03d}.npz", **accuracy_results)
    ###np.savez(f"loss_{run_idx:03d}.npz", **loss_results)

folder_path = '.'
accuracy_files = sorted([f for f in os.listdir(folder_path) if re.match(r'accuracy_\d+\.npz', f)])
loss_files = sorted([f for f in os.listdir(folder_path) if re.match(r'loss_\d+\.npz', f)])

accuracy_collections = {}
for fname in accuracy_files:
    acc_data = np.load(os.path.join(folder_path, fname), allow_pickle=True)
    for key in acc_data.files:
        if key not in accuracy_collections:
            accuracy_collections[key] = []
        accuracy_collections[key].append(acc_data[key])

accuracy_avg = {}
for act_name, vals in accuracy_collections.items():
    stacked = np.vstack(vals)
    accuracy_avg[act_name] = np.mean(stacked, axis=0)

loss_collections = {}
for fname in loss_files:
    los_data = np.load(os.path.join(folder_path, fname), allow_pickle=True)
    for key in los_data.files:
        if key not in loss_collections:
            loss_collections[key] = []
        loss_collections[key].append(los_data[key])

loss_avg = {}
for act_name, vals in loss_collections.items():
    stacked = np.vstack(vals)
    loss_avg[act_name] = np.mean(stacked, axis=0)



for i in range(len(accuracy_files)):
    fname_acc = accuracy_files[i]
    fname_loss = loss_files[i]
    print(f"\n--- Loading {fname_acc} and {fname_loss} ---")
    acc_data = np.load(os.path.join(folder_path, fname_acc), allow_pickle=True)
    loss_data = np.load(os.path.join(folder_path, fname_loss), allow_pickle=True)

    print("Keys accuracy:", list(acc_data.keys()))
    print("Keys loss:", list(loss_data.keys()))

    activation_names = list(acc_data.files)

   
print(f"\nEND at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
