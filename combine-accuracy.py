#--- 2025-09-13 20-11 – by Dr. Thawatchai Chomsiri
import numpy as np
import os

# รายชื่อไฟล์ที่ต้องการอ่าน
activation_files = [
    "accuracy_000_LSGELU9999.npz",
    "accuracy_000_LSGELU9950.npz",

    #"accuracy_000_GELU.npz",    
    #"accuracy_000_LSGELU9900.npz",
    #"accuracy_000_LSGELU9750.npz",
    #"accuracy_000_LSGELU9500.npz",
    #"accuracy_000_LSGELU9332.npz",
    #"accuracy_000_LSGELU9250.npz",
    #"accuracy_000_LSGELU9000.npz",
    #"accuracy_000_LSGELU8000.npz",
    #"accuracy_000_LSGELU7500.npz",
    #"accuracy_000_LSGELU6666.npz",
    
    #"accuracy_000_Swish.npz",
    #"accuracy_000_ELU.npz",
    #"accuracy_000_ReLU.npz",
    
]

# Dictionary เพื่อเก็บความแม่นยำของแต่ละ activation function
combined_accuracies = {}

# อ่านแต่ละไฟล์และดึงค่าความแม่นยำออกมา
for file in activation_files:
    if os.path.exists(file):
        with np.load(file) as data:
            # Extract the activation name from the filename
            activation_name = file.split('_')[2].replace('.npz', '')
            # Store the accuracy
            combined_accuracies[activation_name] = data['accuracy']

# บันทึกค่า combined เป็นไฟล์เดียว
np.savez("accuracy_000.npz", **combined_accuracies)

print(f"Combined accuracies saved to accuracy_000.npz: {combined_accuracies}")
