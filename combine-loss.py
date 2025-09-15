#--- 2025-09-13 20-11 – by Dr. Thawatchai Chomsiri
import numpy as np
import os

# รายชื่อไฟล์ที่ต้องการอ่าน
activation_files = [
    "loss_000_LSGELU9999.npz",
    "loss_000_LSGELU9950.npz",

    #"loss_000_GELU.npz",    
    #"loss_000_LSGELU9900.npz",
    #"loss_000_LSGELU9750.npz",
    #"loss_000_LSGELU9500.npz",
    #"loss_000_LSGELU9332.npz",
    #"loss_000_LSGELU9250.npz",
    #"loss_000_LSGELU9000.npz",
    #"loss_000_LSGELU8000.npz",
    #"loss_000_LSGELU7500.npz",
    #"loss_000_LSGELU6666.npz",
    
    #"loss_000_Swish.npz",
    #"loss_000_ELU.npz",
    #"loss_000_ReLU.npz",
        
]

# Dictionary เพื่อเก็บ loss ของแต่ละ activation function
combined_loss = {}

# อ่านแต่ละไฟล์และดึงค่า loss ออกมา
for file in activation_files:
    if os.path.exists(file):
        with np.load(file) as data:
            # Extract the activation name from the filename
            activation_name = file.split('_')[2].replace('.npz', '')
            # Store the loss
            combined_loss[activation_name] = data['loss']

# บันทึกค่า combined เป็นไฟล์เดียว
np.savez("loss_000.npz", **combined_loss)

print(f"Combined loss saved to loss_000.npz: {combined_loss}")
