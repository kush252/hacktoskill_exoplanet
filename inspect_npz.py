import numpy as np

file_path = r"d:\Kush\2nd Year\Hackathons\hacktoskill_exoplanet\dataset\CONFIRMED\10000941.npz"
data = np.load(file_path)

print("Keys:", data.files)
for key in data.files:
    arr = data[key]
    print(f"Key: {key}, Shape: {arr.shape}, Dtype: {arr.dtype}")
