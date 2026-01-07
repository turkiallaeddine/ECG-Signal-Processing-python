
# ECG Signal Processing – First Project

## 🧠 Introduction
This project demonstrates how to **download real ECG signals** from the **PhysioNet database**, analyze the data, plot the signal, and save the results as **CSV** and image files using **Python**.

The project is ideal for students and researchers in **Biomedical Engineering** and aims to teach:
- Working with real ECG data.
- Performing basic statistical analysis on signals.
- Plotting signals with **Matplotlib**.
- Saving cleaned data for future projects like **advanced signal processing** or **Machine Learning**.

---

## 🔗 Sources
- [PhysioNet – MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [wfdb Python package](https://github.com/MIT-LCP/wfdb-python)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---

## 🛠️ Requirements
- Python 3.x
- Libraries:
```python
pip install wfdb numpy pandas matplotlib


## 🚀 Project Steps
1️⃣ Install Libraries
!pip install wfdb numpy pandas matplotlib

2️⃣ Load Real ECG Data
We use record 100 from the MIT-BIH Arrhythmia database.
Extract the first 3000 samples.
نسخ التعليمات البرمجية
Python
import wfdb
record = wfdb.rdrecord('100', sampfrom=0, sampto=3000, pn_dir='mitdb')
3️⃣ Extract ECG Signal
Take the first channel:
نسخ التعليمات البرمجية
Python
ec_signal = record.p_signal[:, 0]
4️⃣ Convert to DataFrame
نسخ التعليمات البرمجية
Python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Sample': np.arange(len(ec_signal)),
    'ECG': ec_signal
})
5️⃣ Statistical Analysis
نسخ التعليمات البرمجية
Python
import numpy as np
print("Mean:", np.mean(ec_signal))
print("Std:", np.std(ec_signal))
print("Max:", np.max(ec_signal))
print("Min:", np.min(ec_signal))
6️⃣ Plot the Signal
نسخ التعليمات البرمجية
Python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(ec_signal, color='blue')
plt.title("ECG Signal from PhysioNet (Record 100)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("images/ecg_real_signal.png")
plt.show()
7️⃣ Save Data
Save cleaned data as CSV:
نسخ التعليمات البرمجية
Python
df.to_csv("data/ecg_real_data.csv", index=False)
Download files from Colab (optional):
نسخ التعليمات البرمجية
Python
from google.colab import files
files.download("images/ecg_real_signal.png")
files.download("data/ecg_real_data.csv")
📌 Notes
You can change sampto to include more samples.
Different records from PhysioNet can be used to experiment with multiple ECG signals.
This project serves as a foundation before moving to:
Signal denoising / filtering
QRS peak detection
Feature extraction
Machine Learning applications
🎯 Project Goals
Teach how to use Python and wfdb to work with real ECG signals.
Prepare cleaned data for Signal Processing and Biomedical AI applications.
Learn how to analyze, visualize, and store biomedical signal data in a structured way.
Provide a step-by-step educational guide for beginners in biomedical signal processing.
