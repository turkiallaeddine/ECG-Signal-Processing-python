# ECG Signal Processing – First Project

## 🧠 Introduction
This project demonstrates how to **download real ECG signals** from the **PhysioNet database**, analyze the data, plot the signal, and save the results as **CSV** and image files using **Python**.

The project is ideal for students and researchers in **Biomedical Engineering** and aims to teach:
- How to work with **real ECG signals**.
- Performing **basic statistical analysis**.
- Plotting signals with **Matplotlib**.
- Saving cleaned data for future projects like **signal processing** or **Machine Learning**.


![ECG Signal](ecg_real_signal.png)

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

 wfdb
 numpy 
pandas 
matplotlib





## Why WFDB and CSV Are Used in This Project

This project works with **real clinical ECG signals** obtained from PhysioNet, which is a well-known open-source repository for biomedical signals.

PhysioNet stores physiological data using specialized medical formats such as `.dat` and `.hea`. These formats are optimized for clinical storage and annotation, but they are not convenient for direct data analysis or visualization.

To handle these medical formats, we use the **WFDB (WaveForm DataBase) Python library**.  
WFDB is a specialized library designed to:
- Read real physiological signals (ECG, EEG, EMG, etc.)
- Load multi-channel biomedical recordings
- Preserve the original clinical signal quality

Using WFDB allows us to access **raw, real-world ECG data** exactly as recorded in clinical studies.

However, after loading the ECG signal, working directly with WFDB files is not practical for repeated analysis, signal processing, or machine learning. For this reason, the signal is converted and saved in **CSV (Comma-Separated Values) format**.

CSV is a simple, text-based format used to store tabular data where:
- Each row represents a time sample
- Each column represents a variable (e.g., ECG amplitude)

Saving ECG data as CSV provides several advantages:
- It is human-readable and easy to inspect
- It is compatible with Python, MATLAB, Excel, and scientific software
- It simplifies signal processing, filtering, and feature extraction
- It enables reproducible research and easy data sharing

In biomedical engineering workflows, it is common practice to:
1. Load raw clinical data using specialized tools (WFDB)
2. Convert the signal into a universal analysis format (CSV)
3. Perform signal processing, visualization, and machine learning

This approach ensures accuracy, flexibility, and reproducibility in biomedical signal analysis.


---

🚀 Project Steps

1️⃣ Install Libraries

First, install all necessary Python packages:

```python
!pip install wfdb numpy pandas matplotlib
```

2️⃣ Load Real ECG Data

We use record 100 from MIT-BIH Arrhythmia database and extract the first 3000 samples:

```python
import wfdb
record = wfdb.rdrecord('100', sampfrom=0, sampto=3000, pn_dir='mitdb')
```

3️⃣ Extract ECG Signal

Take the first channel of the ECG signal:

```python
ec_signal = record.p_signal[:, 0]
```

4️⃣ Convert to DataFrame

Store the ECG signal in a Pandas DataFrame. Using CSV is simple, human-readable, and compatible with many tools:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Sample': np.arange(len(ec_signal)),
    'ECG': ec_signal
})
print(df.head())
print(f"\nShape of data: {df.shape}")
```

5️⃣ Statistical Analysis

Use NumPy to calculate basic statistics:

```python
import numpy as np
print("Mean:", np.mean(ec_signal))
print("Std:", np.std(ec_signal))
print("Max:", np.max(ec_signal))
print("Min:", np.min(ec_signal))
```

6️⃣ Plot the Signal

Plot the ECG signal using Matplotlib and save it as an image:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(ec_signal, color='blue')
plt.title("ECG Signal from PhysioNet (Record 100)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("ecg_real_signal.png")
plt.show()
```

7️⃣ Save Data

Save the cleaned data as CSV:

```python
df.to_csv("ecg_real_data.csv", index=False)
```



📌 Notes

· CSV (Comma Separated Values) format is simple and widely supported.
· You can modify sampto parameter to include more samples.
· You can experiment with other records from PhysioNet.
· This project lays the foundation for:
  · Signal denoising / filtering
  · QRS peak detection
  · Feature extraction
  · Machine Learning applications

---

🎯 Project Goals

· Teach how to use Python and wfdb to work with real ECG signals.
· Prepare cleaned data for Signal Processing and Biomedical AI applications.
· Learn how to analyze, visualize, and store biomedical signal data.
· Provide a step-by-step educational guide for beginners in biomedical signal processing.

