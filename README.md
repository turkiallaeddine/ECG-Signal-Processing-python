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
ط

```bash
pip install wfdb numpy pandas matplotlib
```

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

