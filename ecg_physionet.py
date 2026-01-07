# ===============================
#  – ECG حقيقي من PhysioNet
# تحميل + تحليل + رسم + حفظ الصورة
# ===============================
# 1️⃣ تثبيت المكتبات
!pip install wfdb numpy pandas matplotlib --quiet
# -------------------------------
# 2️⃣ استيراد المكتبات
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -------------------------------
# 3️⃣ تحميل بيانات ECG من PhysioNet
# هنا نستخدم سجل 100 من قاعدة MIT-BIH Arrhythmia
record = wfdb.rdrecord('100', sampfrom=0, sampto=3000, pn_dir='mitdb')
# -------------------------------
# 4️⃣ استخراج الإشارة
# ECG الحقيقي في أول قناة
ec_signal = record.p_signal[:, 0]
# -------------------------------
# 5️⃣ تحويل البيانات إلى DataFrame
df = pd.DataFrame({
    'Sample': np.arange(len(ec_signal)),
    'ECG': ec_signal
})
print("أول 5 صفوف من البيانات:")
print(df.head())
print(f"\nشكل البيانات: {df.shape}")
# -------------------------------
# 6️⃣ التحليل الإحصائي باستخدام NumPy
print("\nالإحصاءات الأساسية للإشارة:")
print("المتوسط (Mean):", np.mean(ec_signal))
print("الانحراف المعياري (Std):", np.std(ec_signal))
print("أعلى قيمة (Max):", np.max(ec_signal))
print("أدنى قيمة (Min):", np.min(ec_signal))
# -------------------------------
# 7️⃣ رسم الإشارة وحفظ الصورة
plt.figure(figsize=(12, 4))
plt.plot(ec_signal, color='blue')
plt.title("ECG Signal from PhysioNet (Record 100)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
# حفظ الصورة
plt.savefig("ecg_real_signal.png")
print("\nتم حفظ الصورة باسم: ecg_real_signal.png")
# عرض الإشارة داخل Colab
plt.show()
# -------------------------------
# 8️⃣ حفظ البيانات النظيفة في CSV
df.to_csv("ecg_real_data.csv", index=False)
print("\nتم حفظ البيانات في ملف: ecg_real_data.csv")
