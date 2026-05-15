import pandas as pd
import os

# =========================
# Load dataset
# =========================
file_path = "data/plant_health_data.csv"

if not os.path.exists(file_path):
    print("❌ الملف غير موجود:", file_path)
    exit()

df = pd.read_csv(file_path)

# =========================
# Required columns
# =========================
important_columns = [
    'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal', 'Plant_Health_Status'
]

# =========================
# Check missing columns
# =========================
missing = [col for col in important_columns if col not in df.columns]

if missing:
    print("❌ الأعمدة دي مش موجودة في الداتا:")
    print(missing)
    exit()

# =========================
# Clean data
# =========================
df = df[important_columns].copy()

# حذف القيم الفارغة
df.dropna(inplace=True)

# إزالة القيم الشاذة (اختياري)
df = df[(df['Soil_Moisture'] >= 0) & (df['Soil_Moisture'] <= 100)]
df = df[(df['Humidity'] >= 0) & (df['Humidity'] <= 100)]

# =========================
# Save cleaned dataset
# =========================
output_path = "data/cleaned_plant_data.csv"
df.to_csv(output_path, index=False)

print("✅ Cleaned dataset saved:", output_path)
print("\n📊 First 5 rows:")
print(df.head())

print("\n📈 Shape:", df.shape)