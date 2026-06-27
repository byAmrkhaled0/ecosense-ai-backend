import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# =========================
# Load dataset
# =========================
file_path = "data/plant_health_data.csv"

if not os.path.exists(file_path):
    print("❌ الملف غير موجود")
    exit()

df = pd.read_csv(file_path)

# =========================
# Basic Info
# =========================
print("🧾 Column Names:")
print(df.columns.tolist())

print("\n📊 First 5 rows:")
print(df.head())

print("\nℹ️ Data Info:")
df.info()

print("\n📈 Statistical Summary:")
print(df.describe())

# =========================
# Handle missing values
# =========================
if df.isnull().sum().sum() > 0:
    print("\n⚠️ في قيم ناقصة - بيتم حذفها")
    df = df.dropna()

# =========================
# Pairplot (⚠️ تقيل جدًا)
# =========================
try:
    sample_df = df.sample(min(300, len(df)))  # تقليل البيانات
    sns.pairplot(sample_df, hue="Plant_Health_Status")
    plt.suptitle("Pairplot Sample", y=1.02)
    plt.show()
except Exception as e:
    print("❌ Pairplot error:", e)

# =========================
# Histograms
# =========================
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

# =========================
# Correlation Heatmap
# =========================
plt.figure(figsize=(10, 8))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()