import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# =========================
# Load dataset
# =========================
df = pd.read_csv("multi_crop_plant_data.csv")

# =========================
# Encode categorical columns
# =========================
le_crop = LabelEncoder()
df["cropType"] = le_crop.fit_transform(df["cropType"])

le_light = LabelEncoder()
df["light"] = le_light.fit_transform(df["light"])

le_status = LabelEncoder()
df["status_encoded"] = le_status.fit_transform(df["status"])

print("Status encoding:", dict(zip(le_status.classes_, le_status.transform(le_status.classes_))))
print("Crop encoding:", dict(zip(le_crop.classes_, le_crop.transform(le_crop.classes_))))
print("Light encoding:", dict(zip(le_light.classes_, le_light.transform(le_light.classes_))))

# =========================
# Features and target
# =========================
features = [
    "temperature",
    "humidity",
    "soilMoisture",
    "soilTemp",
    "light",
    "cropType"
]

X = df[features]
y = df["status_encoded"]

# =========================
# Scale features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Train model
# =========================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# Evaluate model
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Random Forest Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le_status.classes_))

# =========================
# Cross Validation
# =========================
class_counts = pd.Series(y).value_counts()
min_class_count = class_counts.min()

if min_class_count >= 2:
    cv_value = min(5, min_class_count)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv_value)
    print(f"\nCross Validation Scores (cv={cv_value}):", cv_scores)
    print("Mean CV Score:", round(cv_scores.mean(), 3))
else:
    print("\n⚠️ Cross Validation skipped: not enough samples in at least one class.")

# =========================
# Feature Importance
# =========================
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

plt.figure(figsize=(8, 5))
plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_status.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================
# Save model files
# =========================
joblib.dump(model, "simple_model.pkl")
joblib.dump(scaler, "simple_scaler.pkl")
joblib.dump(le_status, "status_encoder.pkl")
joblib.dump(le_light, "light_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")

print("\n🎯 Model training complete!")
print("Files saved:")
print("  • simple_model.pkl")
print("  • simple_scaler.pkl")
print("  • status_encoder.pkl")
print("  • light_encoder.pkl")
print("  • crop_encoder.pkl")