import requests
import os
import json

API_URL = "http://127.0.0.1:5000/api/predict_with_image"

IMAGE_PATH = "plant.jpg"
data = {
    "cropType": "Tomato",
    "temperature": "25",
    "humidity": "60",
    "soilMoisture": "45",
    "soilTemp": "24",
    "light": "Sufficient"
}
print("🚀 Sending request...")

if not os.path.exists(IMAGE_PATH):
    print(f" الصورة مش موجودة: {IMAGE_PATH}")
    print("حط صورة في نفس فولدر المشروع وسمّيها plant.jpg")
    raise SystemExit

try:
    with open(IMAGE_PATH, "rb") as f:
        files = {
            "file": ("plant.jpg", f, "image/jpeg")  # مهم جدًا: الاسم لازم file
        }

        response = requests.post(
            API_URL,
            data=data,
            files=files,
            timeout=30
        )

    print("Status Code:", response.status_code)

    try:
        result = response.json()
    except Exception:
        print(" Response is not JSON:")
        print(response.text)
        raise SystemExit

    if response.status_code != 200:
        print("\n Server Error Response:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit

    print("\n🌿 Final Status:", result.get("final_status", result.get("status")))
    print(" Sensor Status:", result.get("sensor_status"))
    print(" Image Status:", result.get("image_status"))
    print(" Severity:", result.get("severity"))
    print(" Alert:", result.get("alert"))

    print("\n Diagnosis:")
    diagnosis = result.get("diagnosis", {})
    print("-", diagnosis.get("primary_issue"))
    print("-", diagnosis.get("secondary_issue"))
    print("-", diagnosis.get("explanation"))

    print("\n💡 Recommendations:")
    for rec in result.get("recommendations", []):
        print("-", rec)

    print("\n📸 Full Response:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

except Exception as e:
    print("❌ Error:", e)