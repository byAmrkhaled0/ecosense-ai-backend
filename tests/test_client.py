import requests

API_SENSOR = "http://127.0.0.1:5000/api/mobile_predict"

data = {
    "cropType": "Tomato",
    "temperature": 39,
    "humidity": 25,
    "soilMoisture": 18,
    "soilTemp": 34,
    "light": "Low"
}

print("🚀 Sending request...")

try:
    res = requests.post(API_SENSOR, json=data, timeout=10)
    response = res.json()
except Exception as e:
    print("❌ Error:", e)
    raise SystemExit

print("\n🌿 الحالة:", response.get("final_status", response.get("status")))
print("⚠️ مستوى الخطورة:", response.get("severity"))
print("🔔 Alert:", response.get("alert"))
print("\n🧠 الملخص:")
print("-", response.get("summary", ""))

print("\n💡 التوصيات:")
for r in response.get("recommendations", []):
    print("-", r)

print("\n📊 الثقة:")
for cls, val in response.get("confidence", {}).items():
    print(f"{cls}: {val*100:.1f}%")

notif = response.get("notification", {})

if notif.get("send"):
    try:
        from plyer import notification
        notification.notify(
            title=notif.get("title", "Plant Alert"),
            message=notif.get("message", ""),
            timeout=10
        )
        print("\n🚨 Notification sent!")
    except Exception as e:
        print("\n⚠️ Notification not available:", e)
else:
    print("\n✅ No notification needed.")