from flask import Flask, request, jsonify
from flask_cors import CORS
from image_analyzer import analyze_plant_image
import joblib
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =========================
# Load model files
# =========================
model = joblib.load("simple_model.pkl")
scaler = joblib.load("simple_scaler.pkl")
le_status = joblib.load("status_encoder.pkl")
le_light = joblib.load("light_encoder.pkl")
le_crop = joblib.load("crop_encoder.pkl")

ALLOWED_LIGHT = set(le_light.classes_)
ALLOWED_CROPS = set(le_crop.classes_)

HISTORY_FILE = "prediction_history.json"
RECOMMENDATIONS_FILE = "recommendations.json"

EXPECTED_FEATURES = [
    "temperature",
    "humidity",
    "soilMoisture",
    "soilTemp",
    "light",
    "cropType"
]

# =========================
# Load general recommendations
# =========================
if os.path.exists(RECOMMENDATIONS_FILE):
    with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
        solutions = json.load(f)
else:
    solutions = {
        "Healthy": "✅ النبات بحالة جيدة.",
        "Moderate Stress": "⚠️ يوجد إجهاد متوسط على النبات.",
        "High Stress": "🚨 النبات في حالة حرجة."
    }


# =========================
# Helpers
# =========================
def validate_common_fields(data):
    required_fields = [
        "cropType",
        "temperature",
        "humidity",
        "soilMoisture",
        "soilTemp",
        "light"
    ]
    missing = [
        field for field in required_fields
        if field not in data or str(data[field]).strip() == ""
    ]
    return missing


def build_summary(sensor_status, image_stress, final_status):
    if final_status == "Healthy":
        return "النبات يبدو بحالة جيدة بناءً على التحليل المتاح."
    elif final_status == "Moderate Stress":
        return "يوجد إجهاد متوسط على النبات، ويُنصح بالمتابعة واتخاذ إجراءات تصحيحية."
    else:
        return "النبات في حالة حرجة، ويجب التدخل السريع لتفادي تدهور الحالة."


def get_alert_info(final_status):
    if final_status == "Healthy":
        return {"alert": False, "severity": "low"}
    elif final_status == "Moderate Stress":
        return {"alert": True, "severity": "medium"}
    else:
        return {"alert": True, "severity": "high"}


def build_notification_payload(final_status, diagnosis, actions):
    if final_status == "Healthy":
        return {
            "send": False,
            "title": "✅ حالة النبات مستقرة",
            "message": "النبات بحالة جيدة ولا يحتاج إلى تدخل حاليًا.",
            "type": "info"
        }

    if final_status == "Moderate Stress":
        action_text = actions[0]["title"] if actions else "راجع القراءات الحالية"
        return {
            "send": True,
            "title": "⚠️ تنبيه: إجهاد متوسط",
            "message": f"{diagnosis['primary_issue']} - الإجراء المقترح: {action_text}",
            "type": "warning"
        }

    action_text = actions[0]["title"] if actions else "تدخل سريع مطلوب"
    return {
        "send": True,
        "title": "🚨 تحذير: حالة حرجة",
        "message": f"{diagnosis['primary_issue']} - الإجراء الفوري: {action_text}",
        "type": "critical"
    }


def prepare_model_input(data):
    crop = str(data["cropType"]).strip()
    light = str(data["light"]).strip()

    if crop not in ALLOWED_CROPS:
        raise ValueError(f"Unsupported cropType. Allowed: {sorted(list(ALLOWED_CROPS))}")

    if light not in ALLOWED_LIGHT:
        raise ValueError(f"Unsupported light value. Allowed: {sorted(list(ALLOWED_LIGHT))}")

    crop_encoded = le_crop.transform([crop])[0]
    light_encoded = le_light.transform([light])[0]

    df = pd.DataFrame([{
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
        "soilMoisture": float(data["soilMoisture"]),
        "soilTemp": float(data["soilTemp"]),
        "light": light_encoded,
        "cropType": crop_encoded
    }])

    df = df[EXPECTED_FEATURES]
    return df


def predict_sensor_status(data):
    df = prepare_model_input(data)
    x = scaler.transform(df)
    y = model.predict(x)
    proba = model.predict_proba(x)[0]
    status = le_status.inverse_transform(y)[0]

    confidence = {
        cls: round(float(p), 3)
        for cls, p in zip(le_status.classes_, proba)
    }

    return status, confidence


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            history = json.loads(content)
            return history if isinstance(history, list) else []
    except (json.JSONDecodeError, ValueError, OSError):
        return []


def save_prediction(record):
    try:
        history = load_history()
        history.append(record)

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print("⚠️ Failed to save history:", e)


def build_recommendations(data, status):
    recs = []

    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    soil_moisture = float(data["soilMoisture"])
    soil_temp = float(data["soilTemp"])
    light = str(data["light"]).strip()
    crop_type = str(data["cropType"]).strip()

    if status == "Healthy":
        recs.append(f"✅ النبات ({crop_type}) بحالة جيدة ومستقرة.")
    elif status == "Moderate Stress":
        recs.append(f"⚠️ النبات ({crop_type}) يعاني من إجهاد متوسط ويحتاج متابعة.")
    else:
        recs.append(f"🚨 النبات ({crop_type}) في حالة حرجة ويحتاج تدخلًا فوريًا.")

    if temperature < 20:
        recs.append("🌡️ درجة الحرارة منخفضة؛ يُفضل رفع الحرارة أو تقليل التعرض للبرودة.")
    elif 20 <= temperature <= 35:
        recs.append("🌡️ درجة الحرارة مناسبة.")
    else:
        recs.append("🌡️ درجة الحرارة مرتفعة؛ حسّن التهوية أو التبريد.")

    if soil_moisture < 30:
        recs.append("💧 رطوبة التربة منخفضة؛ زوّد الري تدريجيًا وراقب الاستجابة.")
    elif soil_moisture <= 70:
        recs.append("💧 رطوبة التربة مناسبة.")
    else:
        recs.append("💧 رطوبة التربة مرتفعة؛ قلل الري وافحص الصرف.")

    if humidity < 40:
        recs.append("🌫️ الرطوبة الجوية منخفضة؛ قد يزيد ذلك من الإجهاد المائي.")
    elif humidity <= 80:
        recs.append("🌫️ الرطوبة الجوية مناسبة.")
    else:
        recs.append("🌫️ الرطوبة الجوية مرتفعة؛ حسّن التهوية لتقليل فرص الأمراض.")

    if soil_temp < 18:
        recs.append("🌱 حرارة التربة منخفضة؛ قد تؤثر على امتصاص العناصر.")
    elif soil_temp <= 30:
        recs.append("🌱 حرارة التربة مناسبة.")
    else:
        recs.append("🌱 حرارة التربة مرتفعة؛ راقب منطقة الجذور جيدًا.")

    if light == "Low":
        recs.append("☀️ الإضاءة ضعيفة؛ زوّد الضوء أو مدة التعرض.")
    elif light == "Medium":
        recs.append("☀️ الإضاءة متوسطة؛ استمر في المتابعة.")
    elif light == "Sufficient":
        recs.append("☀️ الإضاءة مناسبة للنمو.")
    else:
        recs.append("☀️ قيمة الإضاءة غير معروفة؛ تأكد من إرسال قيمة صحيحة.")

    return recs


def analyze_risk_factors(data):
    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    soil_moisture = float(data["soilMoisture"])
    soil_temp = float(data["soilTemp"])
    light = str(data["light"]).strip()

    risks = []

    if soil_moisture < 30:
        risks.append({
            "code": "LOW_SOIL_MOISTURE",
            "label": "انخفاض رطوبة التربة",
            "severity": "high",
            "value": soil_moisture,
            "ideal_range": "30-70"
        })
    elif soil_moisture > 70:
        risks.append({
            "code": "HIGH_SOIL_MOISTURE",
            "label": "زيادة رطوبة التربة",
            "severity": "high",
            "value": soil_moisture,
            "ideal_range": "30-70"
        })

    if temperature < 20:
        risks.append({
            "code": "LOW_TEMPERATURE",
            "label": "انخفاض درجة الحرارة",
            "severity": "medium",
            "value": temperature,
            "ideal_range": "20-35"
        })
    elif temperature > 35:
        risks.append({
            "code": "HIGH_TEMPERATURE",
            "label": "ارتفاع درجة الحرارة",
            "severity": "medium",
            "value": temperature,
            "ideal_range": "20-35"
        })

    if humidity < 40:
        risks.append({
            "code": "LOW_HUMIDITY",
            "label": "انخفاض الرطوبة الجوية",
            "severity": "medium",
            "value": humidity,
            "ideal_range": "40-80"
        })
    elif humidity > 80:
        risks.append({
            "code": "HIGH_HUMIDITY",
            "label": "ارتفاع الرطوبة الجوية",
            "severity": "medium",
            "value": humidity,
            "ideal_range": "40-80"
        })

    if soil_temp < 18:
        risks.append({
            "code": "LOW_SOIL_TEMP",
            "label": "انخفاض حرارة التربة",
            "severity": "medium",
            "value": soil_temp,
            "ideal_range": "18-30"
        })
    elif soil_temp > 30:
        risks.append({
            "code": "HIGH_SOIL_TEMP",
            "label": "ارتفاع حرارة التربة",
            "severity": "medium",
            "value": soil_temp,
            "ideal_range": "18-30"
        })

    if light == "Low":
        risks.append({
            "code": "LOW_LIGHT",
            "label": "ضعف الإضاءة",
            "severity": "medium",
            "value": light,
            "ideal_range": "Sufficient"
        })
    elif light == "Medium":
        risks.append({
            "code": "MEDIUM_LIGHT",
            "label": "إضاءة متوسطة",
            "severity": "low",
            "value": light,
            "ideal_range": "Sufficient"
        })

    return risks


def build_diagnosis(data, status, risk_factors):
    if not risk_factors and status == "Healthy":
        return {
            "primary_issue": "لا توجد مشكلة رئيسية",
            "secondary_issue": "الظروف الحالية مستقرة",
            "explanation": "كل القراءات تقريبًا داخل الحدود المناسبة."
        }

    codes = [r["code"] for r in risk_factors]

    if "LOW_SOIL_MOISTURE" in codes and "HIGH_TEMPERATURE" in codes:
        return {
            "primary_issue": "إجهاد مائي وحراري",
            "secondary_issue": "احتمال ضعف في كفاءة التبريد أو الري",
            "explanation": "تم رصد انخفاض في رطوبة التربة مع ارتفاع في درجة الحرارة."
        }

    if "HIGH_SOIL_MOISTURE" in codes:
        return {
            "primary_issue": "زيادة مياه حول الجذور",
            "secondary_issue": "احتمال ضعف صرف أو زيادة ري",
            "explanation": "رطوبة التربة أعلى من النطاق المناسب."
        }

    if "LOW_LIGHT" in codes:
        return {
            "primary_issue": "ضعف الإضاءة",
            "secondary_issue": "تباطؤ محتمل في النمو",
            "explanation": "الإضاءة الحالية أقل من المستوى المناسب للنمو."
        }

    if status == "Moderate Stress":
        return {
            "primary_issue": "إجهاد متوسط",
            "secondary_issue": "خلل بيئي يحتاج تصحيح",
            "explanation": "تم رصد عامل أو أكثر خارج النطاق المثالي."
        }

    if status == "High Stress":
        return {
            "primary_issue": "إجهاد شديد",
            "secondary_issue": "عدة عوامل قد تؤثر على النبات",
            "explanation": "هناك أكثر من قراءة غير مناسبة وتحتاج تدخلًا سريعًا."
        }

    return {
        "primary_issue": "حالة مستقرة نسبيًا",
        "secondary_issue": "مطلوب استمرار المتابعة",
        "explanation": "لا توجد مؤشرات خطورة شديدة حاليًا."
    }


def build_actions(risk_factors):
    actions = []

    for risk in risk_factors:
        code = risk["code"]

        if code == "LOW_SOIL_MOISTURE":
            actions.append({
                "priority": 1,
                "code": "IRRIGATION_UP",
                "title": "زيادة الري تدريجيًا",
                "details": "رطوبة التربة أقل من النطاق المناسب."
            })

        elif code == "HIGH_SOIL_MOISTURE":
            actions.append({
                "priority": 1,
                "code": "IRRIGATION_DOWN",
                "title": "تقليل الري وفحص الصرف",
                "details": "رطوبة التربة مرتفعة وقد تسبب مشاكل للجذور."
            })

        elif code == "HIGH_TEMPERATURE":
            actions.append({
                "priority": 2,
                "code": "COOLING_ON",
                "title": "تحسين التهوية أو التبريد",
                "details": "درجة الحرارة أعلى من النطاق المناسب."
            })

        elif code == "LOW_TEMPERATURE":
            actions.append({
                "priority": 2,
                "code": "HEATING_CHECK",
                "title": "رفع الحرارة أو تقليل البرودة",
                "details": "درجة الحرارة أقل من النطاق المناسب."
            })

        elif code == "HIGH_HUMIDITY":
            actions.append({
                "priority": 2,
                "code": "AIRFLOW_UP",
                "title": "زيادة التهوية",
                "details": "الرطوبة العالية قد تزيد خطر الأمراض."
            })

        elif code == "LOW_HUMIDITY":
            actions.append({
                "priority": 3,
                "code": "HUMIDITY_UP",
                "title": "رفع الرطوبة عند الحاجة",
                "details": "الرطوبة منخفضة وقد تزيد الإجهاد."
            })

        elif code == "LOW_LIGHT":
            actions.append({
                "priority": 3,
                "code": "LIGHT_UP",
                "title": "زيادة الإضاءة",
                "details": "الإضاءة الحالية غير كافية للنمو."
            })

        elif code == "HIGH_SOIL_TEMP":
            actions.append({
                "priority": 2,
                "code": "ROOTZONE_COOL",
                "title": "خفض حرارة منطقة الجذور",
                "details": "حرارة التربة مرتفعة."
            })

        elif code == "LOW_SOIL_TEMP":
            actions.append({
                "priority": 2,
                "code": "ROOTZONE_WARM",
                "title": "رفع حرارة منطقة الجذور",
                "details": "حرارة التربة منخفضة."
            })

    actions = sorted(actions, key=lambda x: x["priority"])
    return actions


def build_monitoring(risk_factors, status):
    monitoring = []
    codes = [r["code"] for r in risk_factors]

    if "LOW_SOIL_MOISTURE" in codes or "HIGH_SOIL_MOISTURE" in codes:
        monitoring.append("إعادة فحص رطوبة التربة بعد التعديل في الري.")

    if "HIGH_TEMPERATURE" in codes or "LOW_TEMPERATURE" in codes:
        monitoring.append("مراقبة درجة الحرارة خلال الساعات القادمة.")

    if "HIGH_HUMIDITY" in codes:
        monitoring.append("متابعة ظهور أي أعراض مرضية أو فطرية.")

    if "LOW_LIGHT" in codes:
        monitoring.append("متابعة تحسن النمو بعد تعديل الإضاءة.")

    if status == "Healthy" and not monitoring:
        monitoring.append("استمر في المراقبة الدورية للحفاظ على استقرار الحالة.")

    if not monitoring:
        monitoring.append("راجع القراءات القادمة وتابع أي تغيرات جديدة في الحالة.")

    return monitoring


def build_backend_flags(risk_factors, status):
    codes = [r["code"] for r in risk_factors]

    return {
        "needs_irrigation": "LOW_SOIL_MOISTURE" in codes,
        "needs_drainage_check": "HIGH_SOIL_MOISTURE" in codes,
        "needs_cooling": "HIGH_TEMPERATURE" in codes,
        "needs_heating": "LOW_TEMPERATURE" in codes,
        "needs_light_adjustment": ("LOW_LIGHT" in codes or "MEDIUM_LIGHT" in codes),
        "needs_humidity_adjustment": ("LOW_HUMIDITY" in codes or "HIGH_HUMIDITY" in codes),
        "needs_urgent_attention": status == "High Stress"
    }


def build_advanced_response(data, status, confidence, final_status=None, image_result=None):
    used_status = final_status or status
    crop_type = str(data.get("cropType", "Unknown")).strip()

    risk_factors = analyze_risk_factors(data)
    diagnosis = build_diagnosis(data, used_status, risk_factors)
    actions = build_actions(risk_factors)
    monitoring = build_monitoring(risk_factors, used_status)
    backend_flags = build_backend_flags(risk_factors, used_status)
    recommendations = build_recommendations(data, used_status)
    alert_info = get_alert_info(used_status)
    notification = build_notification_payload(used_status, diagnosis, actions)

    summary = build_summary(
        status,
        image_result.get("image_stress") if image_result else "Not used",
        used_status
    )

    response = {
        "status": status,
        "final_status": used_status,

        # اسم الزرعة راجع في السينسورات
        "cropType": crop_type,
        "cropName": crop_type,
        "plant_name": crop_type,

        "confidence": confidence,
        "alert": alert_info["alert"],
        "severity": alert_info["severity"],
        "summary": summary,
        "general_recommendation": solutions.get(used_status, ""),
        "diagnosis": diagnosis,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "actions": actions,
        "monitoring": monitoring,
        "backend_flags": backend_flags,
        "notification": notification,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if image_result is not None:
        response["image_analysis"] = image_result
        response["sensor_status"] = status

    return response


def build_image_only_response(image_result, crop_type="Unknown"):
    image_stress = image_result.get("image_stress", "Healthy")
    yellow = float(image_result.get("yellow_ratio", 0))
    brown = float(image_result.get("brown_ratio", 0))
    score = float(image_result.get("health_score", 0.0))

    health_score = max(0.0, min(1.0, score))
    severity_score = round(1.0 - health_score, 3)

    if image_stress == "Healthy":
        confidence = round(health_score, 3)
    else:
        confidence = severity_score

    crop_type = str(crop_type).strip() or "Unknown"
    crop = crop_type.lower()

    if image_stress == "Healthy":
        return {
            "status": "Healthy",

            # اسم الزرعة راجع في الكاميرا
            "cropType": crop_type,
            "cropName": crop_type,
            "plant_name": crop_type,

            "disease_name": "No Disease Detected",
            "confidence": confidence,
            "health_score": round(health_score, 3),
            "severity_score": severity_score,
            "recommendations": [
                "استمر في برنامج الري والتسميد الحالي",
                "المتابعة الدورية للنبات",
                "مراقبة أي تغير جديد في لون الأوراق"
            ],
            "analysis": image_result
        }

    disease_name = "General Plant Stress"
    recommendations = [
        "مراجعة الري",
        "تحسين التهوية",
        "فحص النبات خلال 2-3 أيام"
    ]
    status = "Detected"

    # Tomato
    if crop == "tomato":
        if yellow > 0.25 and brown < 0.05:
            disease_name = "Nutrient Deficiency (Nitrogen or Iron) - Tomato"
            recommendations = [
                "إضافة سماد نيتروجيني أو عناصر صغرى مثل الحديد",
                "مراجعة pH التربة أو المحلول المغذي",
                "تحسين برنامج التسميد",
                "متابعة الأوراق الحديثة والقديمة"
            ]
            status = "Detected"

        elif brown > 0.08 and yellow > 0.10:
            disease_name = "Early Blight / Leaf Spot (Tomato)"
            recommendations = [
                "إزالة الأوراق المصابة فورًا",
                "رش مبيد فطري مناسب مثل مانكوزيب أو كلوروثالونيل حسب التوصية المحلية",
                "تقليل الرطوبة وتحسين التهوية",
                "تجنب الري على الأوراق",
                "مراقبة انتشار البقع خلال 3 أيام"
            ]
            status = "Infected"

        elif brown > 0.10:
            disease_name = "Tomato Fungal Leaf Infection"
            recommendations = [
                "إزالة الأوراق الأكثر إصابة",
                "استخدام مبيد فطري مناسب",
                "تحسين التهوية وتقليل الرطوبة",
                "فحص باقي النباتات المجاورة"
            ]
            status = "Infected"

        else:
            disease_name = "General Tomato Stress"
            recommendations = [
                "مراجعة الري والتسميد",
                "تحسين التهوية",
                "فحص النبات خلال يومين",
                "مراقبة أي زيادة في الاصفرار أو البقع"
            ]
            status = "Detected"

    # Cucumber
    elif crop == "cucumber":
        if yellow > 0.22 and brown < 0.05:
            disease_name = "Nutrient Deficiency or Downy Mildew Suspicion (Cucumber)"
            recommendations = [
                "فحص السطح السفلي للأوراق",
                "تحسين التهوية وتقليل الرطوبة",
                "مراجعة التسميد خاصة النيتروجين والمغنيسيوم",
                "استخدام معاملة فطرية مناسبة إذا زادت الأعراض"
            ]
            status = "Detected"

        elif brown > 0.08:
            disease_name = "Leaf Spot / Fungal Stress (Cucumber)"
            recommendations = [
                "إزالة الأوراق المصابة",
                "تقليل الرطوبة حول النبات",
                "تحسين حركة الهواء",
                "رش مبيد فطري مناسب حسب التوصية المحلية"
            ]
            status = "Infected"

        else:
            disease_name = "General Cucumber Stress"
            recommendations = [
                "مراجعة الري",
                "تحسين التهوية",
                "فحص الأوراق يوميًا",
                "مراجعة التسميد"
            ]
            status = "Detected"

    # Pepper
    elif crop == "pepper":
        if yellow > 0.20 and brown < 0.05:
            disease_name = "Nutrient Deficiency (Pepper)"
            recommendations = [
                "مراجعة التسميد خاصة النيتروجين والحديد والمغنيسيوم",
                "فحص pH",
                "متابعة تطور الاصفرار في الأوراق الحديثة"
            ]
            status = "Detected"

        elif brown > 0.08:
            disease_name = "Bacterial/Fungal Leaf Spot Suspicion (Pepper)"
            recommendations = [
                "إزالة الأوراق المصابة",
                "تقليل البلل على الأوراق",
                "تحسين التهوية",
                "استخدام معاملة مناسبة حسب التشخيص الحقلي"
            ]
            status = "Infected"

        else:
            disease_name = "General Pepper Stress"
            recommendations = [
                "مراجعة الري والتسميد",
                "تحسين الظروف البيئية",
                "مراقبة تطور الأعراض"
            ]
            status = "Detected"

    # Default
    else:
        if yellow > 0.25 and brown < 0.05:
            disease_name = "Nutrient Deficiency"
            recommendations = [
                "مراجعة برنامج التسميد",
                "فحص العناصر الصغرى",
                "مراجعة pH"
            ]
            status = "Detected"

        elif brown > 0.08:
            disease_name = "Fungal Infection / Leaf Spot"
            recommendations = [
                "إزالة الأجزاء المصابة",
                "تحسين التهوية",
                "استخدام مبيد فطري مناسب حسب التوصية المحلية"
            ]
            status = "Infected"

        elif yellow > 0.15 and brown > 0.05:
            disease_name = "Combined Stress (Disease + Nutrient Issue)"
            recommendations = [
                "إزالة الأجزاء المصابة",
                "تحسين التسميد",
                "مراجعة الري والتهوية"
            ]
            status = "Detected"

        else:
            disease_name = "General Plant Stress"
            recommendations = [
                "مراجعة الري",
                "تحسين التهوية",
                "فحص النبات خلال 2-3 أيام"
            ]
            status = "Detected"

    return {
        "status": status,

        # اسم الزرعة راجع في الكاميرا
        "cropType": crop_type,
        "cropName": crop_type,
        "plant_name": crop_type,

        "disease_name": disease_name,
        "confidence": confidence,
        "health_score": round(health_score, 3),
        "severity_score": severity_score,
        "recommendations": recommendations,
        "analysis": image_result
    }


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart Plant Health API is running ✅"
    }), 200


@app.route("/api/simple_predict", methods=["POST"])
def simple_predict():
    try:
        if not request.is_json:
            return jsonify({
                "error": "For /api/simple_predict use raw JSON body with Content-Type: application/json"
            }), 400

        data = request.get_json(silent=True) or {}
        missing = validate_common_fields(data)
        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing": missing
            }), 400

        status, confidence = predict_sensor_status(data)

        response_data = build_advanced_response(
            data=data,
            status=status,
            confidence=confidence
        )

        save_prediction({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "request_type": "simple_predict",
            "input": data,
            "result": response_data
        })

        return jsonify(response_data), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mobile_predict", methods=["POST"])
def mobile_predict():
    try:
        if not request.is_json:
            return jsonify({
                "error": "Use JSON body"
            }), 400

        data = request.get_json(silent=True) or {}
        missing = validate_common_fields(data)
        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing": missing
            }), 400

        status, confidence = predict_sensor_status(data)

        response_data = build_advanced_response(
            data=data,
            status=status,
            confidence=confidence
        )

        save_prediction({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "request_type": "mobile_predict",
            "input": data,
            "result": response_data
        })

        return jsonify(response_data), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_with_image", methods=["POST"])
def predict_with_image():
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "For /api/predict_with_image use Body -> form-data with file field named 'file'"
            }), 400

        image_file = request.files["file"]

        if image_file is None or image_file.filename.strip() == "":
            return jsonify({"error": "No image selected"}), 400

        crop_type = (
            request.form.get("cropType")
            or request.form.get("cropName")
            or request.form.get("plantName")
            or request.form.get("plant_name")
            or "Unknown"
        ).strip()

        image_result = analyze_plant_image(image_file)

        if "error" in image_result:
            return jsonify({"error": image_result["error"]}), 400

        response_data = build_image_only_response(image_result, crop_type)

        save_prediction({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "request_type": "predict_with_image",
            "image_filename": image_file.filename,
            "crop_type": crop_type,
            "image_analysis": image_result,
            "result": response_data
        })

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        history = load_history()

        return jsonify({
            "count": len(history),
            "history": history
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)