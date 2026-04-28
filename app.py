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

STATUS_PRIORITY = {
    "Healthy": 0,
    "Moderate Stress": 1,
    "High Stress": 2
}

# =========================
# Load general recommendations
# =========================
if os.path.exists(RECOMMENDATIONS_FILE):
    with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
        solutions = json.load(f)
else:
    solutions = {
        "Healthy": "✅ النبات بحالة جيدة. حافظ على برنامج الري والتسميد الحالي واستمر في المتابعة الدورية.",
        "Moderate Stress": "⚠️ يوجد إجهاد متوسط على النبات. راجع الري ودرجة الحرارة والإضاءة وقم بضبطهم لتحسين الحالة.",
        "High Stress": "🚨 النبات في حالة حرجة! يجب التدخل فورًا بفحص رطوبة التربة والعناصر الغذائية ودرجة الحرارة وتحسين الظروف البيئية."
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


def normalize_sensor_data(data):
    """Convert form/json values to the exact structure needed by the model."""
    return {
        "cropType": str(data["cropType"]).strip(),
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
        "soilMoisture": float(data["soilMoisture"]),
        "soilTemp": float(data["soilTemp"]),
        "light": str(data["light"]).strip()
    }


def has_all_sensor_fields(data):
    return len(validate_common_fields(data)) == 0


def choose_worst_status(*statuses):
    valid_statuses = [s for s in statuses if s in STATUS_PRIORITY]
    if not valid_statuses:
        return "Healthy"
    return max(valid_statuses, key=lambda s: STATUS_PRIORITY[s])


def build_summary(sensor_status, image_stress, final_status, safety_flags=None):
    safety_flags = safety_flags or []

    if safety_flags and final_status == "High Stress":
        return "النبات في حالة حرجة بسبب قراءات خطرة أو أعراض بصرية واضحة، ويجب التدخل السريع."

    if image_stress and image_stress != "Not used" and image_stress != sensor_status:
        return f"نتيجة الحساسات تشير إلى {sensor_status}، وتحليل الصورة يشير إلى {image_stress}، لذلك تم اعتماد الحالة الأعلى خطورة."

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


def build_recommendations(data, status, image_result=None, safety_flags=None):
    recs = []

    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    soil_moisture = float(data["soilMoisture"])
    soil_temp = float(data["soilTemp"])
    light = str(data["light"]).strip()
    crop_type = str(data["cropType"]).strip()
    safety_flags = safety_flags or []

    if status == "Healthy":
        recs.append(f"✅ النبات ({crop_type}) بحالة جيدة ومستقرة.")
    elif status == "Moderate Stress":
        recs.append(f"⚠️ النبات ({crop_type}) يعاني من إجهاد متوسط ويحتاج متابعة.")
    else:
        recs.append(f"🚨 النبات ({crop_type}) في حالة حرجة ويحتاج تدخلًا فوريًا.")

    if "EXTREME_CONDITION_OVERRIDE" in safety_flags:
        recs.append("🚨 تم رفع الحالة تلقائيًا بسبب وجود قراءات خطرة جدًا حتى لو توقع الموديل غير ذلك.")

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

    if image_result:
        for item in image_result.get("image_recommendations", []):
            if item not in recs:
                recs.append("📸 " + item)

    return recs


def analyze_risk_factors(data):
    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    soil_moisture = float(data["soilMoisture"])
    soil_temp = float(data["soilTemp"])
    light = str(data["light"]).strip()

    risks = []

    if soil_moisture <= 10:
        risks.append({
            "code": "CRITICAL_LOW_SOIL_MOISTURE",
            "label": "انخفاض حاد جدًا في رطوبة التربة",
            "severity": "critical",
            "value": soil_moisture,
            "ideal_range": "30-70"
        })
    elif soil_moisture < 30:
        risks.append({
            "code": "LOW_SOIL_MOISTURE",
            "label": "انخفاض رطوبة التربة",
            "severity": "high",
            "value": soil_moisture,
            "ideal_range": "30-70"
        })
    elif soil_moisture >= 90:
        risks.append({
            "code": "CRITICAL_HIGH_SOIL_MOISTURE",
            "label": "ارتفاع حاد جدًا في رطوبة التربة",
            "severity": "critical",
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

    if temperature >= 45:
        risks.append({
            "code": "CRITICAL_HIGH_TEMPERATURE",
            "label": "ارتفاع حاد جدًا في درجة الحرارة",
            "severity": "critical",
            "value": temperature,
            "ideal_range": "20-35"
        })
    elif temperature < 20:
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

    if humidity <= 15:
        risks.append({
            "code": "CRITICAL_LOW_HUMIDITY",
            "label": "انخفاض حاد جدًا في الرطوبة الجوية",
            "severity": "high",
            "value": humidity,
            "ideal_range": "40-80"
        })
    elif humidity < 40:
        risks.append({
            "code": "LOW_HUMIDITY",
            "label": "انخفاض الرطوبة الجوية",
            "severity": "medium",
            "value": humidity,
            "ideal_range": "40-80"
        })
    elif humidity >= 90:
        risks.append({
            "code": "CRITICAL_HIGH_HUMIDITY",
            "label": "ارتفاع حاد جدًا في الرطوبة الجوية",
            "severity": "high",
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

    if soil_temp >= 40:
        risks.append({
            "code": "CRITICAL_HIGH_SOIL_TEMP",
            "label": "ارتفاع حاد جدًا في حرارة التربة",
            "severity": "critical",
            "value": soil_temp,
            "ideal_range": "18-30"
        })
    elif soil_temp < 18:
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


def apply_safety_layer(sensor_status, risk_factors, image_result=None):
    """
    Safety override prevents impossible outputs like:
    temperature=55 + soilMoisture=5 but final_status=Healthy.
    """
    safety_flags = []
    final_status = sensor_status

    codes = [r["code"] for r in risk_factors]
    severities = [r["severity"] for r in risk_factors]
    critical_count = severities.count("critical")
    high_or_critical_count = sum(1 for s in severities if s in ("high", "critical"))

    extreme_codes = {
        "CRITICAL_LOW_SOIL_MOISTURE",
        "CRITICAL_HIGH_SOIL_MOISTURE",
        "CRITICAL_HIGH_TEMPERATURE",
        "CRITICAL_HIGH_SOIL_TEMP",
    }

    if any(code in codes for code in extreme_codes):
        final_status = "High Stress"
        safety_flags.append("EXTREME_CONDITION_OVERRIDE")

    if critical_count >= 2 or high_or_critical_count >= 3:
        final_status = "High Stress"
        safety_flags.append("MULTIPLE_RISK_FACTORS_OVERRIDE")

    if high_or_critical_count >= 1 and sensor_status == "Healthy":
        final_status = choose_worst_status(final_status, "Moderate Stress")
        safety_flags.append("HEALTHY_MODEL_CORRECTED_TO_MODERATE")

    if image_result:
        image_stress = image_result.get("image_stress")
        visual_problem = image_result.get("visual_problem")
        dark_spot_ratio = float(image_result.get("dark_spot_ratio", 0))
        damaged_ratio = float(image_result.get("damaged_ratio", 0))

        if image_stress in STATUS_PRIORITY:
            final_status = choose_worst_status(final_status, image_stress)

        if visual_problem == "Leaf Spot / Fungal Suspicion" and dark_spot_ratio >= 0.08:
            final_status = choose_worst_status(final_status, "High Stress")
            safety_flags.append("IMAGE_DISEASE_OVERRIDE")

        if damaged_ratio >= 0.38:
            final_status = choose_worst_status(final_status, "High Stress")
            safety_flags.append("HIGH_VISUAL_DAMAGE_OVERRIDE")

    return final_status, sorted(list(set(safety_flags)))


def build_diagnosis(data, status, risk_factors, image_result=None, safety_flags=None):
    safety_flags = safety_flags or []
    codes = [r["code"] for r in risk_factors]

    if image_result:
        visual_problem = image_result.get("visual_problem")
        visual_problem_ar = image_result.get("visual_problem_ar", "")
        visual_explanation = image_result.get("visual_explanation", "")

        if visual_problem == "Leaf Spot / Fungal Suspicion":
            return {
                "primary_issue": "اشتباه تبقع أوراق أو إصابة فطرية",
                "secondary_issue": "ظهور بقع داكنة مع اصفرار في الورقة",
                "explanation": visual_explanation,
                "visual_problem": visual_problem,
                "visual_problem_ar": visual_problem_ar,
                "safety_flags": safety_flags
            }

        if visual_problem == "Chlorosis / Nutrient Deficiency Suspicion":
            return {
                "primary_issue": "اشتباه اصفرار ناتج عن نقص عناصر أو ضعف امتصاص",
                "secondary_issue": "يلزم ربط الصورة ببيانات التربة والتسميد",
                "explanation": visual_explanation,
                "visual_problem": visual_problem,
                "visual_problem_ar": visual_problem_ar,
                "safety_flags": safety_flags
            }

        if visual_problem == "Necrosis / Severe Leaf Damage":
            return {
                "primary_issue": "تلف واضح في نسيج الورقة",
                "secondary_issue": "قد يكون مرتبطًا بإجهاد شديد أو مرض ورقي",
                "explanation": visual_explanation,
                "visual_problem": visual_problem,
                "visual_problem_ar": visual_problem_ar,
                "safety_flags": safety_flags
            }

    if "CRITICAL_LOW_SOIL_MOISTURE" in codes and "CRITICAL_HIGH_TEMPERATURE" in codes:
        return {
            "primary_issue": "إجهاد مائي وحراري حاد",
            "secondary_issue": "خطر ذبول أو تلف سريع إذا لم يتم التدخل",
            "explanation": "رطوبة التربة منخفضة جدًا مع درجة حرارة مرتفعة جدًا، لذلك تم رفع الحالة إلى High Stress.",
            "safety_flags": safety_flags
        }

    if ("LOW_SOIL_MOISTURE" in codes or "CRITICAL_LOW_SOIL_MOISTURE" in codes) and (
        "HIGH_TEMPERATURE" in codes or "CRITICAL_HIGH_TEMPERATURE" in codes
    ):
        return {
            "primary_issue": "إجهاد مائي وحراري",
            "secondary_issue": "احتمال ضعف في كفاءة التبريد أو الري",
            "explanation": "تم رصد انخفاض في رطوبة التربة مع ارتفاع في درجة الحرارة.",
            "safety_flags": safety_flags
        }

    if "HIGH_SOIL_MOISTURE" in codes or "CRITICAL_HIGH_SOIL_MOISTURE" in codes:
        return {
            "primary_issue": "زيادة مياه حول الجذور",
            "secondary_issue": "احتمال ضعف صرف أو زيادة ري",
            "explanation": "رطوبة التربة أعلى من النطاق المناسب وقد تؤثر على الجذور.",
            "safety_flags": safety_flags
        }

    if "LOW_LIGHT" in codes:
        return {
            "primary_issue": "ضعف الإضاءة",
            "secondary_issue": "تباطؤ محتمل في النمو",
            "explanation": "الإضاءة الحالية أقل من المستوى المناسب للنمو.",
            "safety_flags": safety_flags
        }

    if not risk_factors and status == "Healthy":
        return {
            "primary_issue": "لا توجد مشكلة رئيسية",
            "secondary_issue": "الظروف الحالية مستقرة",
            "explanation": "كل القراءات تقريبًا داخل الحدود المناسبة.",
            "safety_flags": safety_flags
        }

    if status == "Moderate Stress":
        return {
            "primary_issue": "إجهاد متوسط",
            "secondary_issue": "خلل بيئي يحتاج تصحيح",
            "explanation": "تم رصد عامل أو أكثر خارج النطاق المثالي.",
            "safety_flags": safety_flags
        }

    if status == "High Stress":
        return {
            "primary_issue": "إجهاد شديد",
            "secondary_issue": "عدة عوامل قد تؤثر على النبات",
            "explanation": "هناك أكثر من قراءة غير مناسبة أو أعراض بصرية تحتاج تدخلًا سريعًا.",
            "safety_flags": safety_flags
        }

    return {
        "primary_issue": "حالة مستقرة نسبيًا",
        "secondary_issue": "مطلوب استمرار المتابعة",
        "explanation": "لا توجد مؤشرات خطورة شديدة حاليًا.",
        "safety_flags": safety_flags
    }


def build_actions(risk_factors, image_result=None):
    actions = []

    if image_result:
        visual_problem = image_result.get("visual_problem")

        if visual_problem == "Leaf Spot / Fungal Suspicion":
            actions.append({
                "priority": 1,
                "code": "LEAF_SPOT_CONTROL",
                "title": "عزل وإزالة الأوراق المصابة وتحسين التهوية",
                "details": "تحليل الصورة رصد بقع داكنة مع اصفرار، وهذا يستدعي تقليل انتشار الإصابة."
            })

        elif visual_problem == "Chlorosis / Nutrient Deficiency Suspicion":
            actions.append({
                "priority": 2,
                "code": "NUTRIENT_CHECK",
                "title": "مراجعة برنامج التسميد و pH",
                "details": "الاصفرار قد يرتبط بنقص عناصر أو ضعف امتصاص."
            })

        elif visual_problem == "Necrosis / Severe Leaf Damage":
            actions.append({
                "priority": 1,
                "code": "REMOVE_DAMAGED_TISSUE",
                "title": "إزالة الأجزاء شديدة التلف",
                "details": "وجود مناطق ميتة أو داكنة بنسبة مرتفعة يحتاج متابعة سريعة."
            })

    for risk in risk_factors:
        code = risk["code"]

        if code in ("LOW_SOIL_MOISTURE", "CRITICAL_LOW_SOIL_MOISTURE"):
            actions.append({
                "priority": 1,
                "code": "IRRIGATION_UP",
                "title": "زيادة الري تدريجيًا",
                "details": "رطوبة التربة أقل من النطاق المناسب."
            })

        elif code in ("HIGH_SOIL_MOISTURE", "CRITICAL_HIGH_SOIL_MOISTURE"):
            actions.append({
                "priority": 1,
                "code": "IRRIGATION_DOWN",
                "title": "تقليل الري وفحص الصرف",
                "details": "رطوبة التربة مرتفعة وقد تسبب مشاكل للجذور."
            })

        elif code in ("HIGH_TEMPERATURE", "CRITICAL_HIGH_TEMPERATURE"):
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

        elif code in ("HIGH_HUMIDITY", "CRITICAL_HIGH_HUMIDITY"):
            actions.append({
                "priority": 2,
                "code": "AIRFLOW_UP",
                "title": "زيادة التهوية",
                "details": "الرطوبة العالية قد تزيد خطر الأمراض."
            })

        elif code in ("LOW_HUMIDITY", "CRITICAL_LOW_HUMIDITY"):
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

        elif code in ("HIGH_SOIL_TEMP", "CRITICAL_HIGH_SOIL_TEMP"):
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

    # Remove duplicate action codes
    unique = {}
    for action in actions:
        unique[action["code"]] = action

    actions = sorted(unique.values(), key=lambda x: x["priority"])
    return actions


def build_monitoring(risk_factors, status, image_result=None):
    monitoring = []
    codes = [r["code"] for r in risk_factors]

    if image_result and image_result.get("image_stress") != "Healthy":
        monitoring.append("التقط صورة جديدة لنفس الورقة أو النبات بعد 24-48 ساعة للمقارنة.")

    if image_result and image_result.get("visual_problem") == "Leaf Spot / Fungal Suspicion":
        monitoring.append("راقب هل البقع الداكنة تزيد أو تنتقل لأوراق جديدة.")

    if any(code in codes for code in [
        "LOW_SOIL_MOISTURE",
        "HIGH_SOIL_MOISTURE",
        "CRITICAL_LOW_SOIL_MOISTURE",
        "CRITICAL_HIGH_SOIL_MOISTURE"
    ]):
        monitoring.append("إعادة فحص رطوبة التربة بعد التعديل في الري.")

    if any(code in codes for code in [
        "HIGH_TEMPERATURE",
        "LOW_TEMPERATURE",
        "CRITICAL_HIGH_TEMPERATURE"
    ]):
        monitoring.append("مراقبة درجة الحرارة خلال الساعات القادمة.")

    if any(code in codes for code in ["HIGH_HUMIDITY", "CRITICAL_HIGH_HUMIDITY"]):
        monitoring.append("متابعة ظهور أي أعراض مرضية أو فطرية.")

    if "LOW_LIGHT" in codes:
        monitoring.append("متابعة تحسن النمو بعد تعديل الإضاءة.")

    if status == "Healthy" and not monitoring:
        monitoring.append("استمر في المراقبة الدورية للحفاظ على استقرار الحالة.")

    if not monitoring:
        monitoring.append("راجع القراءات القادمة وتابع أي تغيرات جديدة في الحالة.")

    return monitoring


def build_backend_flags(risk_factors, status, image_result=None, safety_flags=None):
    codes = [r["code"] for r in risk_factors]
    safety_flags = safety_flags or []

    return {
        "needs_irrigation": ("LOW_SOIL_MOISTURE" in codes or "CRITICAL_LOW_SOIL_MOISTURE" in codes),
        "needs_drainage_check": ("HIGH_SOIL_MOISTURE" in codes or "CRITICAL_HIGH_SOIL_MOISTURE" in codes),
        "needs_cooling": ("HIGH_TEMPERATURE" in codes or "CRITICAL_HIGH_TEMPERATURE" in codes),
        "needs_heating": "LOW_TEMPERATURE" in codes,
        "needs_light_adjustment": ("LOW_LIGHT" in codes or "MEDIUM_LIGHT" in codes),
        "needs_humidity_adjustment": any(code in codes for code in [
            "LOW_HUMIDITY",
            "HIGH_HUMIDITY",
            "CRITICAL_LOW_HUMIDITY",
            "CRITICAL_HIGH_HUMIDITY"
        ]),
        "needs_leaf_disease_check": bool(
            image_result and image_result.get("visual_problem") == "Leaf Spot / Fungal Suspicion"
        ),
        "needs_nutrient_check": bool(
            image_result and image_result.get("visual_problem") == "Chlorosis / Nutrient Deficiency Suspicion"
        ),
        "needs_urgent_attention": status == "High Stress",
        "safety_override_applied": len(safety_flags) > 0,
        "safety_flags": safety_flags
    }


def build_advanced_response(data, status, confidence, image_result=None):
    crop_type = str(data.get("cropType", "Unknown")).strip()

    risk_factors = analyze_risk_factors(data)
    image_status = image_result.get("image_stress") if image_result else "Not used"

    initial_final_status = choose_worst_status(status, image_status)
    final_status, safety_flags = apply_safety_layer(
        sensor_status=initial_final_status,
        risk_factors=risk_factors,
        image_result=image_result
    )

    diagnosis = build_diagnosis(data, final_status, risk_factors, image_result, safety_flags)
    actions = build_actions(risk_factors, image_result)
    monitoring = build_monitoring(risk_factors, final_status, image_result)
    backend_flags = build_backend_flags(risk_factors, final_status, image_result, safety_flags)
    recommendations = build_recommendations(data, final_status, image_result, safety_flags)
    alert_info = get_alert_info(final_status)
    notification = build_notification_payload(final_status, diagnosis, actions)

    summary = build_summary(
        sensor_status=status,
        image_stress=image_status,
        final_status=final_status,
        safety_flags=safety_flags
    )

    response = {"status": final_status,
        "sensor_status": status,
        "image_status": image_status,
        "final_status": final_status,
        # اسم الزرعة راجع في السينسورات والكاميرا
        "cropType": crop_type,
        "cropName": crop_type,
        "plant_name": crop_type,

        "confidence": confidence,
        "alert": alert_info["alert"],
        "severity": alert_info["severity"],
        "summary": summary,
        "general_recommendation": solutions.get(final_status, ""),
        "diagnosis": diagnosis,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "actions": actions,
        "monitoring": monitoring,
        "backend_flags": backend_flags,
       "safety_layer": {
        "applied": len(safety_flags) > 0,
        "flags": safety_flags,
        "sensor_model_status": status,
        "image_status": image_status,
        "combined_status_before_safety": initial_final_status,
        "status_after_safety": final_status
    },
        "notification": notification,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if image_result is not None:
        response["image_analysis"] = image_result

    return response


def build_image_only_response(image_result, crop_type="Unknown"):
    image_stress = image_result.get("image_stress", "Healthy")
    health_score = float(image_result.get("health_score", 0.0))
    severity_score = float(image_result.get("severity_score", round(1.0 - health_score, 3)))
    visual_problem = image_result.get("visual_problem", "General Visual Stress")
    visual_problem_ar = image_result.get("visual_problem_ar", "إجهاد بصري عام")

    if image_stress == "Healthy":
        status = "Healthy"
        disease_name = "No Clear Disease Detected"
        confidence = round(max(health_score, 0.50), 3)
    else:
        status = "Infected" if visual_problem in [
            "Leaf Spot / Fungal Suspicion",
            "Necrosis / Severe Leaf Damage"
        ] else "Detected"
        disease_name = visual_problem
        confidence = round(max(severity_score, 0.50), 3)

    return {
        "status": status,
        "final_status": image_stress,

        # اسم الزرعة راجع في الكاميرا
        "cropType": crop_type,
        "cropName": crop_type,
        "plant_name": crop_type,

        "disease_name": disease_name,
        "disease_name_ar": visual_problem_ar,
        "confidence": confidence,
        "health_score": round(health_score, 3),
        "severity_score": round(severity_score, 3),
        "summary": image_result.get("visual_explanation", ""),
        "recommendations": image_result.get("image_recommendations", []),
        "analysis": image_result,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart Plant Health API V2 is running ✅",
        "version": "2.0",
        "features": [
            "sensor prediction",
            "image analysis",
            "sensor + image fusion",
            "safety override layer",
            "visual disease suspicion"
        ]
    }), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "message": "API is healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

        data = normalize_sensor_data(data)
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

        data = normalize_sensor_data(data)
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
    """
    V2 supports two modes:

    1) Image only:
       form-data:
       - file
       - cropType optional

    2) Image + sensors:
       form-data:
       - file
       - cropType
       - temperature
       - humidity
       - soilMoisture
       - soilTemp
       - light
    """
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "For /api/predict_with_image use Body -> form-data with file field named 'file'"
            }), 400

        image_file = request.files["file"]

        if image_file is None or image_file.filename.strip() == "":
            return jsonify({"error": "No image selected"}), 400

        form_data = request.form.to_dict()

        crop_type = (
            form_data.get("cropType")
            or form_data.get("cropName")
            or form_data.get("plantName")
            or form_data.get("plant_name")
            or "Unknown"
        ).strip()

        image_result = analyze_plant_image(image_file)

        if "error" in image_result:
            return jsonify({"error": image_result["error"]}), 400

        # If all sensor fields exist, fuse image + sensors
        if has_all_sensor_fields(form_data):
            data = {
                "cropType": crop_type,
                "temperature": form_data.get("temperature"),
                "humidity": form_data.get("humidity"),
                "soilMoisture": form_data.get("soilMoisture"),
                "soilTemp": form_data.get("soilTemp"),
                "light": form_data.get("light")
            }

            data = normalize_sensor_data(data)
            status, confidence = predict_sensor_status(data)

            response_data = build_advanced_response(
                data=data,
                status=status,
                confidence=confidence,
                image_result=image_result
            )

            request_type = "predict_with_image_and_sensors"
            saved_input = data

        else:
            response_data = build_image_only_response(image_result, crop_type)
            request_type = "predict_with_image"
            saved_input = {
                "cropType": crop_type,
                "mode": "image_only"
            }

        save_prediction({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "request_type": request_type,
            "image_filename": image_file.filename,
            "input": saved_input,
            "image_analysis": image_result,
            "result": response_data
        })

        return jsonify(response_data), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/image_predict", methods=["POST"])
def image_predict_alias():
    return predict_with_image()


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
    print("🚀 Starting Smart Plant Health API V2...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
