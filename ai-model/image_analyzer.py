import cv2
import numpy as np


# =========================
# Smart image analyzer V7
# =========================
# Rule-based Computer Vision analyzer for plant health.
#
# يرجع:
# - حالة النبات
# - المرض/المشكلة المحتملة
# - شرح التشخيص
# - العلاج
# - التوصيات
# - نسب التحليل
#
# ملاحظة مهمة:
# ده OpenCV rule-based analyzer مش CNN model.
# لكنه مضبوط لتقليل الأخطاء الشائعة:
# - عدم اعتبار الظلال مرض
# - عدم اعتبار النبات المصاب Healthy
# - توحيد status مع visual_problem


def _safe_ratio(part, total):
    if total <= 0:
        return 0.0
    return float(part) / float(total)


def _clean_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_plant_region(mask):
    """
    Keep the largest connected plant-like region.
    Helps reduce background noise.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]

    if largest_area < 800:
        return mask

    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_label] = 255

    return cleaned


def _is_very_green_healthy(
    green_ratio,
    yellow_ratio,
    brown_ratio,
    health_score
):
    """
    Strong healthy override.
    Used for images that are clearly green and healthy,
    even if dark shadows are detected.
    """

    return (
        green_ratio >= 0.65
        and health_score >= 0.70
        and yellow_ratio < 0.10
        and brown_ratio < 0.08
    )


def _has_leaf_spot_pattern(
    yellow_ratio,
    dark_spot_ratio,
    damaged_ratio
):
    """
    Leaf spot pattern:
    yellowing + visible dark spots + enough damage.
    """

    return (
        dark_spot_ratio >= 0.14
        and yellow_ratio >= 0.20
        and damaged_ratio >= 0.35
    )


def _has_severe_leaf_spot_pattern(
    yellow_ratio,
    dark_spot_ratio,
    damaged_ratio
):
    """
    Severe fungal/spot pattern.
    """

    return (
        dark_spot_ratio >= 0.30
        and yellow_ratio >= 0.25
        and damaged_ratio >= 0.60
    )


def _has_necrosis_pattern(
    brown_ratio,
    damaged_ratio
):
    """
    Necrosis / leaf burn pattern.
    """

    return (
        brown_ratio >= 0.22
        and damaged_ratio >= 0.50
    )


def _is_regular_healthy(
    green_ratio,
    yellow_ratio,
    brown_ratio,
    dark_spot_ratio,
    damaged_ratio,
    health_score
):
    """
    Normal healthy condition.
    More strict than very_green_healthy.
    """

    return (
        green_ratio >= 0.40
        and yellow_ratio < 0.25
        and damaged_ratio < 0.42
        and dark_spot_ratio < 0.25
        and brown_ratio < 0.15
        and health_score >= 0.38
    )


def _classify_image_stress(
    green_ratio,
    yellow_ratio,
    brown_ratio,
    dark_spot_ratio,
    damaged_ratio,
    health_score
):
    """
    Final stress classification.
    Order is important.
    """

    # 1) Clearly healthy green image
    if _is_very_green_healthy(
        green_ratio,
        yellow_ratio,
        brown_ratio,
        health_score
    ):
        return "Healthy"

    # 2) Severe disease patterns first
    if _has_severe_leaf_spot_pattern(
        yellow_ratio,
        dark_spot_ratio,
        damaged_ratio
    ):
        return "High Stress"

    if _has_necrosis_pattern(
        brown_ratio,
        damaged_ratio
    ):
        return "High Stress"

    # 3) Leaf spot should override normal healthy
    if _has_leaf_spot_pattern(
        yellow_ratio,
        dark_spot_ratio,
        damaged_ratio
    ):
        return "Moderate Stress"

    # 4) Regular healthy
    if _is_regular_healthy(
        green_ratio,
        yellow_ratio,
        brown_ratio,
        dark_spot_ratio,
        damaged_ratio,
        health_score
    ):
        return "Healthy"

    # 5) General stress
    if damaged_ratio < 0.58:
        return "Moderate Stress"

    return "High Stress"


def _classify_visual_problem(
    green_ratio,
    yellow_ratio,
    brown_ratio,
    dark_spot_ratio,
    damaged_ratio,
    health_score
):
    """
    Detect the likely visual problem.
    Must be consistent with stress classification.
    """

    # 1) Clearly healthy green image
    if _is_very_green_healthy(
        green_ratio,
        yellow_ratio,
        brown_ratio,
        health_score
    ):
        return {
            "visual_problem": "No Clear Disease Detected",
            "visual_problem_ar": "لا توجد أعراض مرضية واضحة",
            "disease_name": "No Clear Disease Detected",
            "disease_name_ar": "لا توجد أعراض مرضية واضحة",
            "visual_explanation":
                "النبات يبدو صحيًا من الصورة؛ نسبة اللون الأخضر عالية ولا يوجد اصفرار أو تلف واضح."
        }

    # 2) Severe leaf spot / fungal suspicion
    if _has_severe_leaf_spot_pattern(
        yellow_ratio,
        dark_spot_ratio,
        damaged_ratio
    ):
        return {
            "visual_problem": "Severe Leaf Spot / Fungal Suspicion",
            "visual_problem_ar": "اشتباه إصابة فطرية أو تبقع أوراق شديد",
            "disease_name": "Severe Leaf Spot / Fungal Suspicion",
            "disease_name_ar": "اشتباه تبقع أوراق أو إصابة فطرية شديدة",
            "visual_explanation":
                "تم رصد اصفرار واضح مع بقع داكنة كثيرة ونسبة تلف مرتفعة، "
                "وده يشير لاحتمال إصابة فطرية أو تبقع أوراق بدرجة شديدة."
        }

    # 3) Necrosis / burn
    if _has_necrosis_pattern(
        brown_ratio,
        damaged_ratio
    ):
        return {
            "visual_problem": "Necrosis / Severe Leaf Damage",
            "visual_problem_ar": "تلف أو احتراق واضح في نسيج الورقة",
            "disease_name": "Possible Necrosis / Leaf Burn",
            "disease_name_ar": "اشتباه احتراق أو تلف نسيج الورقة",
            "visual_explanation":
                "نسبة المناطق البنية أو الجافة مرتفعة، "
                "وده ممكن يكون بسبب حرارة عالية، ملوحة، نقص مياه، أو تلف شديد."
        }

    # 4) Leaf spot / fungal suspicion
    if _has_leaf_spot_pattern(
        yellow_ratio,
        dark_spot_ratio,
        damaged_ratio
    ):
        return {
            "visual_problem": "Leaf Spot / Fungal Suspicion",
            "visual_problem_ar": "اشتباه تبقع أوراق أو إصابة فطرية",
            "disease_name": "Possible Leaf Spot Disease",
            "disease_name_ar": "اشتباه تبقع أوراق أو إصابة فطرية",
            "visual_explanation":
                "تم رصد اصفرار واضح مع بقع داكنة على الورقة، "
                "وده يشير غالبًا لاشتباه تبقع أوراق أو إصابة فطرية."
        }

    # 5) Chlorosis / nutrient deficiency
    if (
        yellow_ratio >= 0.50
        and green_ratio < 0.40
        and dark_spot_ratio < 0.18
        and brown_ratio < 0.15
    ):
        return {
            "visual_problem": "Chlorosis / Nutrient Deficiency Suspicion",
            "visual_problem_ar": "اصفرار أوراق أو اشتباه نقص عناصر",
            "disease_name": "Possible Chlorosis",
            "disease_name_ar": "اشتباه اصفرار بسبب نقص عناصر",
            "visual_explanation":
                "الاصفرار واضح بدون بقع داكنة كثيرة، "
                "وده ممكن يرتبط بنقص عناصر مثل النيتروجين أو الحديد أو ضعف امتصاص."
        }

    # 6) Regular healthy
    if _is_regular_healthy(
        green_ratio,
        yellow_ratio,
        brown_ratio,
        dark_spot_ratio,
        damaged_ratio,
        health_score
    ):
        return {
            "visual_problem": "No Clear Disease Detected",
            "visual_problem_ar": "لا توجد أعراض مرضية واضحة",
            "disease_name": "No Clear Disease Detected",
            "disease_name_ar": "لا توجد أعراض مرضية واضحة",
            "visual_explanation":
                "النبات يبدو بحالة جيدة من الصورة، ولا توجد أعراض مرضية واضحة بدرجة خطيرة."
        }

    return {
        "visual_problem": "General Visual Stress",
        "visual_problem_ar": "إجهاد بصري عام",
        "disease_name": "General Visual Stress",
        "disease_name_ar": "إجهاد بصري عام",
        "visual_explanation":
            "توجد علامات إجهاد على الورقة، لكن الصورة وحدها لا تكفي لتحديد مرض محدد بدقة."
    }


def _build_treatment_plan(image_stress, visual_problem):
    if image_stress == "Healthy":
        return [
            {
                "priority": 3,
                "title": "استمرار المتابعة",
                "details": "النبات يبدو سليمًا من الصورة. استمر في الري والتسميد والمتابعة الدورية."
            },
            {
                "priority": 3,
                "title": "تصوير دوري",
                "details": "التقط صورة جديدة عند ظهور اصفرار أو بقع أو تغير في شكل الورقة."
            }
        ]

    if visual_problem in (
        "Leaf Spot / Fungal Suspicion",
        "Severe Leaf Spot / Fungal Suspicion"
    ):
        return [
            {
                "priority": 1,
                "title": "عزل الأوراق المصابة",
                "details": "لو الإصابة واضحة في ورقة أو أكثر، افصل الأوراق المصابة لتقليل انتشار العدوى."
            },
            {
                "priority": 1,
                "title": "إزالة الأوراق شديدة الإصابة",
                "details": "أزل الأوراق التي تحتوي على بقع كثيرة أو تلف شديد."
            },
            {
                "priority": 2,
                "title": "تحسين التهوية",
                "details": "حسّن حركة الهواء حول النبات وقلل الرطوبة الزائدة."
            },
            {
                "priority": 2,
                "title": "تجنب بلل الأوراق",
                "details": "اسقِ النبات عند التربة وتجنب رش الماء على الأوراق."
            },
            {
                "priority": 2,
                "title": "معاملة فطرية مناسبة",
                "details": "استخدم معاملة فطرية مناسبة حسب نوع النبات أو استشر مختص زراعي."
            }
        ]

    if visual_problem == "Chlorosis / Nutrient Deficiency Suspicion":
        return [
            {
                "priority": 1,
                "title": "مراجعة التسميد",
                "details": "راجع النيتروجين والحديد والمغنيسيوم في برنامج التسميد."
            },
            {
                "priority": 2,
                "title": "فحص pH التربة",
                "details": "اختلال pH قد يمنع امتصاص العناصر حتى لو موجودة."
            },
            {
                "priority": 2,
                "title": "مراجعة الري",
                "details": "زيادة أو نقص الري قد يسبب اصفرار وضعف امتصاص."
            }
        ]

    if visual_problem == "Necrosis / Severe Leaf Damage":
        return [
            {
                "priority": 1,
                "title": "إزالة الأجزاء التالفة",
                "details": "أزل الأجزاء الجافة أو المحترقة إذا كانت شديدة التلف."
            },
            {
                "priority": 1,
                "title": "فحص الحرارة والري",
                "details": "راجع درجة الحرارة ورطوبة التربة والملوحة."
            },
            {
                "priority": 2,
                "title": "فحص باقي النبات",
                "details": "تأكد هل التلف في ورقة واحدة فقط أم منتشر في النبات."
            }
        ]

    return [
        {
            "priority": 2,
            "title": "متابعة خلال 48 ساعة",
            "details": "التقط صورة جديدة من نفس الزاوية وتابع هل الأعراض تزيد أم تقل."
        },
        {
            "priority": 2,
            "title": "مراجعة الري والإضاءة والتهوية",
            "details": "الإجهاد البصري قد يكون بسبب ظروف بيئية غير مناسبة."
        },
        {
            "priority": 3,
            "title": "فحص يدوي",
            "details": "افحص أسفل الورقة والساق بحثًا عن حشرات أو بقع أو عفن."
        }
    ]


def _build_image_recommendations(treatment_plan):
    return [
        item["title"] + ": " + item["details"]
        for item in treatment_plan
    ]


def _build_status_text(image_stress, visual_problem):
    if image_stress == "Healthy":
        return "Healthy"

    if visual_problem in (
        "Leaf Spot / Fungal Suspicion",
        "Severe Leaf Spot / Fungal Suspicion",
        "Necrosis / Severe Leaf Damage",
        "Chlorosis / Nutrient Deficiency Suspicion"
    ):
        return "Infected"

    return "Needs Review"


def _build_confidence(image_stress, health_score, severity_score, visual_problem):
    if image_stress == "Healthy":
        return round(max(0.70, min(0.98, health_score)), 3)

    if visual_problem in (
        "Leaf Spot / Fungal Suspicion",
        "Severe Leaf Spot / Fungal Suspicion",
        "Necrosis / Severe Leaf Damage"
    ):
        return round(max(0.65, min(0.92, severity_score + 0.25)), 3)

    if image_stress == "Moderate Stress":
        return round(max(0.55, min(0.80, severity_score)), 3)

    return round(max(0.70, min(0.95, severity_score)), 3)


def analyze_plant_image(image_file):
    try:
        if image_file is None:
            return {"error": "No image provided"}

        # Flask upload OR local path
        if isinstance(image_file, str):
            img = cv2.imread(image_file)
        else:
            file_bytes = image_file.read()

            if not file_bytes:
                return {"error": "Empty image file"}

            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid or unsupported image"}

        # =========================
        # Resize and convert
        # =========================

        img = cv2.resize(img, (700, 700))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # =========================
        # Color masks
        # =========================

        # Healthy green tissue
        green_mask = cv2.inRange(
            hsv,
            np.array([35, 35, 35]),
            np.array([90, 255, 255])
        )

        # Yellow / chlorosis
        yellow_mask = cv2.inRange(
            hsv,
            np.array([18, 45, 75]),
            np.array([38, 255, 255])
        )

        # Brown / necrosis
        brown_mask = cv2.inRange(
            hsv,
            np.array([5, 45, 20]),
            np.array([22, 255, 210])
        )

        # Dark spots
        # Conservative threshold to reduce false positives from shadows.
        dark_mask = np.where(
            ((v < 70) & (s > 45)),
            255,
            0
        ).astype(np.uint8)

        # =========================
        # Plant mask
        # =========================

        plant_mask = cv2.bitwise_or(green_mask, yellow_mask)
        plant_mask = cv2.bitwise_or(plant_mask, brown_mask)
        plant_mask = cv2.bitwise_or(plant_mask, dark_mask)

        plant_mask = _clean_mask(plant_mask, kernel_size=5)
        plant_mask = _largest_plant_region(plant_mask)

        plant_pixels = int(np.sum(plant_mask > 0))

        if plant_pixels < 800:
            return {
                "error": "No plant detected clearly",
                "message_ar": "الصورة غير واضحة أو النبات غير ظاهر بشكل كافٍ."
            }

        # =========================
        # Pixel counting
        # =========================

        green_pixels = int(np.sum((green_mask > 0) & (plant_mask > 0)))
        yellow_pixels = int(np.sum((yellow_mask > 0) & (plant_mask > 0)))
        brown_pixels = int(np.sum((brown_mask > 0) & (plant_mask > 0)))
        dark_pixels = int(np.sum((dark_mask > 0) & (plant_mask > 0)))

        green_ratio = _safe_ratio(green_pixels, plant_pixels)
        yellow_ratio = _safe_ratio(yellow_pixels, plant_pixels)
        brown_ratio = _safe_ratio(brown_pixels, plant_pixels)
        dark_spot_ratio = _safe_ratio(dark_pixels, plant_pixels)

        # =========================
        # Damage score
        # =========================

        damaged_ratio = min(
            1.0,
            (yellow_ratio * 0.35)
            + (brown_ratio * 0.75)
            + (dark_spot_ratio * 0.85)
        )

        # =========================
        # Health score
        # =========================

        health_score = (
            (1.20 * green_ratio)
            - (0.35 * yellow_ratio)
            - (0.75 * brown_ratio)
            - (0.75 * dark_spot_ratio)
            + 0.25
        )

        health_score = max(0.0, min(1.0, health_score))
        severity_score = round(1.0 - health_score, 3)

        # =========================
        # Classification
        # =========================

        image_stress = _classify_image_stress(
            green_ratio=green_ratio,
            yellow_ratio=yellow_ratio,
            brown_ratio=brown_ratio,
            dark_spot_ratio=dark_spot_ratio,
            damaged_ratio=damaged_ratio,
            health_score=health_score
        )

        visual_info = _classify_visual_problem(
            green_ratio=green_ratio,
            yellow_ratio=yellow_ratio,
            brown_ratio=brown_ratio,
            dark_spot_ratio=dark_spot_ratio,
            damaged_ratio=damaged_ratio,
            health_score=health_score
        )

        treatment_plan = _build_treatment_plan(
            image_stress=image_stress,
            visual_problem=visual_info["visual_problem"]
        )

        recommendations = _build_image_recommendations(treatment_plan)

        status = _build_status_text(
            image_stress=image_stress,
            visual_problem=visual_info["visual_problem"]
        )

        confidence = _build_confidence(
            image_stress=image_stress,
            health_score=health_score,
            severity_score=severity_score,
            visual_problem=visual_info["visual_problem"]
        )

        summary = visual_info["visual_explanation"]

        # =========================
        # Final response
        # =========================

        return {
            "status": status,
            "final_status": image_stress,
            "image_stress": image_stress,

            "disease_name": visual_info["disease_name"],
            "disease_name_ar": visual_info["disease_name_ar"],

            "visual_problem": visual_info["visual_problem"],
            "visual_problem_ar": visual_info["visual_problem_ar"],
            "visual_explanation": visual_info["visual_explanation"],

            "summary": summary,
            "confidence": confidence,

            "health_score": round(health_score, 3),
            "severity_score": severity_score,

            "green_ratio": round(green_ratio, 3),
            "yellow_ratio": round(yellow_ratio, 3),
            "brown_ratio": round(brown_ratio, 3),
            "dark_spot_ratio": round(dark_spot_ratio, 3),
            "damaged_ratio": round(damaged_ratio, 3),

            "plant_pixels": plant_pixels,

            "visual_flags": {
                "has_chlorosis": yellow_ratio >= 0.25,
                "has_dark_spots": (
                    dark_spot_ratio >= 0.14
                    and yellow_ratio >= 0.15
                ),
                "has_necrosis": brown_ratio >= 0.18,
                "needs_attention": image_stress != "Healthy"
            },

            "treatment_plan": treatment_plan,
            "image_recommendations": recommendations,
            "recommendations": recommendations,

            "capture_tips": [
                "صوّر النبات في إضاءة طبيعية بدون فلاش قوي.",
                "خلّي الخلفية بسيطة وفاتحة قدر الإمكان.",
                "قرّب الورقة أو الجزء المصاب من الكاميرا بدون اهتزاز.",
                "تجنب الظلال القوية لأنها قد تظهر كبقع مرضية."
            ],

            "note": (
                "هذا التشخيص تقديري من الصورة فقط. "
                "للحكم الأدق، يُفضل دمجه مع قراءات الحرارة والرطوبة ورطوبة التربة والإضاءة."
            )
        }

    except Exception as e:
        return {
            "error": str(e),
            "message_ar": "حدث خطأ أثناء تحليل الصورة."
        }
