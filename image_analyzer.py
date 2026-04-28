import cv2
import numpy as np


# =========================
# Smart image analyzer V2
# =========================
# This is still a classical Computer Vision analyzer, not a CNN/Deep Learning model.
# It gives stronger visual features than the old color-only version:
# green, yellow/chlorosis, brown/necrosis, dark spots, damaged area, and visual problem.


def _safe_ratio(part, total):
    if total <= 0:
        return 0.0
    return float(part) / float(total)


def _largest_plant_region(mask):
    """Keep the largest connected plant-like area to reduce background noise."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # Skip background label 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]

    # If largest area is too small, return original mask so the caller can handle it
    if largest_area < 500:
        return mask

    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_label] = 255
    return cleaned


def _classify_visual_problem(green_ratio, yellow_ratio, brown_ratio, dark_spot_ratio, damaged_ratio):
    """Return a simple visual diagnosis from color/spot features."""
    if dark_spot_ratio >= 0.10 and yellow_ratio >= 0.10:
        return {
            "visual_problem": "Leaf Spot / Fungal Suspicion",
            "visual_problem_ar": "اشتباه تبقع أوراق أو إصابة فطرية",
            "visual_explanation": "تم رصد بقع داكنة واضحة مع اصفرار حولها، وده غالبًا يشير لمشكلة ورقية أو فطرية.",
        }

    if brown_ratio >= 0.10 or dark_spot_ratio >= 0.16:
        return {
            "visual_problem": "Necrosis / Severe Leaf Damage",
            "visual_problem_ar": "تلف أو احتراق واضح في نسيج الورقة",
            "visual_explanation": "نسبة المناطق الداكنة أو البنية مرتفعة، وده يدل على تلف في نسيج الورقة.",
        }

    if yellow_ratio >= 0.30 and dark_spot_ratio < 0.07:
        return {
            "visual_problem": "Chlorosis / Nutrient Deficiency Suspicion",
            "visual_problem_ar": "اصفرار أوراق أو اشتباه نقص عناصر",
            "visual_explanation": "الاصفرار واضح مقارنة باللون الأخضر، وده ممكن يرتبط بنقص عناصر أو إجهاد في الامتصاص.",
        }

    if damaged_ratio >= 0.25:
        return {
            "visual_problem": "General Visual Stress",
            "visual_problem_ar": "إجهاد بصري عام على النبات",
            "visual_explanation": "فيه نسبة ملحوظة من الاصفرار أو البقع، لكن النمط مش كافي لتحديد سبب واحد بدقة.",
        }

    if green_ratio >= 0.45 and damaged_ratio < 0.18:
        return {
            "visual_problem": "No Clear Disease Detected",
            "visual_problem_ar": "لا توجد أعراض مرضية واضحة",
            "visual_explanation": "معظم المنطقة النباتية ما زالت خضراء ولا توجد بقع أو اصفرار بنسبة خطيرة.",
        }

    return {
        "visual_problem": "Mild Visual Stress",
        "visual_problem_ar": "إجهاد بصري بسيط",
        "visual_explanation": "توجد علامات بسيطة تحتاج متابعة، لكنها ليست شديدة حسب التحليل اللوني الحالي.",
    }


def _build_image_recommendations(image_stress, visual_problem):
    if image_stress == "Healthy":
        return [
            "استمر في المتابعة الدورية للنبات.",
            "حافظ على برنامج الري والتسميد الحالي.",
            "التقط صورة جديدة عند ظهور أي اصفرار أو بقع.",
        ]

    if visual_problem == "Leaf Spot / Fungal Suspicion":
        return [
            "اعزل الورقة أو النبات المصاب إذا كانت الإصابة منتشرة.",
            "أزل الأوراق شديدة الإصابة لتقليل مصدر العدوى.",
            "حسّن التهوية وقلل البلل على الأوراق.",
            "استخدم معاملة فطرية مناسبة حسب توصية المختص أو الإرشاد الزراعي المحلي.",
        ]

    if visual_problem == "Chlorosis / Nutrient Deficiency Suspicion":
        return [
            "راجع برنامج التسميد خصوصًا النيتروجين والحديد والمغنيسيوم.",
            "افحص pH التربة أو المحلول لأن ارتفاعه قد يقلل امتصاص العناصر.",
            "تابع هل الاصفرار في الأوراق القديمة أم الحديثة لتحديد العنصر الأقرب للنقص.",
        ]

    if visual_problem == "Necrosis / Severe Leaf Damage":
        return [
            "أزل الأجزاء الميتة أو شديدة التلف.",
            "راجع الحرارة والري والملوحة لأن التلف قد يكون ناتجًا عن إجهاد شديد.",
            "افحص باقي النبات للتأكد من عدم انتشار الأعراض.",
        ]

    return [
        "راجع الري والتهوية والتسميد.",
        "تابع النبات خلال 48 ساعة بصورة جديدة.",
        "إذا زادت البقع أو الاصفرار، يفضل فحص النبات ميدانيًا.",
    ]


def analyze_plant_image(image_file):
    try:
        if image_file is None:
            return {"error": "No image provided"}

        # Support Flask file objects and direct file paths for testing
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

        img = cv2.resize(img, (700, 700))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # -------------------------
        # Color masks
        # -------------------------
        # Green healthy tissue
        green_mask = cv2.inRange(
            hsv,
            np.array([35, 35, 35]),
            np.array([90, 255, 255])
        )

        # Yellow/chlorotic tissue
        yellow_mask = cv2.inRange(
            hsv,
            np.array([18, 45, 75]),
            np.array([38, 255, 255])
        )

        # Brown/necrotic tissue
        brown_mask = cv2.inRange(
            hsv,
            np.array([5, 45, 20]),
            np.array([22, 255, 210])
        )

        # Dark lesions/spots. This catches black, dark purple, and very dark brown areas.
        h, s, v = cv2.split(hsv)
        dark_mask = np.where(((v < 105) & (s > 25)), 255, 0).astype(np.uint8)

        # Plant-like mask = union of possible leaf colors
        plant_mask = cv2.bitwise_or(green_mask, yellow_mask)
        plant_mask = cv2.bitwise_or(plant_mask, brown_mask)
        plant_mask = cv2.bitwise_or(plant_mask, dark_mask)

        kernel = np.ones((5, 5), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
        plant_mask = _largest_plant_region(plant_mask)

        plant_pixels = int(np.sum(plant_mask > 0))
        if plant_pixels < 800:
            return {"error": "No plant detected clearly"}

        # Restrict all masks to plant area
        green_pixels = int(np.sum((green_mask > 0) & (plant_mask > 0)))
        yellow_pixels = int(np.sum((yellow_mask > 0) & (plant_mask > 0)))
        brown_pixels = int(np.sum((brown_mask > 0) & (plant_mask > 0)))
        dark_pixels = int(np.sum((dark_mask > 0) & (plant_mask > 0)))

        green_ratio = _safe_ratio(green_pixels, plant_pixels)
        yellow_ratio = _safe_ratio(yellow_pixels, plant_pixels)
        brown_ratio = _safe_ratio(brown_pixels, plant_pixels)
        dark_spot_ratio = _safe_ratio(dark_pixels, plant_pixels)

        damaged_ratio = min(1.0, yellow_ratio + brown_ratio + dark_spot_ratio)

        # Stronger health score than old version
        health_score = (
            0.95 * green_ratio
            - 0.60 * yellow_ratio
            - 1.05 * brown_ratio
            - 1.25 * dark_spot_ratio
            + 0.15
        )
        health_score = max(0.0, min(1.0, health_score))
        severity_score = round(1.0 - health_score, 3)

        # -------------------------
        # Stress classification
        # -------------------------
        if dark_spot_ratio >= 0.12 and yellow_ratio >= 0.08:
            image_stress = "High Stress"
        elif brown_ratio >= 0.10 or damaged_ratio >= 0.38:
            image_stress = "High Stress"
        elif health_score >= 0.60 and green_ratio >= 0.45 and damaged_ratio < 0.18:
            image_stress = "Healthy"
        elif health_score >= 0.35 and damaged_ratio < 0.35:
            image_stress = "Moderate Stress"
        else:
            image_stress = "High Stress"

        visual_info = _classify_visual_problem(
            green_ratio=green_ratio,
            yellow_ratio=yellow_ratio,
            brown_ratio=brown_ratio,
            dark_spot_ratio=dark_spot_ratio,
            damaged_ratio=damaged_ratio,
        )

        recommendations = _build_image_recommendations(
            image_stress=image_stress,
            visual_problem=visual_info["visual_problem"]
        )

        return {
            "green_ratio": round(green_ratio, 3),
            "yellow_ratio": round(yellow_ratio, 3),
            "brown_ratio": round(brown_ratio, 3),
            "dark_spot_ratio": round(dark_spot_ratio, 3),
            "damaged_ratio": round(damaged_ratio, 3),
            "health_score": round(health_score, 3),
            "severity_score": severity_score,
            "plant_pixels": plant_pixels,
            "image_stress": image_stress,
            "visual_problem": visual_info["visual_problem"],
            "visual_problem_ar": visual_info["visual_problem_ar"],
            "visual_explanation": visual_info["visual_explanation"],
            "visual_flags": {
                "has_chlorosis": yellow_ratio >= 0.20,
                "has_dark_spots": dark_spot_ratio >= 0.08,
                "has_necrosis": brown_ratio >= 0.08,
                "needs_attention": image_stress != "Healthy"
            },
            "image_recommendations": recommendations
        }

    except Exception as e:
        return {"error": str(e)}
