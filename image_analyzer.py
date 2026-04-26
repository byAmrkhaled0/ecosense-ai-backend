import cv2
import numpy as np

def analyze_plant_image(image_file):
    try:
        if image_file is None:
            return {"error": "No image provided"}

        file_bytes = image_file.read()
        if not file_bytes:
            return {"error": "Empty image file"}

        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid or unsupported image"}

        img = cv2.resize(img, (600, 600))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect plant area
        plant_lower = np.array([20, 20, 20])
        plant_upper = np.array([100, 255, 255])
        plant_mask = cv2.inRange(hsv, plant_lower, plant_upper)

        kernel = np.ones((5, 5), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)

        plant_pixels = np.sum(plant_mask > 0)
        if plant_pixels == 0:
            return {"error": "No plant detected clearly"}

        # Color ranges
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([34, 255, 255])

        brown_lower = np.array([5, 60, 20])
        brown_upper = np.array([20, 255, 200])

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

        green_pixels = np.sum((green_mask > 0) & (plant_mask > 0))
        yellow_pixels = np.sum((yellow_mask > 0) & (plant_mask > 0))
        brown_pixels = np.sum((brown_mask > 0) & (plant_mask > 0))

        green_ratio = green_pixels / plant_pixels
        yellow_ratio = yellow_pixels / plant_pixels
        brown_ratio = brown_pixels / plant_pixels

        health_score = (
            (1.0 * green_ratio)
            - (0.9 * yellow_ratio)
            - (1.6 * brown_ratio)
        )

        health_score = max(0.0, min(1.0, health_score))

        if brown_ratio > 0.07 or yellow_ratio > 0.20:
            stress = "High Stress"
        elif health_score >= 0.50:
            stress = "Healthy"
        elif health_score >= 0.25:
            stress = "Moderate Stress"
        else:
            stress = "High Stress"

        return {
            "green_ratio": round(green_ratio, 3),
            "yellow_ratio": round(yellow_ratio, 3),
            "brown_ratio": round(brown_ratio, 3),
            "health_score": round(health_score, 3),
            "plant_pixels": int(plant_pixels),
            "image_stress": stress
        }

    except Exception as e:
        return {"error": str(e)}