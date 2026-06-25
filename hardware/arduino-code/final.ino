#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "DHT.h"


// API & WiFi Configuration
const char* ssid = "Khaled Younis";
const char* password = "12341234";

//  Pins & Sensor Setup
#define DHTP 12
#define LIGHTP 13
#define soil 15
DHT dht(DHTP, DHT11);

#define CAMERA_MODEL_AI_THINKER 
#include "camera_pins.h"

//Timer
unsigned long lastSensorTime = 0;
unsigned long lastcamTime = 0;
const unsigned long sensorInterval = 300000/5; // 5 minutes
const unsigned long camInterval = 300000*12*24;

bool doneToday = false;

void setup() {  
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
  
  dht.begin();
  pinMode(LIGHTP, INPUT);
  analogReadResolution(12);

  // Camera Configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_RGB565;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);



  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");
  

}
/*void sendData(float temp, float hum, String light, int moisture) {
  String body = "";
  String boundary   = "----ESP32Boundary";
  // deviceSerial
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"deviceSerial\"\r\n\r\n";
  body += "ESP32-UNIT-01\r\n";
  // temp
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"temp\"\r\n\r\n";
  body += String(temp) + "\r\n";
  // hum
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"hum\"\r\n\r\n";
  body += String(hum) + "\r\n";
  // soil
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"Soil\"\r\n\r\n";
  body += String(moisture) + "\r\n";
  // light
  body += "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"light\"\r\n\r\n";
  body += light + "\r\n";
  body += "\r\n--" + boundary + "--\r\n";
  
  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  http.begin(client, sensorUploadUrl);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  int httpResponseCode = http.POST(body);
  
  if (httpResponseCode > 0) {
  Serial.println(http.getString());
  }

  Serial.print("Response: ");
  Serial.println(httpResponseCode);
  http.end();
}*/
void sendData(float temp, float hum, String lightVal, int soilMoisture) {
  WiFiClientSecure client;
  client.setInsecure(); // حل مشكلة Error -1 و -11 مع Vercel
  HTTPClient http;

  // 1. الرابط
  http.begin(client, "https://ecosense-backend.vercel.app/api/sensors/upload");

  // 2. تغيير نوع البيانات ليكون أسهل وأضمن (Url Encoded)
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");

  // 3. بناء الداتا بشكل بسيط (Key=Value)
  // تأكد من استخدام نفس الأسماء اللي في الـ Schema عندك
  String httpRequestData = "deviceSerial=ESP32-UNIT-01";
  httpRequestData += "&temp=" + String(temp);
  httpRequestData += "&hum=" + String(hum);
  httpRequestData += "&Soil=" + String(soilMoisture);
  httpRequestData += "&light=" + lightVal;

  // 4. الإرسال
  int httpResponseCode = http.POST(httpRequestData);

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.print("HTTP Response code: ");
    Serial.println(httpResponseCode);
    Serial.println("Response: " + response);
  } else {
    Serial.print("Error code: ");
    Serial.println(httpResponseCode);
  }

  http.end();
}

/*void SendImage() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  String boundary = "----ESP32Boundary";

  String imagePart  = "--" + boundary + "\r\n";
  imagePart        += "Content-Disposition: form-data; name=\"image\"; filename=\"cam.jpg\"\r\n";
  imagePart        += "Content-Type: image/jpeg\r\n\r\n";

  String serialPart  = "\r\n--" + boundary + "\r\n";
  serialPart        += "Content-Disposition: form-data; name=\"deviceSerial\"\r\n\r\n";
  serialPart        += "ESP32-UNIT-01\r\n";

  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  size_t totalLen = imagePart.length() + fb->len + serialPart.length() + bodyEnd.length();
  uint8_t* postData = (uint8_t*) malloc(totalLen);

  if (!postData) {
    Serial.println("malloc failed");
    esp_camera_fb_return(fb);
    return;
  }

  size_t offset = 0;
  memcpy(postData + offset, imagePart.c_str(),  imagePart.length());  offset += imagePart.length();
  memcpy(postData + offset, fb->buf,            fb->len);             offset += fb->len;
  memcpy(postData + offset, serialPart.c_str(), serialPart.length()); offset += serialPart.length();
  memcpy(postData + offset, bodyEnd.c_str(),    bodyEnd.length());

  esp_camera_fb_return(fb);  

  WiFiClientSecure secureClient;
  secureClient.setInsecure();

  HTTPClient http;
  http.begin(secureClient, photoCheckUrl);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  int httpResponseCode = http.POST(postData, totalLen);

  if (httpResponseCode > 0) {
    Serial.printf("[SendImage] Response %d: %s\n", httpResponseCode, http.getString().c_str());
  } else {
    Serial.printf("[SendImage] Error: %s\n", HTTPClient::errorToString(httpResponseCode).c_str());
  }

  free(postData);  
  http.end();
}*/
void SendImage() {
  // 1. التقاط الصورة من الكاميرا
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }
  uint8_t* jpg_buf = NULL;
  size_t   jpg_len = 0;
  bool converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
  esp_camera_fb_return(fb);   // ✅ ارجع الـ frame buffer فوراً

  if (!converted) {
    Serial.println("JPEG conversion failed");
    free(jpg_buf);
    return;
}

  String boundary = "----ESP32Boundary";

  // -------------------------------------------------------
  // التعديل الجوهري: إرسال الـ deviceSerial أولاً قبل الصورة
  // -------------------------------------------------------
  
  // الجزء الأول: السيريال نمبر (Text Part)
  String serialPart  = "--" + boundary + "\r\n";
  serialPart        += "Content-Disposition: form-data; name=\"deviceSerial\"\r\n\r\n";
  serialPart        += "ESP32-UNIT-01\r\n"; // نفس السيريال اللي في الـ Controller عندك

  // الجزء الثاني: بيانات الصورة (File Part)
  String imagePart   = "--" + boundary + "\r\n";
  imagePart         += "Content-Disposition: form-data; name=\"image\"; filename=\"cam.jpg\"\r\n";
  imagePart         += "Content-Type: image/jpeg\r\n\r\n";

  // قفلة الـ Body
  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  // 2. حساب الحجم الإجمالي وحجز الذاكرة
  size_t totalLen = serialPart.length() + imagePart.length() + jpg_len + bodyEnd.length();
  uint8_t* postData = (uint8_t*) malloc(totalLen);

  if (!postData) {
    Serial.println("Memory allocation (malloc) failed!");
    esp_camera_fb_return(fb);
    return;
  }

  // 3. تجميع الـ Body بالترتيب الصحيح (السيريال ثم الصورة)
  size_t offset = 0;
  memcpy(postData + offset, serialPart.c_str(),  serialPart.length()); offset += serialPart.length();
  memcpy(postData + offset, imagePart.c_str(),   imagePart.length());  offset += imagePart.length();
  memcpy(postData + offset, jpg_buf,             jpg_len);             offset += jpg_len;
  memcpy(postData + offset, bodyEnd.c_str(),     bodyEnd.length());

  // تحرير ذاكرة فريم الكاميرا بعد النسخ
  esp_camera_fb_return(fb);  

  // 4. إرسال الطلب للسيرفر (Vercel)
  WiFiClientSecure secureClient;
  secureClient.setInsecure(); // لتخطي فحص شهادة SSL الخاصة بـ Vercel

  HTTPClient http;
  http.begin(secureClient, "https://ecosense-backend.vercel.app/api/images/upload"); // تأكد إن الرابط هو https://ecosense-backend.vercel.app/api/images/upload
  
  // زيادة وقت الانتظار لأن رفع الصورة وتحليلها بياخد وقت طويل
  http.setTimeout(30000); 
  
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  int httpResponseCode = http.POST(postData, totalLen);

  // 5. قراءة الرد
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("[SendImage] Success! Response Code: %d\n", httpResponseCode);
    Serial.println("Server Response: " + response);
  } else {
    Serial.printf("[SendImage] Error: %s\n", HTTPClient::errorToString(httpResponseCode).c_str());
  }

  // تنظيف الذاكرة
  free(jpg_buf);  
  http.end();
}

void loop() {
  unsigned long currentMillis = millis();

  // TASK 1: Send Sensor Data every 5 minutes
  if (currentMillis - lastSensorTime >= sensorInterval) {
    lastSensorTime = currentMillis;

    float hum = dht.readHumidity();
    float temp = dht.readTemperature();
    int soilmoi = analogRead(soil); 
    int moisture = map(soilmoi, 4095, 0, 0, 100);
    int lightVal = digitalRead(LIGHTP);
    String lightStatus = (lightVal == LOW) ? "Light detected" : "Dark";

    Serial.println(">>> Sending Sensors...");
    sendData(temp, hum, lightStatus, moisture);SendImage();
  }
  //8:00 sending the Image
  /*if (currentMillis - lastcamTime >= camInterval) {
    lastcamTime = currentMillis;
    
}*/

}

