import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { NavLink, Navigate, Outlet, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import {
  Activity, AlertTriangle, BarChart3, Bell, Camera, CheckCircle2, ChevronRight, Cpu,
  Eye, FileText, Globe2, Home, ImagePlus, Layers3, Leaf, Lock, LogIn, LogOut, Menu, Moon,
  Plus, ScanSearch, Search, ShieldCheck, Sparkles, Sprout, Sun, ThermometerSun, UploadCloud,
  Trash2, UserPlus, Users, X, Zap
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from 'recharts';
import { authAPI, dashboardAPI, sectorsAPI, usersAPI, notificationsAPI, devicesAPI, reportsAPI, tasksAPI, sensorsAPI, imagesAPI, diagnosisAPI, API_BASE_URL, PRODUCTION_BACKEND_ORIGIN, DEFAULT_HARDWARE_SECTOR_ID, getApiError } from './services/api';
import { connectSocket, disconnectSocket, onNotification } from './services/socket';

const API_BASE = API_BASE_URL;

const dictionaries = {
  en: {
    name: 'English', dir: 'ltr', brand: 'Smart Plant Health', tagline: 'AI Plant Diagnosis',
    home: 'Home', dashboard: 'Dashboard', diagnosis: 'Diagnosis', sensors: 'Sensors', sectors: 'Sectors', reports: 'Reports', alerts: 'Alerts', login: 'Login', signup: 'Sign Up', logout: 'Logout',
    heroBadge: 'Image + sensor AI diagnosis',
    heroTitle: 'Diagnose plant health before stress becomes damage.',
    heroDesc: 'Smart Plant Health classifies plant condition using sensor readings and plant images, then gives a clear final status, confidence, severity, diagnosis, and practical recommendations.',
    startDiagnosis: 'Start Diagnosis', viewDashboard: 'View Dashboard',
    productNotGrad: 'A complete smart agriculture product experience, built around plant health classification.',
    core1: 'Unified Final Status', core1Text: 'The interface highlights final_status as the official plant condition.',
    core2: 'Image + Sensor Fusion', core2Text: 'Combines crop type, temperature, humidity, soil moisture, soil temperature, light, and optional leaf image.',
    core3: 'Actionable Advice', core3Text: 'Shows diagnosis, severity, model flags, visual ratios, and recommendations.',
    finalStatus: 'Final Status', sensorStatus: 'Sensor Status', imageStatus: 'Image Status', confidence: 'Confidence', severity: 'Severity', recommendations: 'Recommendations', actions: 'Actions', apiMode: 'Live API', demoMode: 'Demo Mode',
    healthy: 'Healthy', moderate: 'Moderate Stress', high: 'High Stress', noImage: 'No Image', low: 'Low', medium: 'Medium', sufficient: 'Sufficient',
    plantDiagnosis: 'Plant Health Diagnosis', diagnosisSubtitle: 'Upload a plant image and sensor readings to classify the plant condition.',
    cropType: 'Crop Type', tomato: 'Tomato', corn: 'Corn', pepper: 'Pepper', mint: 'Mint', temperature: 'Air Temperature', humidity: 'Humidity', soilMoisture: 'Soil Moisture', soilTemp: 'Soil Temperature', light: 'Light Level',
    uploadImage: 'Upload Plant Image', chooseImage: 'Choose or drag plant image', analyze: 'Analyze Plant', analyzing: 'Analyzing...', result: 'AI Result', diagnosisText: 'Diagnosis', imageAnalysis: 'Image Analysis', greenRatio: 'Green Tissue', yellowRatio: 'Yellowing', brownRatio: 'Brown Tissue', darkSpotRatio: 'Dark Spots', damagedRatio: 'Visible Damage', diseaseName: 'Disease / Visual Problem', diseaseDetails: 'Disease Details', modelResponse: 'Model Response', userExplanation: 'What this means', dataSent: 'Submitted Sensor Data', visualProblem: 'Visual Problem', healthScore: 'Health Score', recommendationCards: 'Recommended Steps', actionCards: 'Immediate Actions', serviceFlagsTitle: 'Model Flags', riskFactorsTitle: 'Risk Factors', noDiseaseDetected: 'No clear disease detected', statusExplanation: 'The final result combines sensor context and image evidence, then highlights the safest recommended decision.',
    dashboardTitle: 'Plant Health Overview', dashboardSubtitle: 'Monitor the latest plant status, sensor summary, alerts, and AI trends.', liveReadings: 'Live Readings', latestReading: 'Latest Reading', modelKeywords: 'Model Keywords', fusionLayer: 'Fusion Layer', safetyLayer: 'Safety Layer', serviceFlags: 'Model Flags', riskFactors: 'Risk Factors',
    addReading: 'Add Sensor Reading', saveReading: 'Save Reading', history: 'History', statistics: 'Statistics',
    sectorsTitle: 'Sectors', sectorsSubtitle: 'Optional farm-zone tracking. Sectors help organize plants, but the product remains focused on plant health diagnosis.', addSector: 'Add Sector', sectorName: 'Sector Name', location: 'Location', crop: 'Crop', save: 'Save', delete: 'Delete',
    reportsTitle: 'Reports', reportsSubtitle: 'Export and review plant health trends.', exportData: 'Export Data', alertsTitle: 'Plant Alerts', alertsSubtitle: 'Warnings generated from high-risk readings and AI status.',
    email: 'Email', password: 'Password', fullName: 'Full Name', phone: 'Phone', confirmPassword: 'Confirm Password', remember: 'Remember me', forgot: 'Forgot password?', createAccount: 'Create Account', alreadyHave: 'Already have an account?', noAccount: 'No account yet?', continueGoogle: 'Continue with Google', authSubtitle: 'Access your plant health dashboard, diagnosis history, and sensor insights.',
    language: 'Language', theme: 'Theme', mobileReady: 'Mobile ready', darkLight: 'Dark & Light', multiLang: 'Multilingual', publicProduct: 'Public product',
    empty: 'No data yet', connected: 'Connected', offline: 'Offline', roleText: 'Smart classification for healthier plants.', footer: '© 2026 Smart Plant Health — AI plant diagnosis platform.'
  },
  ar: {
    name: 'العربية', dir: 'rtl', brand: 'Smart Plant Health', tagline: 'تشخيص حالة النبات بالذكاء الاصطناعي',
    home: 'الرئيسية', dashboard: 'لوحة التحكم', diagnosis: 'تشخيص النبات', sensors: 'الحساسات', sectors: 'القطاعات', reports: 'التقارير', alerts: 'التنبيهات', login: 'تسجيل الدخول', signup: 'إنشاء حساب', logout: 'خروج',
    heroBadge: 'تشخيص بالصور وقراءات الحساسات',
    heroTitle: 'شخّص صحة النبات قبل أن يتحول الإجهاد إلى ضرر.',
    heroDesc: 'منصة Smart Plant Health تصنّف حالة النبات باستخدام قراءات الحساسات وصور النبات، ثم تعرض الحالة النهائية ونسبة الثقة ودرجة الخطورة والتشخيص والتوصيات العملية.',
    startDiagnosis: 'ابدأ التشخيص', viewDashboard: 'افتح لوحة التحكم',
    productNotGrad: 'تجربة منتج زراعي ذكي متكاملة قائمة على تصنيف صحة النبات.',
    core1: 'حالة نهائية موحدة', core1Text: 'الواجهة تبرز final_status كالحالة الرسمية للنبات.',
    core2: 'دمج الصورة والحساسات', core2Text: 'دمج نوع النبات والحرارة والرطوبة ورطوبة التربة وحرارة التربة والإضاءة وصورة الورقة عند توفرها.',
    core3: 'توصيات قابلة للتنفيذ', core3Text: 'عرض التشخيص والخطورة وإشارات الموديل ونسب تحليل الصورة والتوصيات.',
    finalStatus: 'الحالة النهائية', sensorStatus: 'حالة الحساسات', imageStatus: 'حالة الصورة', confidence: 'نسبة الثقة', severity: 'درجة الخطورة', recommendations: 'التوصيات', actions: 'الإجراءات', apiMode: 'API مباشر', demoMode: 'وضع تجريبي',
    healthy: 'سليم', moderate: 'إجهاد متوسط', high: 'إجهاد مرتفع', noImage: 'لا توجد صورة', low: 'منخفضة', medium: 'متوسطة', sufficient: 'كافية',
    plantDiagnosis: 'تشخيص حالة النبات', diagnosisSubtitle: 'ارفع صورة النبات وأدخل قراءات الحساسات لتصنيف حالة النبات.',
    cropType: 'نوع النبات', tomato: 'طماطم', corn: 'ذرة', pepper: 'فلفل', mint: 'نعناع', temperature: 'حرارة الجو', humidity: 'الرطوبة', soilMoisture: 'رطوبة التربة', soilTemp: 'حرارة التربة', light: 'مستوى الإضاءة',
    uploadImage: 'رفع صورة النبات', chooseImage: 'اختر أو اسحب صورة النبات', analyze: 'تحليل النبات', analyzing: 'جاري التحليل...', result: 'نتيجة الذكاء الاصطناعي', diagnosisText: 'التشخيص', imageAnalysis: 'تحليل الصورة', greenRatio: 'النسيج الأخضر', yellowRatio: 'الاصفرار', brownRatio: 'النسيج البني', darkSpotRatio: 'البقع الداكنة', damagedRatio: 'التلف الظاهري', diseaseName: 'اسم المرض / المشكلة البصرية', diseaseDetails: 'تفاصيل المرض', modelResponse: 'استجابة الموديل', userExplanation: 'ماذا تعني النتيجة؟', dataSent: 'بيانات الحساسات المرسلة', visualProblem: 'المشكلة البصرية', healthScore: 'درجة الصحة', recommendationCards: 'الخطوات المقترحة', actionCards: 'الإجراءات الفورية', serviceFlagsTitle: 'إشارات الموديل', riskFactorsTitle: 'عوامل الخطورة', noDiseaseDetected: 'لا يوجد مرض واضح', statusExplanation: 'النتيجة النهائية تجمع بين قراءة الحساسات ودليل الصورة ثم تعرض القرار الآمن والتوصيات المناسبة.',
    dashboardTitle: 'نظرة عامة على صحة النبات', dashboardSubtitle: 'تابع آخر حالة للنبات وملخص الحساسات والتنبيهات واتجاهات الذكاء الاصطناعي.', liveReadings: 'قراءات مباشرة', latestReading: 'آخر قراءة', modelKeywords: 'مصطلحات الموديل', fusionLayer: 'طبقة الدمج', safetyLayer: 'طبقة الأمان', serviceFlags: 'إشارات الموديل', riskFactors: 'عوامل الخطورة',
    addReading: 'إضافة قراءة حساس', saveReading: 'حفظ القراءة', history: 'السجل', statistics: 'الإحصائيات',
    sectorsTitle: 'القطاعات', sectorsSubtitle: 'خاصية اختيارية لمتابعة المناطق الزراعية. القطاعات تساعد في التنظيم، لكن المنتج قائم أساسًا على تشخيص صحة النبات.', addSector: 'إضافة قطاع', sectorName: 'اسم القطاع', location: 'الموقع', crop: 'المحصول', save: 'حفظ', delete: 'حذف',
    reportsTitle: 'التقارير', reportsSubtitle: 'تصدير ومراجعة اتجاهات صحة النبات.', exportData: 'تصدير البيانات', alertsTitle: 'تنبيهات النبات', alertsSubtitle: 'تنبيهات ناتجة عن القراءات الخطرة وحالة الذكاء الاصطناعي.',
    email: 'البريد الإلكتروني', password: 'كلمة المرور', fullName: 'الاسم الكامل', phone: 'رقم الهاتف', confirmPassword: 'تأكيد كلمة المرور', remember: 'تذكرني', forgot: 'نسيت كلمة المرور؟', createAccount: 'إنشاء حساب', alreadyHave: 'لديك حساب بالفعل؟', noAccount: 'ليس لديك حساب؟', continueGoogle: 'المتابعة باستخدام جوجل', authSubtitle: 'ادخل إلى لوحة صحة النبات وسجل التشخيص وتحليلات الحساسات.',
    language: 'اللغة', theme: 'المظهر', mobileReady: 'متوافق مع الهاتف', darkLight: 'ليلي ونهاري', multiLang: 'متعدد اللغات', publicProduct: 'منتج عام',
    empty: 'لا توجد بيانات بعد', connected: 'متصل', offline: 'غير متصل', roleText: 'تصنيف ذكي لنباتات أكثر صحة.', footer: '© 2026 Smart Plant Health — منصة تشخيص حالة النبات بالذكاء الاصطناعي.'
  },
  fr: {}, es: {}, de: {}, it: {}, tr: {}
};

const additions = {
  fr: { name:'Français', dir:'ltr', brand:'Smart Plant Health', tagline:'Diagnostic IA des plantes', home:'Accueil', dashboard:'Tableau', diagnosis:'Diagnostic', sensors:'Capteurs', sectors:'Secteurs', reports:'Rapports', alerts:'Alertes', login:'Connexion', signup:'Créer un compte', heroBadge:'Diagnostic par images et capteurs', heroTitle:'Diagnostiquez la santé des plantes avant les dégâts.', heroDesc:'Smart Plant Health classe l’état des plantes avec capteurs et images, puis affiche l’état final, la confiance, la gravité et les recommandations.', startDiagnosis:'Démarrer', viewDashboard:'Ouvrir le tableau', plantDiagnosis:'Diagnostic de santé des plantes', diagnosisSubtitle:'Ajoutez une image et des mesures pour classifier l’état de la plante.', finalStatus:'État final', sensorStatus:'État capteurs', imageStatus:'État image', confidence:'Confiance', severity:'Gravité', recommendations:'Recommandations', healthy:'Saine', moderate:'Stress modéré', high:'Stress élevé', noImage:'Pas d’image', dashboardTitle:'Vue santé des plantes', dashboardSubtitle:'Suivez l’état, les capteurs et les alertes.', sectorsTitle:'Secteurs', sectorsSubtitle:'Fonction optionnelle pour organiser les zones agricoles.', reportsTitle:'Rapports', alertsTitle:'Alertes plante', email:'Email', password:'Mot de passe', fullName:'Nom complet', phone:'Téléphone', createAccount:'Créer un compte', footer:'© 2026 Smart Plant Health — Plateforme de diagnostic IA.' },
  es: { name:'Español', dir:'ltr', brand:'Smart Plant Health', tagline:'Diagnóstico vegetal IA', home:'Inicio', dashboard:'Panel', diagnosis:'Diagnóstico', sensors:'Sensores', sectors:'Sectores', reports:'Reportes', alerts:'Alertas', login:'Entrar', signup:'Crear cuenta', heroBadge:'Diagnóstico por imágenes y sensores', heroTitle:'Diagnostica la salud vegetal antes del daño.', heroDesc:'Smart Plant Health clasifica la condición de las plantas con sensores e imágenes y muestra estado final, confianza, severidad y recomendaciones.', startDiagnosis:'Iniciar diagnóstico', viewDashboard:'Abrir panel', plantDiagnosis:'Diagnóstico de salud vegetal', finalStatus:'Estado final', sensorStatus:'Estado sensores', imageStatus:'Estado imagen', confidence:'Confianza', severity:'Severidad', recommendations:'Recomendaciones', healthy:'Sana', moderate:'Estrés moderado', high:'Estrés alto', noImage:'Sin imagen', dashboardTitle:'Resumen de salud vegetal', sectorsTitle:'Sectores', reportsTitle:'Reportes', alertsTitle:'Alertas', email:'Email', password:'Contraseña', fullName:'Nombre completo', phone:'Teléfono', createAccount:'Crear cuenta', footer:'© 2026 Smart Plant Health — Plataforma IA.' },
  de: { name:'Deutsch', dir:'ltr', brand:'Smart Plant Health', tagline:'KI Pflanzendiagnose', home:'Start', dashboard:'Dashboard', diagnosis:'Diagnose', sensors:'Sensoren', sectors:'Sektoren', reports:'Berichte', alerts:'Warnungen', login:'Login', signup:'Registrieren', heroBadge:'Diagnose mit Bildern und Sensoren', heroTitle:'Pflanzengesundheit erkennen, bevor Schaden entsteht.', heroDesc:'Smart Plant Health klassifiziert Pflanzenzustände mit Sensoren und Bildern und zeigt Status, Vertrauen und Empfehlungen.', startDiagnosis:'Diagnose starten', viewDashboard:'Dashboard öffnen', plantDiagnosis:'Pflanzendiagnose', finalStatus:'Finaler Status', sensorStatus:'Sensorstatus', imageStatus:'Bildstatus', confidence:'Vertrauen', severity:'Schweregrad', recommendations:'Empfehlungen', healthy:'Gesund', moderate:'Mäßiger Stress', high:'Hoher Stress', noImage:'Kein Bild', dashboardTitle:'Pflanzengesundheit', sectorsTitle:'Sektoren', reportsTitle:'Berichte', alertsTitle:'Warnungen', email:'E-Mail', password:'Passwort', fullName:'Vollständiger Name', phone:'Telefon', createAccount:'Konto erstellen', footer:'© 2026 Smart Plant Health — KI-Plattform.' },
  it: { name:'Italiano', dir:'ltr', brand:'Smart Plant Health', tagline:'Diagnosi piante AI', home:'Home', dashboard:'Dashboard', diagnosis:'Diagnosi', sensors:'Sensori', sectors:'Settori', reports:'Report', alerts:'Avvisi', login:'Accedi', signup:'Registrati', heroBadge:'Diagnosi con immagini e sensori', heroTitle:'Diagnostica la salute delle piante prima del danno.', heroDesc:'Smart Plant Health classifica lo stato della pianta usando sensori e immagini.', startDiagnosis:'Avvia diagnosi', viewDashboard:'Apri dashboard', plantDiagnosis:'Diagnosi salute pianta', finalStatus:'Stato finale', sensorStatus:'Stato sensori', imageStatus:'Stato immagine', confidence:'Confidenza', severity:'Gravità', recommendations:'Raccomandazioni', healthy:'Sana', moderate:'Stress moderato', high:'Stress alto', noImage:'Nessuna immagine', dashboardTitle:'Panoramica salute pianta', sectorsTitle:'Settori', reportsTitle:'Report', alertsTitle:'Avvisi', email:'Email', password:'Password', fullName:'Nome completo', phone:'Telefono', createAccount:'Crea account', footer:'© 2026 Smart Plant Health — Piattaforma AI.' },
  tr: { name:'Türkçe', dir:'ltr', brand:'Smart Plant Health', tagline:'Yapay Zeka Bitki Teşhisi', home:'Ana Sayfa', dashboard:'Panel', diagnosis:'Teşhis', sensors:'Sensörler', sectors:'Sektörler', reports:'Raporlar', alerts:'Uyarılar', login:'Giriş', signup:'Kayıt ol', heroBadge:'Görüntü ve sensör teşhisi', heroTitle:'Stres hasara dönüşmeden bitki sağlığını teşhis edin.', heroDesc:'Smart Plant Health sensör ve görüntülerle bitki durumunu sınıflandırır.', startDiagnosis:'Teşhisi başlat', viewDashboard:'Paneli aç', plantDiagnosis:'Bitki Sağlığı Teşhisi', finalStatus:'Son durum', sensorStatus:'Sensör durumu', imageStatus:'Görüntü durumu', confidence:'Güven', severity:'Şiddet', recommendations:'Öneriler', healthy:'Sağlıklı', moderate:'Orta stres', high:'Yüksek stres', noImage:'Görüntü yok', dashboardTitle:'Bitki Sağlığı Özeti', sectorsTitle:'Sektörler', reportsTitle:'Raporlar', alertsTitle:'Uyarılar', email:'E-posta', password:'Şifre', fullName:'Ad Soyad', phone:'Telefon', createAccount:'Hesap oluştur', footer:'© 2026 Smart Plant Health — AI platformu.' }
};
Object.keys(additions).forEach(k => { dictionaries[k] = { ...dictionaries.en, ...additions[k] }; });


const extraDictionaries = {
  en: {
    appName: 'Ecosense AI',
    brand: 'Smart Plant Health',
    tagline: 'AI Plant Diagnosis',
    requireAccount: 'Create an account first to start monitoring your plants.',
    homeNotificationTitle: 'Latest Plant Alert',
    noCriticalAlerts: 'No critical plant alerts right now.',
    goRegisterFirst: 'Register to start',
    userFriendlyResult: 'Plant Diagnosis Summary',
    condition: 'Plant Condition',
    diseaseProblem: 'Detected Disease / Problem',
    diagnosisSummary: 'Diagnosis Summary',
    recommendedCare: 'Recommended Care',
    immediateCare: 'Immediate Care',
    sentReadings: 'Readings Used',
    imageIndicators: 'Image Indicators',
    automaticReadings: 'Readings are loaded automatically from the connected sensor/model feed.',
    refreshReadings: 'Refresh readings',
    autoSensorFeed: 'Automatic Sensor Feed',
    farmTitle: 'Farm Monitoring',
    farmSubtitle: 'Track the whole farm through sectors. Each sector has its own sensors, diagnosis status, alerts, and latest AI reading.',
    sectorHealth: 'Sector Health',
    latestDiagnosis: 'Latest Diagnosis',
    sensorSnapshot: 'Sensor Snapshot',
    highRiskWarning: 'Danger alert: one of your plants is at high risk and needs immediate care.',
    notifications: 'Notifications',
    datasetAnalyzer: 'Dataset Analyzer',
    datasetSubtitle: 'Upload CSV/Excel data to prepare it for plant-condition analysis through the connected service.',
    uploadDataset: 'Upload Dataset',
    chooseDataset: 'Choose CSV or Excel file',
    datasetReady: 'Dataset selected and ready to send to the analysis service.',
    datasetName: 'Dataset Name',
    rowsEstimate: 'Estimated Rows',
    sendDataset: 'Send Dataset',
    datasetNote: 'When connected to the analysis service, this section can upload a full dataset and return batch plant-health results.',
    privateRoutesNote: 'Dashboard, diagnosis, sectors, reports, and alerts are available after registration/login.',
    accountCreated: 'Account created successfully.',
    loggedIn: 'Logged in successfully.',
    wholeFarm: 'Whole Farm',
    critical: 'Critical',
    safe: 'Safe',
    monitor: 'Monitor',
    uploadPlantImageOptional: 'Upload plant image optionally. Sensor values are provided automatically.',
    smartSummary: 'Smart Summary',
    





    diagnosisMode: 'Diagnosis Mode',
    imageOnlyMode: 'Image Only',
    sensorsOnlyMode: 'Sensors Only',
    combinedMode: 'Image + Sensors',
    modelDrivenNote: 'Analyze images and sensor readings through the connected services, then review clear recommendations.',
    manualSensorInput: 'Manual Sensor Values',
    sendToModel: 'Send to Model',
    suspicionAreas: 'Suspicion Areas',
    modelAlerts: 'Model Alerts',
    alarmAlert: 'Danger Alarm',
    linkDiagnosisToSector: 'Link Diagnosis to Sector',
    selectSector: 'Select Sector',
    linkedToSector: 'Diagnosis linked to sector.',
    modelLikelyCause: 'Likely Cause from Model',
    plantLibrary: 'Plant Disease Library',
    plantLibrarySub: 'Educational content; final diagnosis depends on the model.',
    diseaseName: 'Disease Name',
    diseaseSymptoms: 'Symptoms',
    diseaseCauses: 'Causes',
    diseaseInitialSolution: 'Initial Solutions',
    modelDisclaimer: 'Disclaimer: this content is educational; final diagnosis depends on the model result.',
    workerOnlyTasks: 'Worker-only Tasks',
    ownerTaskStatus: 'Owner Task Status',
    proofUpload: 'Upload Proof Image',
    proofNote: 'Completion Note',
    taskNotStarted: 'Not Started',
    taskInProgress: 'In Progress',
    taskCompleted: 'Done',
    startTask: 'Start Task',
    completeAndNotifyOwner: 'Complete and Notify Owner',
    serviceChat: 'Chat is available when the assistant service is ready',
    askRealAssistant: 'Ask real assistant',
    sensorUserHint: 'For users who have sensor values and want to send them to the model.',

    diagnosisCenter: 'Plant Diagnosis Center',
    diagnosisCenterSub: 'The core place to diagnose plants from images or sensor readings, then show results, recommendations, and history.',
    imageDiagnosis: 'Image Diagnosis',
    sensorDiagnosis: 'Sensor Diagnosis',
    combinedDiagnosis: 'Combined Diagnosis',
    diagnosisHistory: 'Diagnosis History',
    uploadOrConnect: 'Upload a plant image or use connected sensor readings to get a clear diagnosis.',
    mainProductFocus: 'The product is built around plant health diagnosis. Sectors are an extra service for farms, workers, and devices.',
    sectorsAsService: 'Farm & Sector Management',
    sectorsAsServiceSub: 'An extra service inside the platform for organizing farms: workers, tasks, devices, sensors, alerts, and sector follow-up.',
    floatingAssistant: 'Agriculture Assistant',
    assistantWelcome: 'I can explain diagnosis, recommend treatment, or help with irrigation and sensors.',
    openAssistant: 'Open Assistant',
    closeAssistant: 'Close Assistant',
    myDiagnosis: 'My Diagnoses',
    imageAndSensorResults: 'Image and sensor results',
    uploadedImageData: 'Uploaded image data',
    sensorValuesUsed: 'Sensor values used',
    explainToUser: 'Simple explanation',
    connectSensors: 'Connect Sensors',
    sensorDeviceId: 'Device ID',
    readingInterval: 'Reading Interval',
    everyFiveMinutes: 'Every 5 minutes',
    lastSync: 'Last Sync',
    connectedSensors: 'Connected Sensors',
    simulateFiveMinutes: 'Simulate 5-min update',
    sectorServiceNote: 'This section is for farm management only; it is not the main product idea.',
    farmWorkers: 'Farm Workers',
    farmTasks: 'Farm Tasks',
    sendTaskToWorker: 'Send Task to Worker',
    workerMobileNotification: 'Worker mobile/account notification',
    workerDone: 'Worker completed task',
    ownerNotifiedDone: 'Owner has been notified',
    completeTask: 'Mark Done',
    ownerView: 'Owner View',
    workerView: 'Worker View',
    taskSent: 'Task sent to worker.',
    taskDone: 'Task completed and owner notified.',
    dangerSolution: 'Suggested danger solution',
    alertWithSolution: 'Alert with solution',
    diseaseLibrary: 'Disease Library',
    possibleDisease: 'Possible Disease',
    symptom: 'Symptom',
    solution: 'Solution',
    smartWorkflow: 'Smart Workflow',
    stepUpload: 'Upload image or use sensors',
    stepAnalyze: 'Run diagnosis',
    stepAdvice: 'Get result and recommendations',
    stepSectorOptional: 'Optionally link to a sector or worker task',

    cleanDashboard: 'Clean Dashboard',
    dashboardFocus: 'A quick overview of plant health and alerts only. Sector details live inside the Sectors section.',
    sectorsHub: 'Sectors Hub',
    openSectorsHub: 'Open Sectors Hub',
    sectorManagement: 'Sector Management',
    sectorEverything: 'Everything related to sectors is here: map, workers, devices, tasks, irrigation, diagnoses, and notes.',
    sectorStats: 'Sector Statistics',
    totalSectors: 'Total Sectors',
    healthySectors: 'Healthy Sectors',
    warningSectors: 'Sectors to Monitor',
    dangerousSectors: 'Danger Sectors',

    adminPanel: 'Admin Panel',
    ownerDashboard: 'Owner Dashboard',
    workers: 'Workers',
    addWorker: 'Add Worker',
    workerName: 'Worker Name',
    workerRole: 'Worker Role',
    role: 'Role',
    owner: 'Owner',
    worker: 'Worker',
    permissions: 'Permissions',
    canView: 'View',
    canAdd: 'Add',
    canEdit: 'Edit',
    canDelete: 'Delete',
    assignedOnly: 'Shows assigned sectors only',
    allFarmAccess: 'Full farm access',
    taskBoard: 'Task Board',
    todo: 'To Do',
    inProgress: 'In Progress',
    done: 'Done',
    assignedTo: 'Assigned To',
    dueDate: 'Due Date',
    createTask: 'Create Task',
    sendWorkerAlert: 'Send Worker Alert',
    createTreatmentTask: 'Create Treatment Task',
    healthScoreCircle: 'Health Score',
    diseaseConfidence: 'Disease Confidence',
    beforeAfter: 'Before / After',
    realtime: 'Realtime',
    autoRefresh: 'Auto refresh every 30 seconds',
    websocketReady: 'WebSocket ready',
    loadingData: 'Loading data...',
    retry: 'Retry',
    unauthorized: 'Unauthorized',
    unauthorizedText: 'You do not have permission to view this data.',
    emptyState: 'No data yet',
    errorState: 'Error loading data',
    onboarding: 'Onboarding',
    farmName: 'Farm Name',
    firstSector: 'First Sector',
    firstDevice: 'First Device',
    enableAlerts: 'Enable Alerts',
    finishSetup: 'Finish Setup',
    skipNow: 'Skip Now',
    welcomeSetup: 'Let’s set up your smart farm in a few steps.',
    brandIdentity: 'Brand Identity',
    faviconReady: 'Logo and favicon placeholders are ready',
    splashScreen: 'Splash Screen',
    demoScenario: 'Demo Scenario',
    scenarioRegister: '1. Register account',
    scenarioSector: '2. Add a farm sector',
    scenarioDiagnosis: '3. Run plant diagnosis',
    scenarioAlert: '4. Alert appears on danger',
    scenarioTask: '5. Create treatment task',
    apiStatus: 'API Status',
    apiConnected: 'Connected',
    apiFallback: 'Demo fallback',
    endpoint: 'Service path',
    method: 'Method',
    authRequired: 'Auth required',
    tokenReady: 'Token ready',
    chatContext: 'Chat uses current sector and latest diagnosis context',
    explainDiagnosisPrompt: 'Explain this plant diagnosis',
    recommendTreatmentPrompt: 'Recommend treatment plan',
    wateringPrompt: 'Does this plant need water now?',
    workerAlertSent: 'Worker alert sent.',
    treatmentTaskCreated: 'Treatment task created.',
    setupSaved: 'Setup saved.',

    plantProfiles: 'Plant Profiles',
    plantProfile: 'Plant Profile',
    addPlant: 'Add Plant',
    plantName: 'Plant Name',
    plantAge: 'Plant Age',
    plantedAt: 'Planted Date',
    lastImage: 'Last Image',
    lastDiagnosis: 'Last Diagnosis',
    diagnosisTimeline: 'Diagnosis Timeline',
    linkedSector: 'Linked Sector',
    linkedSensors: 'Linked Sensors',
    saveDiagnosis: 'Save Diagnosis',
    downloadPdf: 'Download PDF Report',
    likelyCause: 'Likely Cause',
    treatmentPlan: 'Treatment Plan',
    nowAction: 'Immediately',
    next24h: 'Next 24 Hours',
    weeklyFollowup: 'Weekly Follow-up',
    settings: 'Settings',
    accountSettings: 'Account Settings',
    notificationSettings: 'Notification Settings',
    soundAlerts: 'Sound Alerts',
    temperatureUnit: 'Temperature Unit',
    celsius: 'Celsius',
    fahrenheit: 'Fahrenheit',
    profileName: 'Profile Name',
    saveSettings: 'Save Settings',
    markAsRead: 'Mark as read',
    markAllRead: 'Mark all as read',
    unread: 'Unread',
    read: 'Read',
    alertDisease: 'Disease Risk',
    alertMoisture: 'Low Moisture',
    alertHeat: 'High Heat',
    alertLight: 'Weak Light',
    alertDevice: 'Device Offline',
    previewRows: 'Preview Rows',
    columnMapping: 'Column Mapping',
    analyzeDataset: 'Analyze Dataset',
    datasetResults: 'Dataset Results',
    exportResults: 'Export Results',
    selectColumn: 'Select column',
    diagnosisChat: 'Diagnosis Chat',
    explainResult: 'Explain diagnosis',
    doesNeedWater: 'Does it need watering?',
    treatmentQuestion: 'What is the treatment plan?',
    userDataOnly: 'All displayed data belongs to your account only.',
    apiReady: '',
    apiServices: 'API Services',
    connectService: 'Your account is ready to send secure requests.',
    plantPhoto: 'Plant Photo',
    noPlants: 'No saved plants yet.',
    openProfile: 'Open Profile',
    riskReasonStable: 'Current readings are within a stable range.',
    riskReasonWaterHeat: 'Possible stress caused by low moisture or high temperature.',
    riskReasonCritical: 'High risk due to critical environmental conditions.',
    saveSuccess: 'Saved successfully.',
    soundOn: 'Sound enabled',
    soundOff: 'Sound disabled',
    datasetPreviewNote: 'Local preview; the analysis service can process all rows when available.',
    mappedTemperature: 'Temperature column',
    mappedHumidity: 'Humidity column',
    mappedSoilMoisture: 'Soil moisture column',
    mappedSoilTemp: 'Soil temperature column',
    mappedLight: 'Light column',
    mappedCrop: 'Crop column',
    batchHealthy: 'Healthy rows',
    batchModerate: 'Rows to monitor',
    batchHigh: 'High-risk rows',
    rowNumber: 'Row',

    farmView: 'Enter Farm View',
    enterSector: 'Enter Sector',
    fieldMap: 'Field Map',
    plantRows: 'Plant Rows',
    irrigationPlan: 'Irrigation Plan',
    nextIrrigation: 'Next Irrigation',
    careTasks: 'Care Tasks',
    addTask: 'Add Task',
    taskName: 'Task Name',
    priority: 'Priority',
    equipment: 'Equipment',
    notes: 'Sector Notes',
    addNote: 'Add Note',
    noteText: 'Note',
    sectorChat: 'Smart Agriculture Assistant',
    chat: 'Agriculture Chat',
    chatSubtitle: 'Ask about watering, diseases, sensors, diagnosis, or agriculture.',
    chatPlaceholder: 'Ask anything about agriculture or plant health...',
    send: 'Send',
    quickQuestions: 'Quick Questions',
    qWater: 'When should I water?',
    qYellow: 'Why are leaves yellow?',
    qFungus: 'How do I handle fungal spots?',
    qSensors: 'What sensors matter most?',
    botWelcome: 'Hi, I am your agriculture assistant. Ask me about plant health, watering, diseases, sensors, or diagnosis results.',
    rowHealthy: 'Healthy row',
    rowMonitor: 'Needs monitoring',
    rowDanger: 'Danger row',
    irrigationNormal: 'Normal irrigation within 24 hours',
    irrigationUrgent: 'Urgent gradual irrigation and moisture tracking',
    taskInspectLeaves: 'Inspect leaves',
    taskCheckIrrigation: 'Check irrigation system',
    taskCleanArea: 'Clean plant area',
    taskAdjustLight: 'Adjust light and ventilation',
    pump: 'Irrigation Pump',
    camera: 'Monitoring Camera',
    valve: 'Water Valve',
    controller: 'Controller',
    responsibleWorker: 'Responsible Worker',
    sectorEnvironment: 'Sector Environment',
    plantConditionMap: 'Plant Condition Map',
    liveCamera: 'Live Camera',
    cameraPreview: 'Camera Preview',
    createFullFarmExperience: 'Each sector now works like entering the farm: map, rows, workers, devices, irrigation, tasks, notes, and diagnosis.',

    workerAccountDataTitle: 'Worker Account Data',
    workerCredentialsEmpty: 'After creating a new worker account, only that worker credentials appear here, never the owner account.',
    date: 'Date',
    reportName: 'Report Name',
    viewFullReport: 'View Full Report',
    createdAt: 'Created At',
    assign: 'Assign',
    reassign: 'Reassign',
    rawDeveloperFieldsHidden: 'Technical response fields are hidden from the user interface and transformed into clear diagnosis language.'
  },
  ar: {
    brand: 'Smart Plant Health',
    tagline: 'تشخيص حالة النبات بالذكاء الاصطناعي',
    requireAccount: 'أنشئ حسابًا أولًا لبدء متابعة نباتاتك.',
    homeNotificationTitle: 'آخر تنبيه للنبات',
    noCriticalAlerts: 'لا توجد تنبيهات خطيرة حاليًا.',
    goRegisterFirst: 'سجّل للبدء',
    userFriendlyResult: 'ملخص تشخيص النبات',
    condition: 'حالة النبات',
    diseaseProblem: 'المرض / المشكلة المكتشفة',
    diagnosisSummary: 'ملخص التشخيص',
    recommendedCare: 'الرعاية المقترحة',
    immediateCare: 'إجراءات فورية',
    sentReadings: 'القراءات المستخدمة',
    imageIndicators: 'مؤشرات الصورة',
    automaticReadings: 'القراءات يتم تحميلها تلقائيًا من الحساسات أو تغذية الموديل المتصلة.',
    refreshReadings: 'تحديث القراءات',
    autoSensorFeed: 'تغذية الحساسات التلقائية',
    farmTitle: 'متابعة المزرعة',
    farmSubtitle: 'تابع المزرعة كاملة من خلال القطاعات. كل قطاع يحتوي على حساساته وتشخيصاته وتنبيهاته وآخر قراءة ذكاء اصطناعي.',
    sectorHealth: 'صحة القطاع',
    latestDiagnosis: 'آخر تشخيص',
    sensorSnapshot: 'ملخص الحساسات',
    highRiskWarning: 'تنبيه خطر: إحدى النباتات في حالة خطيرة وتحتاج تدخلًا سريعًا.',
    notifications: 'الإشعارات',
    datasetAnalyzer: 'تحليل ملف بيانات',
    datasetSubtitle: 'ارفع ملف CSV أو Excel لتجهيزه لتحليل حالة النباتات عبر الخدمة.',
    uploadDataset: 'رفع ملف بيانات',
    chooseDataset: 'اختر ملف CSV أو Excel',
    datasetReady: 'تم اختيار الملف وهو جاهز للإرسال للخدمة.',
    datasetName: 'اسم الملف',
    rowsEstimate: 'عدد الصفوف المتوقع',
    sendDataset: 'إرسال الملف',
    datasetNote: 'عند ربط الخدمة، هذا القسم يرفع داتا سيت كاملة ويعرض نتائج تشخيص النباتات دفعة واحدة.',
    privateRoutesNote: 'لوحة التحكم والتشخيص والقطاعات والتقارير والتنبيهات متاحة بعد التسجيل أو تسجيل الدخول.',
    accountCreated: 'تم إنشاء الحساب بنجاح.',
    loggedIn: 'تم تسجيل الدخول بنجاح.',
    wholeFarm: 'المزرعة كاملة',
    critical: 'خطر',
    safe: 'آمن',
    monitor: 'متابعة',
    uploadPlantImageOptional: 'يمكن رفع صورة للنبات اختياريًا. قيم الحساسات يتم توفيرها تلقائيًا.',
    smartSummary: 'ملخص ذكي',
    rawDeveloperFieldsHidden: 'تم إخفاء أسماء الحقول التقنية من واجهة المستخدم وتحويلها إلى لغة تشخيص واضحة.',
    modelNotificationBar: 'تنبيهات الموديل',
    noModelAlerts: 'لا توجد تنبيهات من الموديل لهذه النتيجة.',
    compactDiagnosisView: 'عرض مختصر وواضح لنتيجة التشخيص بدون ازدحام.',
    diagnosisReports: 'تقارير التشخيص',
    reportsPurpose: 'التقارير هنا هدفها حفظ سجل التشخيصات وقراءات الحساسات وتصديرها كملف للمناقشة أو المتابعة.',
    libraryPurpose: 'مكتبة الأمراض جزء تعليمي فقط يشرح أشهر مشاكل النبات، وليست بديلة عن نتيجة الموديل.',
    farmManagementTitle: 'إدارة المزرعة',
    farmManagementSub: 'مكان واحد منظم لإدارة القطاعات والعاملين والأجهزة والمهام والصلاحيات بدون ازدحام.',
    farmControlCenter: 'مركز التحكم بالمزرعة',
    farmControlHint: 'اختار الخدمة التي تريد إدارتها من التبويبات بالأسفل.',
    diagnosisReportSimple: 'ملخصات التشخيص',
    diseaseGuide: 'دليل أمراض النبات'
  },
  fr: {
    brand: 'Smart Plant Health', requireAccount: 'Créez un compte pour commencer.', homeNotificationTitle: 'Dernière alerte plante', noCriticalAlerts: 'Aucune alerte critique.', goRegisterFirst: 'Créer un compte', userFriendlyResult: 'Résumé du diagnostic', condition: 'État de la plante', diseaseProblem: 'Maladie / problème détecté', diagnosisSummary: 'Résumé', recommendedCare: 'Soins recommandés', immediateCare: 'Actions immédiates', sentReadings: 'Lectures utilisées', imageIndicators: 'Indicateurs image', automaticReadings: 'Les lectures sont chargées automatiquement.', refreshReadings: 'Actualiser', autoSensorFeed: 'Flux automatique', farmTitle: 'Suivi de la ferme', farmSubtitle: 'Suivez toute la ferme par secteurs.', sectorHealth: 'Santé du secteur', latestDiagnosis: 'Dernier diagnostic', sensorSnapshot: 'Capteurs', highRiskWarning: 'Alerte: une plante est en danger.', notifications: 'Notifications', datasetAnalyzer: 'Analyse de dataset', uploadDataset: 'Importer dataset', chooseDataset: 'Choisir CSV ou Excel', datasetReady: 'Fichier prêt.', sendDataset: 'Envoyer', privateRoutesNote: 'Inscription requise.', rawDeveloperFieldsHidden: 'Les champs techniques sont masqués.'
  },
  es: {
    brand: 'Smart Plant Health', requireAccount: 'Crea una cuenta para empezar.', homeNotificationTitle: 'Última alerta', noCriticalAlerts: 'Sin alertas críticas.', goRegisterFirst: 'Registrarse', userFriendlyResult: 'Resumen del diagnóstico', condition: 'Estado de la planta', diseaseProblem: 'Enfermedad / problema', diagnosisSummary: 'Resumen', recommendedCare: 'Cuidados recomendados', immediateCare: 'Acciones inmediatas', sentReadings: 'Lecturas usadas', imageIndicators: 'Indicadores de imagen', automaticReadings: 'Las lecturas se cargan automáticamente.', refreshReadings: 'Actualizar', autoSensorFeed: 'Flujo automático', farmTitle: 'Monitoreo de la granja', farmSubtitle: 'Controla toda la granja por sectores.', sectorHealth: 'Salud del sector', latestDiagnosis: 'Último diagnóstico', sensorSnapshot: 'Sensores', highRiskWarning: 'Alerta: una planta está en peligro.', notifications: 'Notificaciones', datasetAnalyzer: 'Analizador de dataset', uploadDataset: 'Subir dataset', chooseDataset: 'Elegir CSV o Excel', datasetReady: 'Archivo listo.', sendDataset: 'Enviar', privateRoutesNote: 'Registro requerido.', rawDeveloperFieldsHidden: 'Los campos técnicos están ocultos.'
  },
  de: {
    brand: 'Smart Plant Health', requireAccount: 'Erstellen Sie ein Konto, um zu beginnen.', homeNotificationTitle: 'Letzte Pflanzenwarnung', noCriticalAlerts: 'Keine kritischen Warnungen.', goRegisterFirst: 'Registrieren', userFriendlyResult: 'Diagnoseübersicht', condition: 'Pflanzenzustand', diseaseProblem: 'Krankheit / Problem', diagnosisSummary: 'Zusammenfassung', recommendedCare: 'Empfohlene Pflege', immediateCare: 'Sofortmaßnahmen', sentReadings: 'Verwendete Werte', imageIndicators: 'Bildindikatoren', automaticReadings: 'Messwerte werden automatisch geladen.', refreshReadings: 'Aktualisieren', autoSensorFeed: 'Automatischer Feed', farmTitle: 'Farmüberwachung', farmSubtitle: 'Überwachen Sie die ganze Farm nach Sektoren.', sectorHealth: 'Sektorgesundheit', latestDiagnosis: 'Letzte Diagnose', sensorSnapshot: 'Sensoren', highRiskWarning: 'Warnung: Eine Pflanze ist gefährdet.', notifications: 'Benachrichtigungen', datasetAnalyzer: 'Dataset-Analyse', uploadDataset: 'Dataset hochladen', chooseDataset: 'CSV oder Excel wählen', datasetReady: 'Datei bereit.', sendDataset: 'Senden', privateRoutesNote: 'Registrierung erforderlich.', rawDeveloperFieldsHidden: 'Technische Felder werden ausgeblendet.'
  },
  it: {
    brand: 'Smart Plant Health', requireAccount: 'Crea un account per iniziare.', homeNotificationTitle: 'Ultimo avviso pianta', noCriticalAlerts: 'Nessun avviso critico.', goRegisterFirst: 'Registrati', userFriendlyResult: 'Riepilogo diagnosi', condition: 'Stato pianta', diseaseProblem: 'Malattia / problema', diagnosisSummary: 'Riepilogo', recommendedCare: 'Cura consigliata', immediateCare: 'Azioni immediate', sentReadings: 'Letture usate', imageIndicators: 'Indicatori immagine', automaticReadings: 'Le letture sono caricate automaticamente.', refreshReadings: 'Aggiorna', autoSensorFeed: 'Feed automatico', farmTitle: 'Monitoraggio fattoria', farmSubtitle: 'Monitora la fattoria per settori.', sectorHealth: 'Salute settore', latestDiagnosis: 'Ultima diagnosi', sensorSnapshot: 'Sensori', highRiskWarning: 'Allerta: una pianta è a rischio.', notifications: 'Notifiche', datasetAnalyzer: 'Analisi dataset', uploadDataset: 'Carica dataset', chooseDataset: 'Scegli CSV o Excel', datasetReady: 'File pronto.', sendDataset: 'Invia', privateRoutesNote: 'Registrazione richiesta.', rawDeveloperFieldsHidden: 'I campi tecnici sono nascosti.'
  },
  tr: {
    brand: 'Smart Plant Health', requireAccount: 'Başlamak için hesap oluşturun.', homeNotificationTitle: 'Son bitki uyarısı', noCriticalAlerts: 'Kritik uyarı yok.', goRegisterFirst: 'Kayıt ol', userFriendlyResult: 'Teşhis özeti', condition: 'Bitki durumu', diseaseProblem: 'Hastalık / sorun', diagnosisSummary: 'Özet', recommendedCare: 'Önerilen bakım', immediateCare: 'Acil işlemler', sentReadings: 'Kullanılan ölçümler', imageIndicators: 'Görüntü göstergeleri', automaticReadings: 'Ölçümler otomatik yüklenir.', refreshReadings: 'Yenile', autoSensorFeed: 'Otomatik sensör akışı', farmTitle: 'Çiftlik izleme', farmSubtitle: 'Çiftliği sektörlerle izleyin.', sectorHealth: 'Sektör sağlığı', latestDiagnosis: 'Son teşhis', sensorSnapshot: 'Sensörler', highRiskWarning: 'Uyarı: bir bitki risk altında.', notifications: 'Bildirimler', datasetAnalyzer: 'Veri seti analizi', uploadDataset: 'Veri seti yükle', chooseDataset: 'CSV veya Excel seç', datasetReady: 'Dosya hazır.', sendDataset: 'Gönder', privateRoutesNote: 'Kayıt gerekli.', rawDeveloperFieldsHidden: 'Teknik alanlar gizlenir.'
  }
};


const arText = {
  brand: 'Smart Plant Health',
  tagline: 'تشخيص حالة النبات بالذكاء الاصطناعي',
  home: 'الرئيسية', dashboard: 'لوحة التحكم', diagnosis: 'تشخيص النبات', sensors: 'الحساسات',
  sectors: 'القطاعات', reports: 'التقارير', alerts: 'التنبيهات', workers: 'العاملون',
  devices: 'الأجهزة', images: 'الصور', login: 'تسجيل الدخول', signup: 'إنشاء حساب', logout: 'تسجيل الخروج',
  heroBadge: 'تشخيص ذكي بالصور والحساسات',
  heroTitle: 'تابع صحة نباتاتك واكتشف الخطر قبل انتشاره.',
  heroDesc: 'منصة Smart Plant Health تساعدك على تشخيص حالة النبات من صورة النبات وقراءات الحساسات، وتعرض لك حالة النبات، المشكلة المحتملة، والتنبيهات والتوصيات بطريقة واضحة.',
  startDiagnosis: 'ابدأ التشخيص', viewDashboard: 'افتح لوحة التحكم', goRegisterFirst: 'أنشئ حساب للبدء',
  requireAccount: 'أنشئ حسابًا أولًا لبدء متابعة نباتاتك.',
  productNotGrad: 'منصة ذكية لتشخيص ومتابعة صحة النباتات.',
  publicProduct: 'منتج عام', multiLang: 'متعدد اللغات', darkLight: 'ليلي ونهاري', mobileReady: 'متوافق مع الهاتف',
  core1: 'تشخيص واضح', core1Text: 'يعرض حالة النبات واسم المشكلة والتوصيات بدون مصطلحات تقنية معقدة.',
  core2: 'صورة + حساسات', core2Text: 'يعتمد على صورة النبات وقراءات الحرارة والرطوبة ورطوبة التربة والإضاءة.',
  core3: 'تنبيهات ذكية', core3Text: 'يرسل تنبيهًا فوريًا عند وجود خطر على النبات.',
  condition: 'حالة النبات', diseaseProblem: 'المرض / المشكلة المكتشفة',
  plantDiagnosis: 'تشخيص حالة النبات', diagnosisSubtitle: 'ارفع صورة النبات واستخدم قراءات الحساسات التلقائية لمعرفة الحالة.',
  uploadPlantImageOptional: 'يمكنك رفع صورة للنبات، وقراءات الحساسات يتم تحميلها تلقائيًا.',
  autoSensorFeed: 'تغذية الحساسات التلقائية', automaticReadings: 'القراءات يتم تحميلها تلقائيًا من الحساسات المتصلة.',
  refreshReadings: 'تحديث القراءات', refreshSensors: 'تحديث الحساسات', simulateReadings: 'محاكاة قراءة', analyze: 'تحليل النبات', analyzing: 'جاري التحليل...',
  uploadImage: 'رفع صورة النبات', chooseImage: 'اختر أو اسحب صورة النبات',
  userFriendlyResult: 'ملخص تشخيص النبات', confidence: 'نسبة الثقة', severity: 'درجة الخطورة',
  sensorSnapshot: 'ملخص الحساسات', imageIndicators: 'مؤشرات الصورة', diagnosisSummary: 'ملخص التشخيص',
  recommendedCare: 'الرعاية المقترحة', immediateCare: 'إجراءات فورية', sentReadings: 'القراءات المستخدمة',
  noImage: 'لا توجد صورة', noDiseaseDetected: 'لا يوجد مرض واضح', rawDeveloperFieldsHidden: 'تم تحويل استجابة الموديل إلى نتيجة واضحة للمستخدم.',
  modelNotificationBar: 'تنبيهات الموديل',
  noModelAlerts: 'لا توجد تنبيهات من الموديل لهذه النتيجة.',
  compactDiagnosisView: 'عرض مختصر وواضح لنتيجة التشخيص بدون ازدحام.',
  diagnosisReports: 'تقارير التشخيص',
  reportsPurpose: 'التقارير هنا هدفها حفظ سجل التشخيصات وقراءات الحساسات وتصديرها كملف للمناقشة أو المتابعة.',
  libraryPurpose: 'مكتبة الأمراض جزء تعليمي فقط يشرح أشهر مشاكل النبات، وليست بديلة عن نتيجة الموديل.',
  farmManagementTitle: 'إدارة المزرعة',
  farmManagementSub: 'مكان واحد منظم لإدارة القطاعات والعاملين والأجهزة والمهام والصلاحيات بدون ازدحام.',
  farmControlCenter: 'مركز التحكم بالمزرعة',
  farmControlHint: 'اختار الخدمة التي تريد إدارتها من التبويبات بالأسفل.',
  diagnosisReportSimple: 'ملخصات التشخيص',
  diseaseGuide: 'دليل أمراض النبات',
  visualProblem: 'المشكلة البصرية', healthScore: 'درجة الصحة',
  greenRatio: 'النسيج الأخضر', yellowRatio: 'الاصفرار', brownRatio: 'النسيج البني', darkSpotRatio: 'البقع الداكنة', damagedRatio: 'التلف الظاهري',
  cropType: 'نوع النبات', crop: 'المحصول', tomato: 'طماطم', corn: 'ذرة', pepper: 'فلفل', mint: 'نعناع',
  temperature: 'حرارة الجو', humidity: 'الرطوبة', soilMoisture: 'رطوبة التربة', soilTemp: 'حرارة التربة', light: 'الإضاءة',
  low: 'منخفضة', medium: 'متوسطة', sufficient: 'كافية', healthy: 'سليم', moderate: 'إجهاد متوسط', high: 'خطر مرتفع',
  dashboardTitle: 'نظرة عامة على صحة النباتات', dashboardSubtitle: 'تابع حالة النباتات، الحساسات، القطاعات، والتنبيهات.',
  liveReadings: 'القراءات المباشرة', latestReading: 'آخر قراءة', statistics: 'الإحصائيات', history: 'السجل',
  smartSummary: 'ملخص ذكي', wholeFarm: 'المزرعة كاملة',
  notifications: 'الإشعارات', homeNotificationTitle: 'آخر تنبيه للنبات', noCriticalAlerts: 'لا توجد تنبيهات خطيرة حاليًا.',
  alertsTitle: 'تنبيهات النبات', alertsSubtitle: 'تنبيهات يتم إنشاؤها عند وجود خطر أو إجهاد في النبات.',
  highRiskWarning: 'تنبيه خطر: إحدى النباتات في حالة خطيرة وتحتاج تدخلًا سريعًا.',
  farmTitle: 'متابعة المزرعة', farmSubtitle: 'تابع المزرعة من خلال القطاعات. كل قطاع يحتوي على حساساته وتشخيصاته والعاملين المسؤولين عنه.',
  addSector: 'إضافة قطاع', sectorName: 'اسم القطاع', location: 'الموقع', save: 'حفظ', delete: 'حذف',
  sectorHealth: 'صحة القطاع', latestDiagnosis: 'آخر تشخيص', viewDetails: 'عرض التفاصيل', backToSectors: 'العودة للقطاعات',
  sectorDetails: 'تفاصيل القطاع', sectorOverview: 'نظرة عامة على القطاع', assignedWorkers: 'العاملون المسؤولون',
  sectorSensors: 'حساسات القطاع', diagnosisHistory: 'سجل التشخيصات', addWorker: 'إضافة عامل', workerName: 'اسم العامل',
  workerRole: 'دور العامل', workerPhone: 'رقم الهاتف', addSensor: 'إضافة حساس', sensorName: 'اسم الحساس',
  sensorType: 'نوع الحساس', sensorValue: 'قيمة الحساس', sensorUnit: 'الوحدة', status: 'الحالة',
  reportsTitle: 'التقارير', reportsSubtitle: 'راجع اتجاهات صحة النباتات وارفع ملفات بيانات للتحليل.',
  exportData: 'تصدير البيانات', datasetAnalyzer: 'تحليل ملف بيانات', datasetNote: 'ارفع ملف CSV أو Excel ليتم تحليله عند ربط الخدمة.',
  chooseDataset: 'اختر ملف CSV أو Excel', datasetName: 'اسم الملف', rowsEstimate: 'عدد الصفوف المتوقع', sendDataset: 'إرسال الملف',
  email: 'البريد الإلكتروني', password: 'كلمة المرور', fullName: 'الاسم الكامل', phone: 'رقم الهاتف',
  confirmPassword: 'تأكيد كلمة المرور', remember: 'تذكرني', forgot: 'نسيت كلمة المرور؟', createAccount: 'إنشاء حساب',
  alreadyHave: 'لديك حساب بالفعل؟', noAccount: 'ليس لديك حساب؟', continueGoogle: 'المتابعة باستخدام جوجل',
  authSubtitle: 'ادخل إلى لوحة المتابعة، التشخيصات، القطاعات، التنبيهات، وتحليل الحساسات.',
  privateRoutesNote: 'يجب إنشاء حساب أو تسجيل الدخول لاستخدام النظام الداخلي.',
  accountCreated: 'تم إنشاء الحساب بنجاح.', loggedIn: 'تم تسجيل الدخول بنجاح.',
  language: 'اللغة', theme: 'المظهر', roleText: 'تشخيص ذكي ومتابعة مستمرة لصحة النبات.',
  footer: '© 2026 Smart Plant Health — منصة ذكية لتشخيص صحة النبات.',
  safe: 'آمن', monitor: 'متابعة', critical: 'خطر', or: 'أو', time: 'الوقت',
  datasetReady: 'تم اختيار الملف وهو جاهز للإرسال.',
  
  farmView: 'الدخول إلى المزرعة',
  enterSector: 'دخول القطاع',
  fieldMap: 'خريطة الأرض',
  plantRows: 'صفوف الزراعة',
  irrigationPlan: 'خطة الري',
  nextIrrigation: 'موعد الري القادم',
  careTasks: 'مهام الرعاية',
  addTask: 'إضافة مهمة',
  taskName: 'اسم المهمة',
  priority: 'الأولوية',
  highPriority: 'عالية',
  mediumPriority: 'متوسطة',
  lowPriority: 'منخفضة',
  equipment: 'المعدات',
  notes: 'ملاحظات القطاع',
  addNote: 'إضافة ملاحظة',
  noteText: 'نص الملاحظة',
  sectorChat: 'مساعد الزراعة الذكي',
  chat: 'الشات الزراعي',
  chatSubtitle: 'اسأل عن الري، الأمراض، الحساسات، التشخيص، أو أي شيء يخص الزراعة.',
  chatPlaceholder: 'اكتب سؤالك عن الزراعة أو حالة النبات...',
  send: 'إرسال',
  quickQuestions: 'أسئلة سريعة',
  qWater: 'متى أروي النبات؟',
  qYellow: 'ما سبب اصفرار الأوراق؟',
  qFungus: 'كيف أتعامل مع البقع الفطرية؟',
  qSensors: 'ما أهم الحساسات لمتابعة النبات؟',
  botWelcome: 'مرحبًا، أنا مساعدك الزراعي. اسألني عن صحة النبات، الري، الأمراض، الحساسات، أو نتائج التشخيص.',
  rowHealthy: 'صف سليم',
  rowMonitor: 'صف يحتاج متابعة',
  rowDanger: 'صف في خطر',
  irrigationNormal: 'ري طبيعي خلال 24 ساعة',
  irrigationUrgent: 'ري تدريجي عاجل ومتابعة الرطوبة',
  taskInspectLeaves: 'فحص الأوراق',
  taskCheckIrrigation: 'مراجعة نظام الري',
  taskCleanArea: 'تنظيف المنطقة حول النبات',
  taskAdjustLight: 'ضبط الإضاءة والتهوية',
  pump: 'مضخة ري',
  camera: 'كاميرا مراقبة',
  valve: 'صمام مياه',
  controller: 'وحدة تحكم',
  responsibleWorker: 'العامل المسؤول',
  sectorEnvironment: 'بيئة القطاع',
  plantConditionMap: 'خريطة حالة النباتات',
  liveCamera: 'كاميرا مباشرة',
  cameraPreview: 'معاينة الكاميرا',
  createFullFarmExperience: 'تم تحويل القطاع إلى مساحة متابعة كاملة كأنك داخل الأرض: خريطة، صفوف، عمال، أجهزة، ري، مهام، ملاحظات، وتشخيص.',

  workerAccountDataTitle: 'بيانات حساب العامل',
  workerCredentialsEmpty: 'بعد إنشاء حساب عامل جديد ستظهر بياناته هنا فقط، وليست بيانات المالك.',
  date: 'التاريخ',
  reportName: 'اسم التقرير',
  viewFullReport: 'عرض التقرير',
  createdAt: 'وقت الإنشاء',
  assign: 'تعيين',
  reassign: 'إعادة تعيين',

  plantProfiles: 'ملفات النباتات',
  plantProfile: 'ملف النبات',
  addPlant: 'إضافة نبات',
  plantName: 'اسم النبات',
  plantAge: 'عمر النبات',
  plantedAt: 'تاريخ الزراعة',
  lastImage: 'آخر صورة',
  lastDiagnosis: 'آخر تشخيص',
  diagnosisTimeline: 'سجل التشخيصات',
  linkedSector: 'القطاع المرتبط',
  linkedSensors: 'الحساسات المرتبطة',
  saveDiagnosis: 'حفظ التشخيص',
  downloadPdf: 'تحميل تقرير PDF',
  likelyCause: 'السبب المحتمل',
  treatmentPlan: 'خطة العلاج',
  nowAction: 'فوريًا',
  next24h: 'خلال 24 ساعة',
  weeklyFollowup: 'متابعة أسبوعية',
  settings: 'الإعدادات',
  accountSettings: 'إعدادات الحساب',
  notificationSettings: 'إعدادات التنبيهات',
  soundAlerts: 'صوت التنبيهات',
  temperatureUnit: 'وحدة الحرارة',
  celsius: 'مئوية',
  fahrenheit: 'فهرنهايت',
  profileName: 'اسم المستخدم',
  saveSettings: 'حفظ الإعدادات',
  markAsRead: 'تحديد كمقروء',
  markAllRead: 'تحديد الكل كمقروء',
  unread: 'غير مقروء',
  read: 'مقروء',
  alertDisease: 'خطر مرض',
  alertMoisture: 'رطوبة منخفضة',
  alertHeat: 'حرارة عالية',
  alertLight: 'إضاءة ضعيفة',
  alertDevice: 'حساس غير متصل',
  previewRows: 'معاينة أول الصفوف',
  columnMapping: 'مطابقة الأعمدة',
  analyzeDataset: 'تحليل ملف البيانات',
  datasetResults: 'نتائج تحليل البيانات',
  exportResults: 'تصدير النتائج',
  selectColumn: 'اختر العمود',
  diagnosisChat: 'شات التشخيص',
  explainResult: 'اشرح نتيجة التشخيص',
  doesNeedWater: 'هل النبات يحتاج ري؟',
  treatmentQuestion: 'ما الخطة المناسبة للعلاج؟',
  userDataOnly: 'كل البيانات المعروضة خاصة بحسابك فقط.',
  apiReady: '',
  apiServices: 'خدمات API',
  connectService: 'حسابك جاهز لإرسال الطلبات بشكل آمن.',
  plantPhoto: 'صورة النبات',
  noPlants: 'لا توجد نباتات محفوظة بعد.',
  openProfile: 'فتح الملف',
  riskReasonStable: 'القراءات الحالية ضمن نطاق مستقر.',
  riskReasonWaterHeat: 'يوجد احتمال إجهاد بسبب انخفاض الرطوبة أو ارتفاع الحرارة.',
  riskReasonCritical: 'الخطر مرتفع بسبب ظروف بيئية حرجة وقد يحتاج النبات لتدخل عاجل.',
  saveSuccess: 'تم الحفظ بنجاح.',
  soundOn: 'الصوت مفعل',
  soundOff: 'الصوت مغلق',
  datasetPreviewNote: 'هذه معاينة محلية؛ عند ربط الخدمة سيتم إرسال الملف وتحليل كل الصفوف.',
  mappedTemperature: 'عمود الحرارة',
  mappedHumidity: 'عمود الرطوبة',
  mappedSoilMoisture: 'عمود رطوبة التربة',
  mappedSoilTemp: 'عمود حرارة التربة',
  mappedLight: 'عمود الإضاءة',
  mappedCrop: 'عمود نوع النبات',
  batchHealthy: 'صفوف سليمة',
  batchModerate: 'صفوف تحتاج متابعة',
  batchHigh: 'صفوف في خطر',
  rowNumber: 'رقم الصف',


  adminPanel: 'لوحة الإدارة',
  ownerDashboard: 'لوحة المالك',
  role: 'الدور',
  owner: 'مالك',
  worker: 'عامل',
  permissions: 'الصلاحيات',
  canView: 'عرض',
  canAdd: 'إضافة',
  canEdit: 'تعديل',
  canDelete: 'حذف',
  assignedOnly: 'يعرض القطاعات المكلف بها فقط',
  allFarmAccess: 'صلاحية كاملة على المزرعة',
  taskBoard: 'لوحة المهام',
  todo: 'قيد التنفيذ',
  inProgress: 'جاري العمل',
  done: 'تم',
  assignedTo: 'مسند إلى',
  dueDate: 'تاريخ الانتهاء',
  createTask: 'إنشاء مهمة',
  sendWorkerAlert: 'إرسال تنبيه للعامل',
  createTreatmentTask: 'إنشاء مهمة علاج تلقائيًا',
  healthScoreCircle: 'مؤشر صحة النبات',
  diseaseConfidence: 'ثقة المرض',
  beforeAfter: 'قبل / بعد',
  realtime: 'تحديث مباشر',
  autoRefresh: 'تحديث تلقائي كل 30 ثانية',
  websocketReady: 'جاهز للربط بـ WebSocket',
  loadingData: 'جاري تحميل البيانات...',
  retry: 'إعادة المحاولة',
  unauthorized: 'غير مصرح بالدخول',
  unauthorizedText: 'ليس لديك صلاحية لعرض هذه البيانات.',
  emptyState: 'لا توجد بيانات بعد',
  errorState: 'حدث خطأ أثناء تحميل البيانات',
  onboarding: 'التهيئة الأولى',
  farmName: 'اسم المزرعة',
  firstSector: 'أول قطاع',
  firstDevice: 'أول جهاز',
  enableAlerts: 'تفعيل التنبيهات',
  finishSetup: 'إنهاء الإعداد',
  skipNow: 'تخطي الآن',
  welcomeSetup: 'لنجهز مزرعتك الذكية في خطوات بسيطة.',
  brandIdentity: 'هوية الموقع',
  faviconReady: 'تم تجهيز مكان favicon واللوجو',
  splashScreen: 'شاشة بداية',
  demoScenario: 'سيناريو العرض',
  scenarioRegister: '١. تسجيل حساب جديد',
  scenarioSector: '٢. إضافة قطاع داخل المزرعة',
  scenarioDiagnosis: '٣. تشغيل تشخيص النبات',
  scenarioAlert: '٤. ظهور تنبيه عند الخطر',
  scenarioTask: '٥. إنشاء مهمة علاج للعامل',
  apiStatus: 'حالة الربط',
  apiConnected: 'متصل',
  apiFallback: 'وضع تجريبي',
  endpoint: 'مسار الخدمة',
  method: 'الطريقة',
  authRequired: 'يتطلب تسجيل دخول',
  tokenReady: 'التوكن جاهز للإرسال',
  chatContext: 'الشات يستخدم سياق آخر تشخيص والقطاع الحالي',
  explainDiagnosisPrompt: 'اشرح نتيجة التشخيص لهذا النبات',
  recommendTreatmentPrompt: 'اقترح خطة علاج مناسبة',
  wateringPrompt: 'هل يحتاج النبات إلى ري الآن؟',
  workerAlertSent: 'تم إرسال تنبيه للعامل.',
  treatmentTaskCreated: 'تم إنشاء مهمة علاج تلقائيًا.',
  setupSaved: 'تم حفظ إعدادات البداية.',


  cleanDashboard: 'لوحة مختصرة بدون ازدحام',
  dashboardFocus: 'نظرة سريعة على صحة النباتات والتنبيهات فقط. تفاصيل القطاعات موجودة داخل قسم القطاعات.',
  sectorsHub: 'مركز القطاعات',
  openSectorsHub: 'فتح مركز القطاعات',
  sectorManagement: 'إدارة القطاعات',
  sectorEverything: 'كل ما يخص القطاعات موجود هنا: الخريطة، العمال، الأجهزة، المهام، الري، التشخيصات والملاحظات.',
  sectorStats: 'إحصائيات القطاعات',
  totalSectors: 'إجمالي القطاعات',
  healthySectors: 'قطاعات سليمة',
  warningSectors: 'قطاعات تحتاج متابعة',
  dangerousSectors: 'قطاعات في خطر',


  diagnosisCenter: 'مركز تشخيص حالة النبات',
  diagnosisCenterSub: 'المكان الأساسي لتشخيص النبات من الصورة أو قراءات الحساسات، وعرض النتيجة والتوصيات وسجل التشخيص.',
  imageDiagnosis: 'تشخيص بالصورة',
  sensorDiagnosis: 'تشخيص بالحساسات',
  combinedDiagnosis: 'تشخيص مدمج',
  uploadOrConnect: 'ارفع صورة النبات أو استخدم قراءات الحساسات المتصلة للحصول على تشخيص واضح.',
  mainProductFocus: 'الموقع قائم على تشخيص حالة النبات. إدارة القطاعات خدمة إضافية لتنظيم المزارع والعمال والأجهزة.',
  sectorsAsService: 'إدارة المزرعة والقطاعات',
  sectorsAsServiceSub: 'خدمة إضافية داخل الموقع لتنظيم المزرعة: عمال، مهام، أجهزة، حساسات، تنبيهات، ومتابعة كل قطاع.',
  floatingAssistant: 'المساعد الزراعي',
  assistantWelcome: 'أنا مساعدك الزراعي. أقدر أشرح التشخيص، أقترح علاج، أو أساعدك في الري والحساسات.',
  openAssistant: 'فتح المساعد',
  closeAssistant: 'إغلاق المساعد',
  myDiagnosis: 'تشخيصاتي',
  imageAndSensorResults: 'نتائج الصورة والحساسات',
  uploadedImageData: 'بيانات الصورة المرفوعة',
  sensorValuesUsed: 'قيم الحساسات المستخدمة',
  explainToUser: 'شرح مبسط للمستخدم',
  connectSensors: 'ربط الحساسات',
  sensorDeviceId: 'معرف الجهاز',
  readingInterval: 'تحديث القراءات',
  everyFiveMinutes: 'كل 5 دقائق',
  lastSync: 'آخر مزامنة',
  connectedSensors: 'الحساسات المتصلة',
  simulateFiveMinutes: 'محاكاة تحديث 5 دقائق',
  sectorServiceNote: 'هذا القسم مخصص لإدارة المزرعة فقط، وليس هو فكرة الموقع الأساسية.',
  farmWorkers: 'عمال المزرعة',
  farmTasks: 'مهام المزرعة',
  sendTaskToWorker: 'إرسال مهمة للعامل',
  workerMobileNotification: 'إشعار يصل للعامل على حسابه/الموبايل',
  workerDone: 'العامل أنهى المهمة',
  ownerNotifiedDone: 'تم إشعار المالك بانتهاء المهمة',
  completeTask: 'تم التنفيذ',
  ownerView: 'عرض المالك',
  workerView: 'عرض العامل',
  taskSent: 'تم إرسال المهمة للعامل.',
  taskDone: 'تم إنهاء المهمة وإشعار المالك.',
  dangerSolution: 'الحل المقترح للخطر',
  alertWithSolution: 'تنبيه مع الحل',
  diseaseLibrary: 'مكتبة الأمراض',
  possibleDisease: 'مرض محتمل',
  symptom: 'العرض',
  solution: 'الحل',
  smartWorkflow: 'تدفق العمل الذكي',
  stepUpload: 'ارفع صورة أو استخدم الحساسات',
  stepAnalyze: 'شغّل التشخيص',
  stepAdvice: 'احصل على النتيجة والتوصيات',
  stepSectorOptional: 'اختياريًا: اربط التشخيص بقطاع أو مهمة عامل',


  diagnosisMode: 'طريقة التشخيص',
  imageOnlyMode: 'صورة فقط',
  sensorsOnlyMode: 'حساسات فقط',
  combinedMode: 'صورة + حساسات',
  modelDrivenNote: 'حلل صورة النبات أو قراءات الحساسات من خلال الخدمات المتصلة واعرض نتيجة واضحة وتوصيات عملية.',
  manualSensorInput: 'إدخال قيم الحساسات',
  sendToModel: 'إرسال للموديل',
  suspicionAreas: 'مناطق الاشتباه في الصورة',
  modelAlerts: 'تنبيهات الموديل',
  alarmAlert: 'إنذار خطر',
  linkDiagnosisToSector: 'ربط التشخيص بسيكتور',
  selectSector: 'اختر سيكتور',
  linkedToSector: 'تم ربط التشخيص بالسيكتور.',
  modelLikelyCause: 'السبب المحتمل من الموديل',
  plantLibrary: 'مكتبة أمراض النبات',
  plantLibrarySub: 'محتوى تعليمي يساعد المستخدم، مع التأكيد أن التشخيص النهائي يأتي من الموديل.',
  diseaseName: 'اسم المرض',
  diseaseSymptoms: 'الأعراض',
  diseaseCauses: 'الأسباب',
  diseaseInitialSolution: 'حلول أولية',
  modelDisclaimer: 'تنبيه: هذه المعلومات تعليمية، والتشخيص النهائي يعتمد على نتيجة الموديل.',
  workerOnlyTasks: 'مهام العامل فقط',
  ownerTaskStatus: 'متابعة المالك لحالة المهام',
  proofUpload: 'رفع صورة إثبات',
  proofNote: 'ملاحظة التنفيذ',
  taskNotStarted: 'لم تبدأ',
  taskInProgress: 'جاري التنفيذ',
  taskCompleted: 'تمت',
  startTask: 'بدء المهمة',
  completeAndNotifyOwner: 'إنهاء وإشعار المالك',
  serviceChat: 'الشات متصل بالخدمة عند توفره',
  askRealAssistant: 'اسأل المساعد الحقيقي',
  sensorUserHint: 'هذه الصفحة تخدم المستخدم الذي يملك قيم حساسات ويريد إرسالها للخدمة ثم للموديل.',

  sensorOnline: 'متصل', sensorOffline: 'غير متصل',
  soilMoistureSensor: 'حساس رطوبة التربة', tempSensor: 'حساس الحرارة', humiditySensor: 'حساس الرطوبة', lightSensor: 'حساس الإضاءة',
  farmManager: 'مدير المزرعة', irrigationWorker: 'مسؤول الري', plantCareWorker: 'مسؤول رعاية النباتات',
  greenhouseA: 'الصوبة أ', mintZone: 'منطقة النعناع', pepperSector: 'قطاع الفلفل',
  northField: 'الحقل الشمالي', eastSide: 'الجانب الشرقي', southGreenhouse: 'الصوبة الجنوبية',
  takePhoto: 'التقاط صورة', chooseFromGallery: 'اختيار من المعرض', changeImage: 'تغيير الصورة', removeImage: 'حذف الصورة',
  imagePreview: 'معاينة الصورة', takePhotoHint: 'افتح كاميرا الهاتف مباشرة', galleryHint: 'اختر صورة محفوظة من المعرض',
  imageReady: 'الصورة جاهزة', mobileImagePickerNote: 'اختر من الكاميرا أو المعرض ثم راجع الصورة قبل التحليل',
  mobileUploadHint: 'على الموبايل اختار التصوير بالكاميرا أو صورة من المعرض قبل التحليل.',
  noClearDisease: 'لا يوجد مرض واضح', leafSpotFungal: 'اشتباه بقع أوراق / فطريات', generalVisualStress: 'إجهاد بصري عام', severeEnvironmentalStress: 'إجهاد بيئي شديد'
};

const productionTranslations = {
  en: {
    appName:'Ecosense AI', goodMorning:'Good morning', goodAfternoon:'Good afternoon', goodEvening:'Good evening', overviewSubtitle:'Here’s a complete overview of your farm.', online:'Online', role:'Role', signOut:'Sign out', currentAccount:'Current Account', accountOnline:'Account online', markDone:'Mark as Done', profile:'Profile', farmPreferences:'Farm Preferences', appearance:'Appearance', profileInfo:'Profile Information', accountRolePermissions:'Account role & permissions', modelConnection:'Model connection', modelFinalStatusNote:'AI diagnosis summary and saved farm reports.',
    reopenOnboarding:'Open onboarding', onboardingStepAddSector:'Add sector', onboardingStepLinkDevice:'Link device', onboardingStepFirstDiagnosis:'Run first diagnosis', onboardingStepAddWorker:'Add worker', onboardingStepEnableAlerts:'Enable alerts',
    farmManagementTitle:'Farm Management', farmManagementSub:'Manage sectors, devices, farm status, sensor condition, and sector summaries from one clean production view.', farmOverview:'Farm Overview', sectorsDevicesStatus:'Sectors, devices, and farm status', devices:'Devices', sector:'Sector', sectorOverview:'Sector Overview', backToSectors:'Back to Farm Management', sectorSensors:'Sector Sensors', sensorName:'Sensor Name', sensorType:'Sensor Type', sensorValue:'Sensor Value', sensorUnit:'Unit', addSensor:'Add Sensor', assignedWorkers:'Responsible Workers', workersSectionSub:'Manage all workers, assign them to sectors, create accounts, and control permissions.',
    liveStreamMode:'Live Stream Mode', latestImageMode:'Latest Captured Image', cameraOfflineState:'Camera Offline', liveStreamReady:'Live camera feed will appear here when available.', latestCapturedFallback:'Showing the latest captured image when available.', cameraServiceHint:'Camera data will appear here when a capture is available.', lastCapturedImage:'Last captured image', retryCamera:'Retry camera', cameraOnline:'Camera online', cameraOffline:'Camera offline',
    viewSensorHistory:'View Sensor History', sensorHistory:'Sensor Reading History', readingsHistorySub:'Previous readings with time, sensor type, sector filters, and trend chart.', filterBySector:'Filter by sector', filterBySensor:'Filter by sensor', allSectors:'All sectors', allSensors:'All sensors', dateTime:'Date & Time', close:'Close',
    addExistingWorker:'Choose existing worker', createNewWorker:'Create new worker', generatedCredentials:'Generated Credentials', username:'Username', workerStatus:'Status', active:'Active', away:'Away', lastTask:'Last Task', permissionsOwnerOnly:'Sensitive services are owner-only.', ownerPermissions:'Owner permissions', workerPermissions:'Worker permissions', accountPasswordHint:'Copy the generated username and password and give them directly to the worker.',
    criticalAlerts:'Critical Alerts', warningAlerts:'Warning Alerts', deviceAlerts:'Device Alerts', diseaseAlerts:'Disease Alerts', sensorOfflineAlerts:'Sensor Offline Alerts', alertReason:'Alert reason', alertSeverity:'Severity', viewDiagnosis:'View Diagnosis',
    reportPreview:'Diagnosis Report Preview', exportPdf:'Export PDF', exportCsv:'Export CSV', filterByDate:'Filter by date', filterByStatus:'Filter by status', filterByStatusAll:'All statuses', workerTask:'Worker task', diagnosisTime:'Diagnosis time', finalStatusPrimary:'AI diagnosis summary used in reports and saved records.',
    mobileQuickDiagnosis:'Quick mobile diagnosis', takePlantPhoto:'Take plant photo', tasks:'Tasks', myTasks:'My Tasks', assignedSector:'Assigned Sector', pending:'Pending', completed:'Completed', startTask:'Start Task', items:'items', ownerOnlyAction:'Owner only action', workerLimitedView:'Worker limited view',
    farmManagerPermissions:'Farm Manager permissions', workerSettingsOnly:'Worker account settings only', activeWorkersNow:'Active workers now', lastSeen:'Last seen', onlineNow:'Online now', offlineNow:'Offline', currentTask:'Current task', deviceDetails:'Device Details', farmAlerts:'Farm Alerts', farmReports:'Farm Reports', operationalReady:'Operational and ready', serviceStructureReady:'No devices available yet', linkedServiceFeature:'Showing live farm data when available.', deviceReconnectRequested:'Device reconnect request sent.', deviceRemoved:'Device removed.', noServiceRealtime:'Live presence is not available yet.', systemMode:'System Mode', accountStatus:'Account Status', copyUsername:'Copy Username', copyPassword:'Copy Password', copyAll:'Copy All', ownerSensitiveBlocked:'Owner-sensitive action blocked for this role', reportsManagement:'Reports Management', alertsManagement:'Alerts Management', devicesManagement:'Devices Management', tasksManagement:'Tasks Management', viewFullReport:'View Full Report', openPlatform:'Open Platform', lowSoilMoistureAlert:'Alert: soil moisture is low', highTemperatureAlert:'Alert: temperature is high', fungalRiskAlert:'Alert: possible fungal infection',
    takePhoto:'Take Photo', chooseFromGallery:'Choose from Gallery', changeImage:'Change Image', removeImage:'Remove Image', imagePreview:'Image Preview', mobileUploadHint:'On mobile, choose camera or gallery before analysis.', serviceSensorReadings:'Sensor Readings', refreshSensors:'Refresh Sensors', simulateReadings:'Simulate Reading', sensorLoading:'Loading sensor readings...', sensorConnected:'Connected to sensor feed', sensorOfflineState:'Sensors offline', sensorErrorState:'Could not load sensor readings', simulationMode:'Simulation Mode', serviceMode:'Live Mode', lastUpdated:'Last Updated', noSensorData:'No sensor data available', retry:'Retry', latestServiceImage:'Latest camera image', combinedReady:'Image and readings are ready for combined diagnosis', serviceRequired:'Service unavailable', imageRequired:'Please add or capture a plant image first.', diagnosisSuccess:'Diagnosis completed successfully.', diagnosisFallback:'Diagnosis service could not complete the request.', sensorsUpdated:'Sensor readings updated.', sensorsSimulationUpdated:'Simulation readings refreshed.', sensorsOfflineFallback:'No live sensor reading is available. Enter readings manually or use simulation.', imageRemoved:'Image removed.', imageSelected:'Image selected successfully.', cameraCapture:'Open camera', gallerySelect:'Open gallery', takePhotoHint:'Open the mobile camera', galleryHint:'Select an existing image', imageReady:'Image ready', mobileImagePickerNote:'Use camera or gallery, then review the preview before analysis'
  },
  ar: {
    appName:'Ecosense AI', goodMorning:'صباح الخير', goodAfternoon:'مساء الخير', goodEvening:'مساء الخير', overviewSubtitle:'إليك نظرة عامة على حالة مزرعتك.', online:'متصل', role:'الدور', signOut:'تسجيل الخروج', currentAccount:'الحساب الحالي', accountOnline:'الحساب متصل', markDone:'إنهاء المهمة', profile:'البروفايل', farmPreferences:'تفضيلات المزرعة', appearance:'المظهر', profileInfo:'بيانات البروفايل', accountRolePermissions:'نوع الحساب والصلاحيات', modelConnection:'اتصال الموديل', modelFinalStatusNote:'ملخص التشخيص الذكي وتقارير المزرعة المحفوظة.',
    reopenOnboarding:'فتح التهيئة الأولى', onboardingStepAddSector:'أضف قطاع', onboardingStepLinkDevice:'اربط جهاز', onboardingStepFirstDiagnosis:'اعمل أول تشخيص', onboardingStepAddWorker:'أضف عامل', onboardingStepEnableAlerts:'فعل التنبيهات',
    farmManagementTitle:'إدارة المزرعة', farmManagementSub:'إدارة القطاعات والأجهزة وحالة المزرعة وحالة الحساسات وملخص كل قطاع من صفحة واحدة منظمة.', farmOverview:'نظرة عامة على المزرعة', sectorsDevicesStatus:'القطاعات والأجهزة وحالة المزرعة', devices:'الأجهزة', sector:'القطاع', sectorOverview:'ملخص القطاع', backToSectors:'الرجوع لإدارة المزرعة', sectorSensors:'حساسات القطاع', sensorName:'اسم الحساس', sensorType:'نوع الحساس', sensorValue:'قيمة الحساس', sensorUnit:'الوحدة', addSensor:'إضافة حساس', assignedWorkers:'العمال المسؤولون', workersSectionSub:'إدارة كل العمال وتعيينهم للقطاعات وإنشاء حسابات وصلاحيات لكل عامل.',
    liveStreamMode:'وضع البث المباشر', latestImageMode:'آخر صورة ملتقطة', cameraOfflineState:'الكاميرا غير متصلة', liveStreamReady:'سيظهر البث المباشر هنا عند توفره.', latestCapturedFallback:'يتم عرض آخر صورة ملتقطة عند توفرها.', cameraServiceHint:'ستظهر بيانات الكاميرا هنا عند توفر صورة ملتقطة.', lastCapturedImage:'آخر صورة ملتقطة', retryCamera:'إعادة محاولة الكاميرا', cameraOnline:'الكاميرا متصلة', cameraOffline:'الكاميرا غير متصلة',
    viewSensorHistory:'سجل القراءات', sensorHistory:'سجل قراءات الحساسات', readingsHistorySub:'قراءات سابقة بالوقت ونوع الحساس والقطاع مع فلاتر ورسم بياني بسيط.', filterBySector:'فلترة حسب القطاع', filterBySensor:'فلترة حسب الحساس', allSectors:'كل القطاعات', allSensors:'كل الحساسات', dateTime:'الوقت والتاريخ', close:'إغلاق',
    addExistingWorker:'اختيار عامل موجود', createNewWorker:'إنشاء عامل جديد', generatedCredentials:'بيانات الحساب الناتج', username:'اسم المستخدم', workerStatus:'الحالة', active:'نشط', away:'غير متاح', lastTask:'آخر مهمة', permissionsOwnerOnly:'الخدمات الحساسة مخصصة للمالك فقط.', ownerPermissions:'صلاحيات المالك', workerPermissions:'صلاحيات العامل', accountPasswordHint:'انسخ اسم المستخدم وكلمة المرور وسلمهم للعامل مباشرة.',
    criticalAlerts:'تنبيهات حرجة', warningAlerts:'تنبيهات تحذيرية', deviceAlerts:'تنبيهات الأجهزة', diseaseAlerts:'تنبيهات الأمراض', sensorOfflineAlerts:'حساس غير متصل', alertReason:'سبب التنبيه', alertSeverity:'درجة الخطورة', viewDiagnosis:'عرض التشخيص',
    reportPreview:'معاينة تقرير التشخيص', exportPdf:'تصدير PDF', exportCsv:'تصدير CSV', filterByDate:'فلترة بالتاريخ', filterByStatus:'فلترة بالحالة', filterByStatusAll:'كل الحالات', workerTask:'مهمة العامل', diagnosisTime:'وقت التشخيص', finalStatusPrimary:'ملخص التشخيص المستخدم في التقارير والسجلات المحفوظة.',
    mobileQuickDiagnosis:'تشخيص سريع من الهاتف', takePlantPhoto:'التقاط صورة النبات', tasks:'المهام', myTasks:'مهامي', assignedSector:'القطاع المسؤول عنه', pending:'قيد الانتظار', completed:'مكتملة', startTask:'بدء المهمة', items:'عناصر', ownerOnlyAction:'إجراء خاص بالمالك', workerLimitedView:'عرض محدود للعامل',
    farmManagerPermissions:'صلاحيات مدير المزرعة', workerSettingsOnly:'إعدادات حساب العامل فقط', activeWorkersNow:'العمال النشطين الآن', lastSeen:'آخر ظهور', onlineNow:'متصل الآن', offlineNow:'غير متصل', currentTask:'المهمة الحالية', deviceDetails:'تفاصيل الجهاز', farmAlerts:'تنبيهات المزرعة', farmReports:'تقارير المزرعة', operationalReady:'شغال وجاهز', serviceStructureReady:'لا توجد أجهزة متاحة بعد', linkedServiceFeature:'يتم عرض بيانات المزرعة المتاحة حاليًا.', deviceReconnectRequested:'تم إرسال طلب إعادة الاتصال بالجهاز.', deviceRemoved:'تم حذف الجهاز.', noServiceRealtime:'التواجد اللحظي غير متاح حاليًا.', systemMode:'وضع النظام', accountStatus:'حالة الحساب', copyUsername:'نسخ اسم المستخدم', copyPassword:'نسخ كلمة المرور', copyAll:'نسخ الكل', ownerSensitiveBlocked:'تم منع إجراء حساس مخصص للمالك فقط', reportsManagement:'إدارة التقارير', alertsManagement:'إدارة التنبيهات', devicesManagement:'إدارة الأجهزة', tasksManagement:'إدارة المهام', viewFullReport:'عرض التقرير كامل', openPlatform:'دخول المنصة', lowSoilMoistureAlert:'تنبيه: رطوبة التربة منخفضة', highTemperatureAlert:'تنبيه: درجة الحرارة مرتفعة', fungalRiskAlert:'تنبيه: احتمال إصابة فطرية', sensorsOfflineFallback:'لا توجد قراءة مباشرة من الحساسات حاليًا. أدخل القيم يدويًا أو استخدم المحاكاة.'
  },
  fr: { profile:'Profil', farmPreferences:'Préférences ferme', appearance:'Apparence', profileInfo:'Informations du profil', accountRolePermissions:'Rôle et permissions', modelConnection:'Connexion modèle', modelFinalStatusNote:'L’interface utilise toujours final_status comme état officiel.', reopenOnboarding:'Ouvrir l’onboarding', onboardingStepAddSector:'Ajouter un secteur', onboardingStepLinkDevice:'Lier un appareil', onboardingStepFirstDiagnosis:'Premier diagnostic', onboardingStepAddWorker:'Ajouter un ouvrier', onboardingStepEnableAlerts:'Activer les alertes', farmManagementTitle:'Gestion de la ferme', farmManagementSub:'Gérez secteurs, appareils, état de la ferme et capteurs.', farmOverview:'Vue ferme', sectorsDevicesStatus:'Secteurs, appareils et état', devices:'Appareils', sector:'Secteur', sectorOverview:'Aperçu secteur', backToSectors:'Retour gestion ferme', sectorSensors:'Capteurs secteur', sensorName:'Nom du capteur', sensorType:'Type de capteur', sensorValue:'Valeur', sensorUnit:'Unité', addSensor:'Ajouter capteur', assignedWorkers:'Ouvriers responsables', workersSectionSub:'Gérez les ouvriers, affectations, comptes et permissions.', liveStreamMode:'Mode direct', latestImageMode:'Dernière image', cameraOfflineState:'Caméra hors ligne', liveStreamReady:'Flux prêt pour service/hardware.', latestCapturedFallback:'Dernière image affichée si le direct est indisponible.', cameraServiceHint:'Prêt pour URL stream, snapshot et statut caméra.', lastCapturedImage:'Dernière capture', retryCamera:'Réessayer', cameraOnline:'Caméra en ligne', cameraOffline:'Caméra hors ligne', viewSensorHistory:'Historique capteurs', sensorHistory:'Historique des mesures', readingsHistorySub:'Mesures précédentes avec filtres et graphique.', filterBySector:'Filtrer par secteur', filterBySensor:'Filtrer par capteur', allSectors:'Tous les secteurs', allSensors:'Tous les capteurs', dateTime:'Date et heure', close:'Fermer', addExistingWorker:'Choisir ouvrier existant', createNewWorker:'Créer nouvel ouvrier', generatedCredentials:'Identifiants générés', username:'Nom utilisateur', workerStatus:'Statut', active:'Actif', away:'Absent', lastTask:'Dernière tâche', permissionsOwnerOnly:'Services sensibles réservés au propriétaire.', ownerPermissions:'Permissions propriétaire', workerPermissions:'Permissions ouvrier', accountPasswordHint:'Copiez l’utilisateur et le mot de passe.', criticalAlerts:'Alertes critiques', warningAlerts:'Alertes avertissement', deviceAlerts:'Alertes appareils', diseaseAlerts:'Alertes maladie', sensorOfflineAlerts:'Capteur hors ligne', alertReason:'Raison', alertSeverity:'Gravité', viewDiagnosis:'Voir diagnostic', reportPreview:'Aperçu du rapport', exportPdf:'Exporter PDF', exportCsv:'Exporter CSV', filterByDate:'Filtrer par date', filterByStatus:'Filtrer par état', filterByStatusAll:'Tous les états', workerTask:'Tâche ouvrier', diagnosisTime:'Heure diagnostic', finalStatusPrimary:'Résumé du diagnostic IA dans les rapports.', tasks:'Tâches', items:'éléments' },
  es: { profile:'Perfil', farmPreferences:'Preferencias de granja', appearance:'Apariencia', profileInfo:'Información del perfil', accountRolePermissions:'Rol y permisos', modelConnection:'Conexión del modelo', modelFinalStatusNote:'La interfaz usa siempre final_status como estado oficial.', reopenOnboarding:'Abrir onboarding', onboardingStepAddSector:'Agregar sector', onboardingStepLinkDevice:'Vincular dispositivo', onboardingStepFirstDiagnosis:'Primer diagnóstico', onboardingStepAddWorker:'Agregar trabajador', onboardingStepEnableAlerts:'Activar alertas', farmManagementTitle:'Gestión de granja', farmManagementSub:'Gestiona sectores, dispositivos, estado y sensores.', farmOverview:'Vista de granja', sectorsDevicesStatus:'Sectores, dispositivos y estado', devices:'Dispositivos', sector:'Sector', sectorOverview:'Resumen del sector', backToSectors:'Volver a gestión', sectorSensors:'Sensores del sector', sensorName:'Nombre del sensor', sensorType:'Tipo de sensor', sensorValue:'Valor', sensorUnit:'Unidad', addSensor:'Agregar sensor', assignedWorkers:'Trabajadores responsables', workersSectionSub:'Gestiona trabajadores, asignaciones, cuentas y permisos.', liveStreamMode:'Modo en vivo', latestImageMode:'Última imagen', cameraOfflineState:'Cámara sin conexión', liveStreamReady:'Stream listo para service/hardware.', latestCapturedFallback:'Mostrando última captura si no hay vivo.', cameraServiceHint:'Listo para URL de stream, snapshot y estado.', lastCapturedImage:'Última captura', retryCamera:'Reintentar', cameraOnline:'Cámara online', cameraOffline:'Cámara offline', viewSensorHistory:'Historial de sensores', sensorHistory:'Historial de lecturas', readingsHistorySub:'Lecturas anteriores con filtros y gráfico.', filterBySector:'Filtrar por sector', filterBySensor:'Filtrar por sensor', allSectors:'Todos los sectores', allSensors:'Todos los sensores', dateTime:'Fecha y hora', close:'Cerrar', addExistingWorker:'Elegir trabajador existente', createNewWorker:'Crear trabajador', generatedCredentials:'Credenciales generadas', username:'Usuario', workerStatus:'Estado', active:'Activo', away:'Ausente', lastTask:'Última tarea', permissionsOwnerOnly:'Servicios sensibles solo para propietario.', ownerPermissions:'Permisos propietario', workerPermissions:'Permisos trabajador', accountPasswordHint:'Copia usuario y contraseña.', criticalAlerts:'Alertas críticas', warningAlerts:'Alertas de advertencia', deviceAlerts:'Alertas de dispositivos', diseaseAlerts:'Alertas de enfermedad', sensorOfflineAlerts:'Sensor sin conexión', alertReason:'Razón', alertSeverity:'Severidad', viewDiagnosis:'Ver diagnóstico', reportPreview:'Vista previa del reporte', exportPdf:'Exportar PDF', exportCsv:'Exportar CSV', filterByDate:'Filtrar por fecha', filterByStatus:'Filtrar por estado', filterByStatusAll:'Todos los estados', workerTask:'Tarea trabajador', diagnosisTime:'Hora diagnóstico', finalStatusPrimary:'El resultado final es el estado principal; los detalles de sensores e imagen son explicativos.', tasks:'Tareas', items:'elementos' },
  de: { profile:'Profil', farmPreferences:'Farm-Einstellungen', appearance:'Darstellung', profileInfo:'Profilinformationen', accountRolePermissions:'Rolle und Rechte', modelConnection:'Modellverbindung', modelFinalStatusNote:'Die Oberfläche nutzt immer final_status als offiziellen Status.', reopenOnboarding:'Onboarding öffnen', onboardingStepAddSector:'Sektor hinzufügen', onboardingStepLinkDevice:'Gerät verbinden', onboardingStepFirstDiagnosis:'Erste Diagnose', onboardingStepAddWorker:'Arbeiter hinzufügen', onboardingStepEnableAlerts:'Warnungen aktivieren', farmManagementTitle:'Farmverwaltung', farmManagementSub:'Sektoren, Geräte, Farmstatus und Sensorstatus verwalten.', farmOverview:'Farmübersicht', sectorsDevicesStatus:'Sektoren, Geräte und Status', devices:'Geräte', sector:'Sektor', sectorOverview:'Sektorübersicht', backToSectors:'Zur Farmverwaltung', sectorSensors:'Sektorsensoren', sensorName:'Sensorname', sensorType:'Sensortyp', sensorValue:'Wert', sensorUnit:'Einheit', addSensor:'Sensor hinzufügen', assignedWorkers:'Zuständige Arbeiter', workersSectionSub:'Arbeiter, Zuweisungen, Konten und Rechte verwalten.', liveStreamMode:'Live-Stream', latestImageMode:'Letztes Bild', cameraOfflineState:'Kamera offline', liveStreamReady:'Stream bereit für Service/Hardware.', latestCapturedFallback:'Letzte Aufnahme wird angezeigt.', cameraServiceHint:'Bereit für Stream-URL, Snapshot und Kamerastatus.', lastCapturedImage:'Letzte Aufnahme', retryCamera:'Erneut versuchen', cameraOnline:'Kamera online', cameraOffline:'Kamera offline', viewSensorHistory:'Sensorverlauf', sensorHistory:'Messwertverlauf', readingsHistorySub:'Frühere Messwerte mit Filtern und Diagramm.', filterBySector:'Nach Sektor filtern', filterBySensor:'Nach Sensor filtern', allSectors:'Alle Sektoren', allSensors:'Alle Sensoren', dateTime:'Datum & Zeit', close:'Schließen', addExistingWorker:'Vorhandenen Arbeiter wählen', createNewWorker:'Neuen Arbeiter erstellen', generatedCredentials:'Generierte Zugangsdaten', username:'Benutzername', workerStatus:'Status', active:'Aktiv', away:'Abwesend', lastTask:'Letzte Aufgabe', permissionsOwnerOnly:'Sensible Dienste nur für Eigentümer.', ownerPermissions:'Eigentümerrechte', workerPermissions:'Arbeiterrechte', accountPasswordHint:'Benutzername und Passwort kopieren.', criticalAlerts:'Kritische Warnungen', warningAlerts:'Warnungen', deviceAlerts:'Gerätewarnungen', diseaseAlerts:'Krankheitswarnungen', sensorOfflineAlerts:'Sensor offline', alertReason:'Grund', alertSeverity:'Schweregrad', viewDiagnosis:'Diagnose ansehen', reportPreview:'Berichtsvorschau', exportPdf:'PDF exportieren', exportCsv:'CSV exportieren', filterByDate:'Nach Datum filtern', filterByStatus:'Nach Status filtern', filterByStatusAll:'Alle Status', workerTask:'Arbeiteraufgabe', diagnosisTime:'Diagnosezeit', finalStatusPrimary:'KI-Diagnoseübersicht für Berichte.', tasks:'Aufgaben', items:'Elemente' },
  it: { profile:'Profilo', farmPreferences:'Preferenze fattoria', appearance:'Aspetto', profileInfo:'Informazioni profilo', accountRolePermissions:'Ruolo e permessi', modelConnection:'Connessione modello', modelFinalStatusNote:'L’interfaccia usa sempre final_status come stato ufficiale.', reopenOnboarding:'Apri onboarding', onboardingStepAddSector:'Aggiungi settore', onboardingStepLinkDevice:'Collega dispositivo', onboardingStepFirstDiagnosis:'Prima diagnosi', onboardingStepAddWorker:'Aggiungi operaio', onboardingStepEnableAlerts:'Attiva avvisi', farmManagementTitle:'Gestione fattoria', farmManagementSub:'Gestisci settori, dispositivi, stato e sensori.', farmOverview:'Vista fattoria', sectorsDevicesStatus:'Settori, dispositivi e stato', devices:'Dispositivi', sector:'Settore', sectorOverview:'Panoramica settore', backToSectors:'Torna alla gestione', sectorSensors:'Sensori settore', sensorName:'Nome sensore', sensorType:'Tipo sensore', sensorValue:'Valore', sensorUnit:'Unità', addSensor:'Aggiungi sensore', assignedWorkers:'Operatori responsabili', workersSectionSub:'Gestisci operatori, assegnazioni, account e permessi.', liveStreamMode:'Modalità live', latestImageMode:'Ultima immagine', cameraOfflineState:'Camera offline', liveStreamReady:'Stream pronto per service/hardware.', latestCapturedFallback:'Ultima cattura mostrata se live non disponibile.', cameraServiceHint:'Pronto per URL stream, snapshot e stato.', lastCapturedImage:'Ultima cattura', retryCamera:'Riprova', cameraOnline:'Camera online', cameraOffline:'Camera offline', viewSensorHistory:'Storico sensori', sensorHistory:'Storico letture', readingsHistorySub:'Letture precedenti con filtri e grafico.', filterBySector:'Filtra per settore', filterBySensor:'Filtra per sensore', allSectors:'Tutti i settori', allSensors:'Tutti i sensori', dateTime:'Data e ora', close:'Chiudi', addExistingWorker:'Scegli operaio esistente', createNewWorker:'Crea operaio', generatedCredentials:'Credenziali generate', username:'Username', workerStatus:'Stato', active:'Attivo', away:'Assente', lastTask:'Ultimo task', permissionsOwnerOnly:'Servizi sensibili solo al proprietario.', ownerPermissions:'Permessi proprietario', workerPermissions:'Permessi operaio', accountPasswordHint:'Copia username e password.', criticalAlerts:'Avvisi critici', warningAlerts:'Avvisi warning', deviceAlerts:'Avvisi dispositivi', diseaseAlerts:'Avvisi malattia', sensorOfflineAlerts:'Sensore offline', alertReason:'Motivo', alertSeverity:'Gravità', viewDiagnosis:'Vedi diagnosi', reportPreview:'Anteprima report', exportPdf:'Esporta PDF', exportCsv:'Esporta CSV', filterByDate:'Filtra per data', filterByStatus:'Filtra per stato', filterByStatusAll:'Tutti gli stati', workerTask:'Task operaio', diagnosisTime:'Ora diagnosi', finalStatusPrimary:'Sintesi diagnosi IA nei report.', tasks:'Task', items:'elementi' },
  tr: { profile:'Profil', farmPreferences:'Çiftlik tercihleri', appearance:'Görünüm', profileInfo:'Profil bilgileri', accountRolePermissions:'Rol ve izinler', modelConnection:'Model bağlantısı', modelFinalStatusNote:'Arayüz resmi durum olarak her zaman final_status kullanır.', reopenOnboarding:'Onboarding aç', onboardingStepAddSector:'Sektör ekle', onboardingStepLinkDevice:'Cihaz bağla', onboardingStepFirstDiagnosis:'İlk teşhis', onboardingStepAddWorker:'Çalışan ekle', onboardingStepEnableAlerts:'Uyarıları aç', farmManagementTitle:'Çiftlik Yönetimi', farmManagementSub:'Sektörleri, cihazları, çiftlik durumunu ve sensörleri yönetin.', farmOverview:'Çiftlik görünümü', sectorsDevicesStatus:'Sektörler, cihazlar ve durum', devices:'Cihazlar', sector:'Sektör', sectorOverview:'Sektör özeti', backToSectors:'Çiftlik yönetimine dön', sectorSensors:'Sektör sensörleri', sensorName:'Sensör adı', sensorType:'Sensör tipi', sensorValue:'Değer', sensorUnit:'Birim', addSensor:'Sensör ekle', assignedWorkers:'Sorumlu çalışanlar', workersSectionSub:'Çalışanları, atamaları, hesapları ve izinleri yönetin.', liveStreamMode:'Canlı yayın modu', latestImageMode:'Son yakalanan görüntü', cameraOfflineState:'Kamera çevrimdışı', liveStreamReady:'Service/hardware canlı yayın için hazır.', latestCapturedFallback:'Canlı yayın yoksa son görüntü gösterilir.', cameraServiceHint:'Stream URL, snapshot ve kamera durumu için hazır.', lastCapturedImage:'Son görüntü', retryCamera:'Kamerayı yeniden dene', cameraOnline:'Kamera çevrimiçi', cameraOffline:'Kamera çevrimdışı', viewSensorHistory:'Sensör geçmişi', sensorHistory:'Okuma geçmişi', readingsHistorySub:'Filtreler ve grafik ile önceki okumalar.', filterBySector:'Sektöre göre filtrele', filterBySensor:'Sensöre göre filtrele', allSectors:'Tüm sektörler', allSensors:'Tüm sensörler', dateTime:'Tarih ve saat', close:'Kapat', addExistingWorker:'Mevcut çalışan seç', createNewWorker:'Yeni çalışan oluştur', generatedCredentials:'Oluşturulan bilgiler', username:'Kullanıcı adı', workerStatus:'Durum', active:'Aktif', away:'Uzakta', lastTask:'Son görev', permissionsOwnerOnly:'Hassas hizmetler sadece sahibindir.', ownerPermissions:'Sahip izinleri', workerPermissions:'Çalışan izinleri', accountPasswordHint:'Kullanıcı adı ve şifreyi kopyalayın.', criticalAlerts:'Kritik uyarılar', warningAlerts:'Uyarılar', deviceAlerts:'Cihaz uyarıları', diseaseAlerts:'Hastalık uyarıları', sensorOfflineAlerts:'Sensör çevrimdışı', alertReason:'Neden', alertSeverity:'Şiddet', viewDiagnosis:'Teşhisi görüntüle', reportPreview:'Rapor önizleme', exportPdf:'PDF dışa aktar', exportCsv:'CSV dışa aktar', filterByDate:'Tarihe göre filtrele', filterByStatus:'Duruma göre filtrele', filterByStatusAll:'Tüm durumlar', workerTask:'Çalışan görevi', diagnosisTime:'Teşhis zamanı', finalStatusPrimary:'Raporlarda kullanılan AI tanı özeti.', tasks:'Görevler', items:'öğe' }
};


const finalPolishTranslations = {
  en: {
    diagnosisReport: 'Diagnosis Report', sensorReport: 'Sensor Report', farmStatusReport: 'Farm Status Report', diseaseAlertReport: 'Disease Alert Report',
    reportNameDiagnosis: 'Diagnosis Report', reportNameSensor: 'Sensor Report', reportNameFarmStatus: 'Farm Status Report', reportNameDiseaseAlert: 'Disease Alert Report',
    reportName: 'Report Name', diagnosisReportSimple: 'Diagnosis Summary Report', diseaseGuide: 'Disease Guide',
    greenhouseA: 'Greenhouse A', mintZone: 'Mint Zone', pepperSector: 'Pepper Sector', northField: 'North Field', eastSide: 'East Side', southGreenhouse: 'South Greenhouse',
    leafSpotFungal: 'Leaf Spot / Fungal Suspicion', generalVisualStress: 'General Visual Stress', severeEnvironmentalStress: 'Severe Environmental Stress',
    diseaseSymptoms: 'Symptoms', diseaseCauses: 'Causes', diseaseInitialSolution: 'Initial Solutions', diseasePrevention: 'Prevention',
    diseaseLeafSpotSymptoms: 'Dark or brown spots on leaves with possible yellow halos around the damaged area.',
    diseaseLeafSpotCauses: 'High leaf wetness, weak ventilation, contaminated tools, or fungal spread between plants.',
    diseaseLeafSpotSolution: 'Inspect affected leaves, improve ventilation, reduce water on leaf surfaces, and create a treatment task if symptoms spread.',
    diseaseLeafSpotPrevention: 'Avoid overhead watering, keep enough spacing between plants, and clean tools regularly.',
    diseaseVisualStressSymptoms: 'Yellowing, light wilting, pale leaves, or small damage at leaf edges.',
    diseaseVisualStressCauses: 'Water imbalance, weak light, heat stress, nutrient stress, or unstable environmental readings.',
    diseaseVisualStressSolution: 'Review irrigation, light, temperature, and humidity, then run a new diagnosis using image and sensor readings.',
    diseaseVisualStressPrevention: 'Keep readings within the recommended range and monitor changes every few days.',
    diseaseSevereStressSymptoms: 'Severe wilting, very dry soil, high heat, weak light, or fast decline in plant appearance.',
    diseaseSevereStressCauses: 'Critical environmental stress caused by heat, dry root zone, poor irrigation, or delayed response to alerts.',
    diseaseSevereStressSolution: 'Water gradually, reduce heat exposure, check sensors immediately, and assign an urgent worker task.',
    diseaseSevereStressPrevention: 'Enable alerts, check sensor status daily, and review sector reports regularly.',
    serviceRequired: 'Service Required', sensorsApiNotConnected: 'No sensor readings are available yet', simulationMode: 'Manual Test Mode', retryConnection: 'Retry Connection',
    assistantMobileNote: 'On mobile, the assistant opens as a compact bottom sheet.', askAnythingPlaceholder: 'Ask anything about agriculture or plant health...',
    accountAccessHint: 'Your account controls your access. Owners get full access; workers only see assigned tools and tasks.',
    viewFullReport: 'View Full Report', plantPhoto: 'Plant photo', emptyState: 'No data available yet',
    loginSecurityTitle: 'Your account controls your access', loginSecurityBody: 'Owners get full access; workers only see assigned tools and tasks.',
    serviceSensorStatus: 'Sensor feed status', noRepeatedRequests: 'Automatic sensor refresh is paused until readings are available.'
  },
  ar: {
    diagnosisReport: 'تقرير التشخيص', sensorReport: 'تقرير الحساسات', farmStatusReport: 'تقرير حالة المزرعة', diseaseAlertReport: 'تقرير تنبيهات الأمراض',
    reportNameDiagnosis: 'تقرير التشخيص', reportNameSensor: 'تقرير الحساسات', reportNameFarmStatus: 'تقرير حالة المزرعة', reportNameDiseaseAlert: 'تقرير تنبيهات الأمراض',
    reportName: 'اسم التقرير', diagnosisReportSimple: 'تقرير ملخص التشخيص', diseaseGuide: 'دليل أمراض النبات',
    greenhouseA: 'الصوبة أ', mintZone: 'منطقة النعناع', pepperSector: 'قطاع الفلفل', northField: 'الحقل الشمالي', eastSide: 'الجانب الشرقي', southGreenhouse: 'الصوبة الجنوبية',
    leafSpotFungal: 'اشتباه تبقع الأوراق / فطريات', generalVisualStress: 'إجهاد بصري عام', severeEnvironmentalStress: 'إجهاد بيئي شديد',
    diseaseSymptoms: 'الأعراض', diseaseCauses: 'الأسباب', diseaseInitialSolution: 'الحلول الأولية', diseasePrevention: 'الوقاية',
    diseaseLeafSpotSymptoms: 'بقع داكنة أو بنية على الأوراق مع احتمال ظهور اصفرار حول منطقة الإصابة.',
    diseaseLeafSpotCauses: 'بلل زائد على الأوراق، ضعف التهوية، أدوات ملوثة، أو انتقال إصابة فطرية بين النباتات.',
    diseaseLeafSpotSolution: 'افحص الأوراق المصابة، حسّن التهوية، قلل المياه على سطح الورقة، وأنشئ مهمة علاج إذا انتشرت الأعراض.',
    diseaseLeafSpotPrevention: 'تجنب الري فوق الأوراق، اترك مسافات مناسبة بين النباتات، ونظف الأدوات بانتظام.',
    diseaseVisualStressSymptoms: 'اصفرار أو ذبول بسيط أو بهتان في لون الأوراق أو تلف بسيط في الأطراف.',
    diseaseVisualStressCauses: 'عدم توازن الري، ضعف الإضاءة، إجهاد حراري، نقص عناصر، أو اضطراب في القراءات البيئية.',
    diseaseVisualStressSolution: 'راجع الري والإضاءة والحرارة والرطوبة، ثم أعد التشخيص باستخدام الصورة والحساسات.',
    diseaseVisualStressPrevention: 'حافظ على القراءات داخل النطاق المناسب وراجع التغيرات كل عدة أيام.',
    diseaseSevereStressSymptoms: 'ذبول شديد، تربة جافة جدًا، حرارة مرتفعة، إضاءة ضعيفة، أو تدهور سريع في مظهر النبات.',
    diseaseSevereStressCauses: 'إجهاد بيئي حرج بسبب الحرارة أو جفاف منطقة الجذور أو ضعف الري أو تأخر التعامل مع التنبيهات.',
    diseaseSevereStressSolution: 'اسقِ النبات تدريجيًا، قلل التعرض للحرارة، افحص الحساسات فورًا، وعيّن مهمة عاجلة للعامل.',
    diseaseSevereStressPrevention: 'فعّل التنبيهات، راجع حالة الحساسات يوميًا، وتابع تقارير القطاعات بانتظام.',
    serviceRequired: 'الخدمة غير متاحة حاليًا', sensorsApiNotConnected: 'لا توجد قراءات حساسات متاحة حاليًا', simulationMode: 'وضع المحاكاة', retryConnection: 'إعادة المحاولة',
    assistantMobileNote: 'على الموبايل يفتح المساعد كنافذة سفلية مدمجة.', askAnythingPlaceholder: 'اسأل عن الزراعة أو صحة النبات...',
    accountAccessHint: 'حسابك يحدد صلاحيات الوصول. المالك لديه صلاحيات كاملة، والعامل يرى الأدوات والمهام المخصصة له فقط.',
    viewFullReport: 'عرض التقرير كامل', plantPhoto: 'صورة النبات', emptyState: 'لا توجد بيانات متاحة بعد',
    loginSecurityTitle: 'حسابك يحدد صلاحيات الوصول', loginSecurityBody: 'المالك لديه صلاحيات كاملة، والعامل يرى الأدوات والمهام المخصصة له فقط.',
    serviceSensorStatus: 'حالة تغذية الحساسات', noRepeatedRequests: 'تم إيقاف تحديث الحساسات التلقائي حتى تتوفر القراءات.'
  }
};

function humanizeKey(key){
  const raw = String(key ?? '');
  if(!raw) return '';
  const spaced = raw
    .replace(/[_-]+/g,' ')
    .replace(/([a-z])([A-Z])/g,'$1 $2')
    .replace(/\s+/g,' ')
    .trim();
  return spaced ? spaced.charAt(0).toUpperCase()+spaced.slice(1) : raw;
}

function displayReportName(type='diagnosis', sector='', t){
  const map = { diagnosis:'diagnosisReport', sensor:'sensorReport', farm:'farmStatusReport', disease:'diseaseAlertReport' };
  const base = t(map[type] || 'diagnosisReport');
  const localSector = sector ? localizeValue(sector, t) : '';
  return localSector ? `${base} - ${localSector}` : base;
}


function professionalReportHtml({t, lang, rows=[], titleKey='farmReports'}){
  const dir = lang === 'ar' ? 'rtl' : 'ltr';
  const settings = getStore('sph_settings', {farmName:'Ecosense Smart Farm'});
  const userName = displayUserName();
  const role = roleLabel(t);
  const now = new Date().toLocaleString();
  const esc = (v)=>String(v ?? '').replace(/[&<>]/g, ch=>({ '&':'&amp;', '<':'&lt;', '>':'&gt;' }[ch]));
  const renderRows = rows.slice(0,10).map((r,i)=>`<section class="report-card">
    <div class="report-card-head"><h2>${esc(displayReportName('diagnosis', r.sector, t))}</h2><span class="status">${esc(labelStatus(r.final_status,t))}</span></div>
    <div class="meta-grid">
      <div><b>${esc(t('sector'))}</b><span>${esc(localizeValue(r.sector,t))}</span></div>
      <div><b>${esc(t('date'))}</b><span>${esc(r.date || '')} ${esc(r.time || '')}</span></div>
      <div><b>${esc(t('finalStatus'))}</b><span>${esc(labelStatus(r.final_status,t))}</span></div>
      <div><b>${esc(t('confidence'))}</b><span>${esc(r.confidence!=null ? Math.round(Number(r.confidence)*100)+'%' : '-')}</span></div>
      <div><b>${esc(t('diseaseName'))}</b><span>${esc(translateModelText(r.disease_name || r.disease || '-',t))}</span></div>
      <div><b>${esc(t('diagnosisMode'))}</b><span>${esc(translateModelText(r.analysisType || r.diagnosisMode || '-',t))}</span></div>
    </div>
    ${r.image ? `<img class="plant-img" src="${esc(r.image)}" alt="Plant image"/>` : ''}
    <h3>${esc(t('imageDiagnosis') || 'Image Analysis')}</h3>
    <div class="meta-grid">
      <div><b>${esc(t('imageStatus'))}</b><span>${esc(translateModelText(r.image_status || '-',t))}</span></div>
      <div><b>${esc(t('diseaseDetection') || t('diseaseProblem'))}</b><span>${esc(translateModelText(r.disease_name || r.disease || '-',t))}</span></div>
    </div>
    <h3>${esc(t('sentReadings') || 'Readings')}</h3>
    <div class="reading-grid-print">
      <span>${esc(t('temperature'))}: <b>${esc(formatInputValue('temperature',r.temperature))}</b></span>
      <span>${esc(t('humidity'))}: <b>${esc(formatInputValue('humidity',r.humidity))}</b></span>
      <span>${esc(t('soilMoisture'))}: <b>${esc(formatInputValue('soilMoisture',r.soilMoisture))}</b></span>
      <span>${esc(t('soilTemp'))}: <b>${esc(formatInputValue('soilTemp',r.soilTemp))}</b></span>
      <span>${esc(t('light'))}: <b>${esc(localizeValue(r.light,t))}</b></span>
    </div>
    <h3>${esc(t('diagnosisSummary'))}</h3><p>${esc(translateModelText(r.diagnosis || '-',t))}</p>
    <h3>${esc(t('recommendations'))}</h3><ul>${(Array.isArray(r.recommendations)?r.recommendations:[]).map(x=>`<li>${esc(translateModelText(x,t))}</li>`).join('') || `<li>${esc(t('empty'))}</li>`}</ul>
    <h3>${esc(t('actions'))}</h3><ul>${(Array.isArray(r.actions)?r.actions:[]).map(x=>`<li>${esc(translateModelText(x,t))}</li>`).join('') || `<li>${esc(t('empty'))}</li>`}</ul>
  </section>`).join('');
  return `<!doctype html><html dir="${dir}" lang="${lang}"><head><meta charset="utf-8"><title>${esc(t(titleKey))}</title><style>
    body{font-family:Arial,Tahoma,sans-serif;margin:0;background:#f4fbf6;color:#10261b;direction:${dir};}
    .wrap{max-width:920px;margin:0 auto;padding:28px}.cover{background:linear-gradient(135deg,#064e3b,#16a34a);color:white;border-radius:22px;padding:28px;margin-bottom:18px;display:flex;gap:18px;align-items:center;justify-content:space-between}.brand{display:flex;gap:14px;align-items:center}.logo{width:64px;height:64px;border-radius:18px;background:white;padding:8px;object-fit:contain}.cover h1{margin:0 0 6px;font-size:30px}.cover p{margin:3px 0;opacity:.9}.meta{background:white;border:1px solid #d8eadf;border-radius:18px;padding:16px;margin-bottom:16px}.meta-grid,.reading-grid-print{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.meta-grid div,.reading-grid-print span{border:1px solid #e1eee6;background:#fbfffc;border-radius:12px;padding:10px}.meta-grid b{display:block;color:#52715f;font-size:12px}.report-card{background:white;border:1px solid #d8eadf;border-radius:20px;padding:20px;margin:16px 0;page-break-inside:avoid}.report-card-head{display:flex;align-items:center;justify-content:space-between;gap:12px;border-bottom:1px solid #e1eee6;padding-bottom:10px}.report-card h2{font-size:22px;margin:0}.report-card h3{font-size:15px;margin:18px 0 8px;color:#0f7a45}.status{background:#dcfce7;color:#166534;border-radius:999px;padding:8px 12px;font-weight:700}.plant-img{max-width:100%;border-radius:16px;margin:12px 0;border:1px solid #e1eee6}ul{margin-top:8px}li{margin:6px 0}.note{font-size:12px;color:#567363;margin-top:18px}@media print{body{background:white}.wrap{padding:0}.no-print{display:none}.cover,.report-card,.meta{box-shadow:none}button{display:none}}
  </style></head><body><div class="wrap">
    <section class="cover"><div class="brand"><img class="logo" src="/project-logo.png"/><div><h1>${esc(t(titleKey))}</h1><p>${esc(t('appName'))} — Smart Plant Health</p></div></div><button class="no-print" onclick="window.print()">${esc(t('printOrSavePdf'))}</button></section>
    <section class="meta"><div class="meta-grid">
      <div><b>${esc(t('farmName'))}</b><span>${esc(settings.farmName || 'Ecosense Smart Farm')}</span></div>
      <div><b>${esc(t('reportGeneratedBy'))}</b><span>${esc(userName)}</span></div>
      <div><b>${esc(t('role'))}</b><span>${esc(role)}</span></div>
      <div><b>${esc(t('date'))}</b><span>${esc(now)}</span></div>
    </div><p class="note">${esc(t('reportOwnerNameNote'))}</p></section>
    ${renderRows || `<section class="report-card"><p>${esc(t('emptyState'))}</p></section>`}
    <footer class="note">Generated by Ecosense AI - Smart Plant Health Diagnosis System</footer>
  </div><script>setTimeout(()=>window.print(),500)</script></body></html>`;
}
function openProfessionalReport(args){
  const html = professionalReportHtml(args);
  const w = window.open('', '_blank', 'width=980,height=900');
  if(w){ w.document.open(); w.document.write(html); w.document.close(); }
  else { download('ecosense-professional-report.html', html, 'text/html'); }
  try{ addNotification(args.t('reportExportReady'),'success'); }catch{}
}

function objectToDisplayText(value) {
  if (value == null) return '';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) return value.map(objectToDisplayText).filter(Boolean).join(' - ');
  if (typeof value === 'object') {
    const direct = value.name || value.title || value.label || value.message || value.text || value.body || value.reason || value.status || value.value || value.sectorName || value.cropType || value.plantType || value.type || value.email || value._id || value.id;
    if (direct != null && direct !== '') return objectToDisplayText(direct);
    try { return JSON.stringify(value); } catch { return ''; }
  }
  return String(value);
}

function localizeValue(value, t) {
  const v = objectToDisplayText(value);
  const map = {
    Tomato: t('tomato'), Corn: t('corn'), Pepper: t('pepper'), Mint: t('mint'),
    Low: t('low'), Medium: t('medium'), Sufficient: t('sufficient'),
    Healthy: t('healthy'), 'Moderate Stress': t('moderate'), 'High Stress': t('high'),
    'No Image': t('noImage'), 'No Clear Disease Detected': t('noDiseaseDetected'),
    'Leaf Spot / Fungal Suspicion': t('leafSpotFungal'), 'General Visual Stress': t('generalVisualStress'),
    'Severe Environmental Stress': t('severeEnvironmentalStress'),
    'Greenhouse A': t('greenhouseA'), 'Mint Zone': t('mintZone'), 'Pepper Sector': t('pepperSector'),
    'North Field': t('northField'), 'East Side': t('eastSide'), 'South Greenhouse': t('southGreenhouse'),
    'Online': t('sensorOnline'), 'Offline': t('sensorOffline'), 'Inspect leaves': t('taskInspectLeaves'), 'Check irrigation system': t('taskCheckIrrigation'), 'Clean plant area': t('taskCleanArea'), 'Adjust light and ventilation': t('taskAdjustLight'), 'Irrigation Pump': t('pump'), 'Monitoring Camera': t('camera'), 'Water Valve': t('valve'), 'Controller': t('controller'), 'Farm Manager': t('farmManager'), 'Irrigation Worker': t('irrigationWorker'), 'Plant Care Worker': t('plantCareWorker'), 'Soil Moisture Sensor': t('soilMoistureSensor'), 'Temperature Sensor': t('tempSensor'), 'Humidity Sensor': t('humiditySensor'), 'Light Sensor': t('lightSensor'), 'Soil Moisture': t('soilMoisture'), 'Temperature': t('temperature'), 'Light': t('light')
  };
  return map[v] || v;
}


const modelTextPairs = [
  ['Readings are stable. No urgent plant-health risk detected.', 'القراءات مستقرة ولا يوجد خطر عاجل على صحة النبات.'],
  ['Keep monitoring sensor readings.', 'استمر في متابعة قراءات الحساسات.'],
  ['Maintain stable watering and light exposure.', 'حافظ على انتظام الري وتعرض النبات لإضاءة مناسبة.'],
  ['Recheck leaves weekly.', 'افحص الأوراق مرة كل أسبوع.'],
  ['Continue normal care.', 'استمر في الرعاية الطبيعية للنبات.'],
  ['The plant may be under water or heat stress. Monitoring and corrective care are recommended.', 'قد يكون النبات تحت إجهاد مائي أو حراري، لذلك يُنصح بالمتابعة واتخاذ رعاية تصحيحية.'],
  ['Adjust irrigation according to soil moisture.', 'اضبط الري حسب نسبة رطوبة التربة.'],
  ['Improve airflow and reduce heat exposure.', 'حسّن التهوية وقلل تعرض النبات للحرارة العالية.'],
  ['Inspect leaves for spots, yellowing, or brown tissue.', 'افحص الأوراق للتأكد من وجود بقع أو اصفرار أو أنسجة بنية.'],
  ['Water gradually if soil is dry.', 'اسقِ النبات تدريجيًا إذا كانت التربة جافة.'],
  ['Move plant away from extreme heat.', 'أبعد النبات عن الحرارة الشديدة.'],
  ['Isolate visibly damaged plants if disease symptoms spread.', 'اعزل النباتات المتضررة إذا بدأت أعراض المرض في الانتشار.'],
  ['The plant is in a dangerous condition. Low moisture, high heat, or weak light may cause fast deterioration.', 'النبات في حالة خطرة؛ انخفاض الرطوبة أو ارتفاع الحرارة أو ضعف الإضاءة قد يؤدي إلى تدهور سريع.'],
  ['Stable plant condition with balanced readings.', 'حالة النبات مستقرة والقراءات متوازنة.'],
  ['Moisture is below the preferred range and requires monitoring.', 'رطوبة التربة أقل من النطاق المناسب وتحتاج إلى متابعة.'],
  ['Critical heat and low soil moisture indicate urgent plant risk.', 'الحرارة المرتفعة وانخفاض رطوبة التربة يشيران إلى خطر عاجل على النبات.'],
  ['New sector is ready for automatic monitoring.', 'القطاع الجديد جاهز للمتابعة التلقائية.'],
  ['Sector is stable. Continue weekly leaf inspection.', 'القطاع مستقر. استمر في فحص الأوراق أسبوعيًا.'],
  ['Mint sector needs moisture follow-up.', 'قطاع النعناع يحتاج إلى متابعة رطوبة التربة.'],
  ['High risk area. Inspect disease symptoms today.', 'منطقة عالية الخطورة. افحص أعراض المرض اليوم.'],
  ['No critical stress indicators detected.', 'لم يتم اكتشاف مؤشرات إجهاد خطيرة.'],
  ['Water or heat stress indicators detected.', 'تم اكتشاف مؤشرات إجهاد مائي أو حراري.'],
  ['Critical environmental risk detected. Image review may indicate leaf stress symptoms.', 'تم اكتشاف خطر بيئي حرج، وقد تشير الصورة إلى أعراض إجهاد على الأوراق.'],
  ['Low soil moisture or high temperature', 'انخفاض رطوبة التربة أو ارتفاع درجة الحرارة'],
  ['Critical low soil moisture', 'انخفاض حرج في رطوبة التربة'],
  ['Low light', 'إضاءة منخفضة'],
  ['Heat stress', 'إجهاد حراري'],
  ['Urgent plant risk', 'خطر عاجل على النبات'],
  ['Leaf Spot / Fungal Suspicion', 'اشتباه بقع أوراق / فطريات'],
  ['Possible Leaf Spot Disease', 'مرض تبقع الأوراق المحتمل'],
  ['Possible Chlorosis', 'اصفرار أوراق محتمل'],
  ['General Visual Stress', 'إجهاد بصري عام'],
  ['Severe Environmental Stress', 'إجهاد بيئي شديد'],
  ['No Clear Disease Detected', 'لا يوجد مرض واضح'],
  ['No clear disease detected', 'لا يوجد مرض واضح'],
  ['Possible disease detected', 'تم اكتشاف مرض محتمل'],
  ['Apply approved fungicide', 'استخدم مبيدًا فطريًا معتمدًا'],
  ['Improve ventilation', 'حسّن التهوية'],
  ['Remove infected leaves', 'أزل الأوراق المصابة'],
  ['Monitor readings', 'تابع القراءات'],
  ['Create treatment task', 'أنشئ مهمة علاج'],
  ['Weekly follow-up', 'متابعة أسبوعية'],
  ['Inspect leaves', 'فحص الأوراق'],
  ['Check irrigation system', 'فحص نظام الري'],
  ['Clean plant area', 'تنظيف منطقة النبات'],
  ['Adjust light and ventilation', 'ضبط الإضاءة والتهوية'],
  ['Leaf spots', 'بقع على الأوراق'],
  ['Analyzed', 'تم التحليل']
];
function translateModelText(value, t) {
  if (value == null) return value;
  const isArabic = t('name') === 'العربية' || t('healthy') === 'سليم';
  const enToAr = Object.fromEntries(modelTextPairs);
  const arToEn = Object.fromEntries(modelTextPairs.map(([en, ar]) => [ar, en]));
  const translateOne = (txt) => {
    const s = objectToDisplayText(txt).trim();
    if (!s) return s;
    if (isArabic) return enToAr[s] || localizeValue(s, t);
    return arToEn[s] || localizeValue(s, t);
  };
  if (Array.isArray(value)) return value.map(translateOne);
  return translateOne(value);
}


const finalDeclutterTranslations = {
  en: {
    askAssistant: 'Ask Assistant',
    diseaseGuideMovedTitle: 'Disease guide moved to Assistant',
    diseaseGuideMovedBody: 'Educational disease information is now available inside the Assistant to keep the interface lighter on mobile.',
    diseaseGuideInAssistant: 'Disease Guide in Assistant',
    showDiseaseRecommendations: 'Show disease recommendations',
    leafSpotQuestion: 'What does leaf spot mean?',
    chlorosisQuestion: 'How do I handle yellow leaves?',
    diseaseAssistantHint: 'Ask about symptoms, causes, severity, recommendations, or whether a worker task is needed.',
    diseaseSeverity: 'Severity',
    suggestedTask: 'Suggested worker task',
    createWorkerTaskRecommended: 'Create a worker task if symptoms spread or the sector risk is high.',
    createWorkerTaskNotRequired: 'A worker task is not required now; keep monitoring readings and visual changes.',
    assistantDiseaseCardTitle: 'Disease guide summary',
    assistantCanHelpWithGuide: 'I can now explain the disease guide here instead of using a separate page.',
    clutterReductionReport: 'Clutter reduction applied',
    movedToAssistant: 'Moved to Assistant',
    movedToModal: 'Moved to Modal',
    movedToAccordion: 'Moved to Accordion',
    movedToMoreMenu: 'Moved to More Menu',
    keepVisible: 'Must stay visible'
  },
  ar: {
    askAssistant: 'اسأل المساعد',
    diseaseGuideMovedTitle: 'تم نقل دليل الأمراض إلى المساعد',
    diseaseGuideMovedBody: 'معلومات الأمراض التعليمية أصبحت داخل المساعد حتى تظل الواجهة أخف وأنسب للموبايل.',
    diseaseGuideInAssistant: 'دليل الأمراض داخل المساعد',
    showDiseaseRecommendations: 'اعرض توصيات الأمراض',
    leafSpotQuestion: 'ما معنى تبقع الأوراق؟',
    chlorosisQuestion: 'كيف أتعامل مع اصفرار النبات؟',
    diseaseAssistantHint: 'اسأل عن الأعراض أو الأسباب أو الخطورة أو التوصيات أو هل تحتاج مهمة للعامل.',
    diseaseSeverity: 'درجة الخطورة',
    suggestedTask: 'مهمة مقترحة للعامل',
    createWorkerTaskRecommended: 'أنشئ مهمة للعامل إذا انتشرت الأعراض أو كان القطاع عالي الخطورة.',
    createWorkerTaskNotRequired: 'لا تحتاج مهمة عامل الآن؛ استمر في متابعة القراءات والتغيرات البصرية.',
    assistantDiseaseCardTitle: 'ملخص دليل الأمراض',
    assistantCanHelpWithGuide: 'أقدر أشرح دليل الأمراض هنا بدل وجود صفحة منفصلة.',
    clutterReductionReport: 'تم تقليل الزحمة في الواجهة',
    movedToAssistant: 'تم نقله إلى المساعد',
    movedToModal: 'تم نقله إلى نافذة',
    movedToAccordion: 'تم نقله إلى Accordion',
    movedToMoreMenu: 'تم نقله إلى قائمة المزيد',
    keepVisible: 'لازم يفضل ظاهر'
  }
};



const roleDashboardTranslations = {
  en: {
    dashboardStart: 'Dashboard',
    reportsCombinedNotice: 'Diagnosis reports were merged into Farm Reports to reduce menu clutter.',
    reportGeneratedBy: 'Generated by',
    farmName: 'Farm Name',
    professionalReport: 'Professional Report',
    saveAsPdfHint: 'Use the browser print dialog and choose Save as PDF.',
    openReport: 'Open Report',
    printOrSavePdf: 'Print / Save PDF',
    ownerDashboard: 'Owner Dashboard',
    farmManagerDashboard: 'Farm Manager Dashboard',
    workerDashboard: 'Worker Dashboard',
    reportExportReady: 'Professional report opened. Choose Save as PDF from the print dialog.',
    duplicateReductionTitle: 'Clutter review',
    removedFromMenu: 'Removed from menu',
    mergedIntoFarmReports: 'Merged into Farm Reports',
    reportOwnerNameNote: 'Reports use the real logged-in account name, role, and current language direction.',
    compactFarmReports: 'Compact Farm Reports',
    diagnosisReportsMerged: 'Diagnosis Reports moved inside Farm Management',
    sensorHistoryModalNote: 'Detailed sensor history should stay in a modal instead of the main page.',
    workerDetailsModalNote: 'Long worker information should stay in Worker Details modal.',
    assistantLongAdviceNote: 'Long educational recommendations should stay inside Assistant.'
  },
  ar: {
    dashboardStart: 'لوحة التحكم',
    reportsCombinedNotice: 'تم دمج تقارير التشخيص داخل تقارير المزرعة لتقليل زحمة المنيو.',
    reportGeneratedBy: 'تم إنشاء التقرير بواسطة',
    farmName: 'اسم المزرعة',
    professionalReport: 'تقرير احترافي',
    saveAsPdfHint: 'استخدم نافذة الطباعة واختر Save as PDF.',
    openReport: 'فتح التقرير',
    printOrSavePdf: 'طباعة / حفظ PDF',
    ownerDashboard: 'لوحة تحكم المالك',
    farmManagerDashboard: 'لوحة تحكم مدير المزرعة',
    workerDashboard: 'لوحة تحكم العامل',
    reportExportReady: 'تم فتح التقرير الاحترافي. اختر Save as PDF من نافذة الطباعة.',
    duplicateReductionTitle: 'مراجعة تقليل التكرار',
    removedFromMenu: 'تمت إزالته من المنيو',
    mergedIntoFarmReports: 'تم دمجه داخل تقارير المزرعة',
    reportOwnerNameNote: 'التقارير تستخدم اسم الحساب الحقيقي والدور واتجاه اللغة الحالي.',
    compactFarmReports: 'تقارير المزرعة المختصرة',
    diagnosisReportsMerged: 'تم نقل تقارير التشخيص داخل إدارة المزرعة',
    sensorHistoryModalNote: 'تفاصيل سجل الحساسات الطويلة يفضل أن تكون داخل نافذة Modal.',
    workerDetailsModalNote: 'تفاصيل العامل الطويلة يفضل أن تكون داخل Worker Details Modal.',
    assistantLongAdviceNote: 'التوصيات التعليمية الطويلة يفضل أن تكون داخل المساعد.'
  }
};



const finalProductOrganizationTranslations = {
  en: {
    more: 'More',
    farm: 'Farm',
    farmSensors: 'Farm Sensors',
    farmSensorsSub: 'Sensor readings are now organized by sector inside Farm Management instead of a separate menu page.',
    sensorsMergedNotice: 'Farm sensors are organized inside Farm Management.',
    devicesAndSensors: 'Devices & Sensors',
    alertsSummary: 'Alerts Summary',
    reportsSummary: 'Reports Summary',
    tasksSummary: 'Tasks Summary',
    currentSensorReadings: 'Current Sensor Readings',
    sensorSource: 'Sensor Source',
    liveSource: 'Live',
    simulationSource: 'Simulation',
    serviceRequiredSource: 'Service Required',
    backendLatestSource: 'Backend latest reading',
    backendHistorySource: 'Backend history',
    manualSource: 'Manual input',
    latestBackendReading: 'Latest backend reading',
    latestSavedReading: 'Latest saved reading',
    manualReadingsMode: 'Manual readings',
    viewSensorHistory: 'View Sensor History',
    sensorHistoryCompact: 'Sensor history is available in a modal so it does not crowd the main page.',
    moreMenuTitle: 'More Tools',
    moreMenuSub: 'Less-used tools are grouped here to keep the mobile navigation clean.',
    reportsMovedNotice: 'Reports are available inside Farm Management as compact previews.',
    assistantLearningNotice: 'Long explanations, disease guide, and general recommendations are handled by the Assistant.',
    openAssistant: 'Open Assistant',
    viewFarmReports: 'View Farm Reports',
    compactDashboardNotice: 'Dashboard shows only the most important farm summary. Detailed data is inside Farm Management.',
    fullReportOpened: 'Full report opened in a printable professional view.',
    sectionOrganizationSummary: 'Product organization summary',
    removedFromMainMenu: 'Removed from main menu',
    mergedInsideFarmManagement: 'Merged inside Farm Management',
    movedInsideAssistant: 'Moved inside Assistant',
    convertedToModal: 'Converted to Modal / Bottom Sheet',
    mustStayVisible: 'Must stay visible'
  },
  ar: {
    more: 'المزيد',
    farm: 'المزرعة',
    farmSensors: 'حساسات المزرعة',
    farmSensorsSub: 'تم تنظيم قراءات الحساسات حسب القطاع داخل إدارة المزرعة بدل صفحة منفصلة في المنيو.',
    sensorsMergedNotice: 'حساسات المزرعة منظمة داخل إدارة المزرعة.',
    devicesAndSensors: 'الأجهزة والحساسات',
    alertsSummary: 'ملخص التنبيهات',
    reportsSummary: 'ملخص التقارير',
    tasksSummary: 'ملخص المهام',
    currentSensorReadings: 'قراءات الحساسات الحالية',
    sensorSource: 'مصدر الحساسات',
    liveSource: 'مباشر',
    simulationSource: 'محاكاة',
    serviceRequiredSource: 'يتطلب خدمة',
    backendLatestSource: 'آخر قراءة من الباك إند',
    backendHistorySource: 'سجل الباك إند',
    manualSource: 'إدخال يدوي',
    latestBackendReading: 'آخر قراءة من الباك إند',
    latestSavedReading: 'آخر قراءة محفوظة',
    manualReadingsMode: 'قراءات يدوية',
    viewSensorHistory: 'عرض سجل الحساسات',
    sensorHistoryCompact: 'سجل الحساسات متاح داخل نافذة حتى لا يزحم الصفحة الأساسية.',
    moreMenuTitle: 'أدوات إضافية',
    moreMenuSub: 'الأدوات الأقل استخدامًا تم تجميعها هنا للحفاظ على بساطة منيو الموبايل.',
    reportsMovedNotice: 'التقارير متاحة داخل إدارة المزرعة كمعاينة مختصرة.',
    assistantLearningNotice: 'الشرح الطويل ودليل الأمراض والتوصيات العامة أصبحت داخل المساعد.',
    openAssistant: 'فتح المساعد',
    viewFarmReports: 'عرض تقارير المزرعة',
    compactDashboardNotice: 'لوحة التحكم تعرض أهم ملخص فقط، والتفاصيل داخل إدارة المزرعة.',
    fullReportOpened: 'تم فتح التقرير الكامل في عرض احترافي قابل للطباعة.',
    sectionOrganizationSummary: 'ملخص تنظيم المنتج',
    removedFromMainMenu: 'تمت إزالته من المنيو الرئيسي',
    mergedInsideFarmManagement: 'تم دمجه داخل إدارة المزرعة',
    movedInsideAssistant: 'تم نقله داخل المساعد',
    convertedToModal: 'تم تحويله إلى نافذة / Bottom Sheet',
    mustStayVisible: 'لازم يفضل ظاهر'
  }
};

const LangContext = createContext(null);
const useLang = () => useContext(LangContext);

const initialReadings = [];
const initialSectors = [];
const initialPlants = [];
const initialTasks = [];

const ACCOUNT_DATA_KEYS = new Set([
  'sph_worker_accounts','sph_sectors','sph_readings','sph_plants','sph_tasks_global','sph_notifications',
  'sph_settings','sph_account_settings','sph_onboarding','sph_devices','sph_reports','sph_alerts','sph_farm_data',
  'sph_diagnoses','sph_images_history','sph_sensors_history'
]);
const accountScope = () => {
  try {
    const user = JSON.parse(localStorage.getItem('ecosense_user') || '{}');
    return String(user._id || user.id || user.email || localStorage.getItem('sph_user_email') || '').toLowerCase().trim();
  } catch { return String(localStorage.getItem('sph_user_email') || '').toLowerCase().trim(); }
};
const storageKeyFor = (key) => {
  if (!ACCOUNT_DATA_KEYS.has(key)) return key;
  const scope = accountScope();
  return scope ? `${key}__account__${scope}` : `${key}__account__anonymous`;
};
const getStore = (key, fallback) => { try { return JSON.parse(localStorage.getItem(storageKeyFor(key))) || fallback; } catch { return fallback; } };
const setStore = (key, value) => localStorage.setItem(storageKeyFor(key), JSON.stringify(value));
const clearAccountDataCache = ({includeAllAccounts=false}={}) => {
  const bases=[...ACCOUNT_DATA_KEYS];
  const current=accountScope();
  Object.keys(localStorage).forEach((key)=>{
    const matched=bases.some(base => key===base || key.startsWith(`${base}__account__`) || key.startsWith(`${base}_`));
    if(!matched) return;
    if(includeAllAccounts || !current || key.includes(`__account__${current}`) || key.endsWith(`_${current}`) || key===key.split('__account__')[0]) localStorage.removeItem(key);
  });
  sessionStorage.removeItem('ecosense_runtime_cache');
};
const currentUser = () => getStore('ecosense_user', { email: '', name: '', role: 'owner' });
const currentUserEmail = () => (currentUser()?.email || localStorage.getItem('sph_user_email') || '').toLowerCase();
const normalizeOwnerMarker = (value) => {
  if (value === undefined || value === null || value === '') return '';
  if (typeof value === 'object') return String(value._id || value.id || value.email || value.username || value.ownerId || '').toLowerCase().trim();
  return String(value).toLowerCase().trim();
};
const currentAccountMarkers = () => {
  const user = currentUser() || {};
  return new Set([
    user._id, user.id, user.email, user.username, user.ownerId, localStorage.getItem('sph_user_email')
  ].map(normalizeOwnerMarker).filter(Boolean));
};
const ownerMarkersFromRecord = (item = {}) => [
  item.ownerId, item.owner_id, item.owner, item.farmOwner, item.farmOwnerId, item.createdBy, item.createdById,
  item.userId, item.user_id, item.createdByUser, item.accountOwner, item.accountOwnerId,
  item.owner?.email, item.owner?.username, item.owner?._id, item.owner?.id,
  item.createdBy?.email, item.createdBy?.username, item.createdBy?._id, item.createdBy?.id,
  item.farmOwner?.email, item.farmOwner?._id, item.farmOwner?.id,
].map(normalizeOwnerMarker).filter(Boolean);
const recordBelongsToCurrentAccount = (item = {}) => {
  const current = currentAccountMarkers();
  const markers = ownerMarkersFromRecord(item);
  if (!markers.length) return false;
  return markers.some(m => current.has(m));
};
const filterWorkersForCurrentOwner = (list = []) => {
  const rows = Array.isArray(list) ? list : [];
  if (!rows.length) return [];
  // /api/users/workers is protected and should already be scoped by the backend token.
  // If owner markers are present, filter defensively. If the backend omits ownerId
  // fields from the response, trust the protected endpoint instead of hiding valid workers.
  const rowsWithOwnerMarkers = rows.filter((row) => ownerMarkersFromRecord(row).length > 0);
  if (!rowsWithOwnerMarkers.length) return rows;
  return rows.filter(recordBelongsToCurrentAccount);
};
const currentUserName = () => currentUser()?.name || currentUser()?.fullName || currentUserEmail();
const scopedKey = (key, email=currentUserEmail()) => `${key}_${String(email || 'anonymous').toLowerCase()}`;
const getUserStore = (key, fallback, email=currentUserEmail()) => getStore(scopedKey(key, email), fallback);
const setUserStore = (key, value, email=currentUserEmail()) => setStore(scopedKey(key, email), value);
const normalizeTaskStatus = (status='pending') => status === 'todo' ? 'pending' : status === 'done' ? 'completed' : status;
const getGlobalTasks = () => getStore('sph_tasks_global', []).map(t => ({...t, status: normalizeTaskStatus(t.status)}));
const setGlobalTasks = (tasks) => setStore('sph_tasks_global', tasks.map(t => ({...t, status: normalizeTaskStatus(t.status)})));
const workerAccounts = () => getStore('sph_worker_accounts', []);
const findWorkerAccountMeta = (identifier='') => {
  const id = String(identifier || '').trim().toLowerCase();
  if(!id) return null;
  return workerAccounts().find(acc => {
    const username = String(acc.username || '').toLowerCase();
    const email = String(acc.email || '').toLowerCase();
    return username === id || email === id;
  }) || null;
};
const findLocalAccount = (identifier='', passwordInput='') => {
  const id = String(identifier || '').trim().toLowerCase();
  if(!id) return null;
  return workerAccounts().find(acc => {
    const username = String(acc.username || acc.email || '').toLowerCase();
    const email = String(acc.email || '').toLowerCase();
    return (username === id || email === id) && String(acc.password || '') === String(passwordInput || '');
  }) || null;
};
const pickFirstText = (...values) => values.find(v => typeof v === 'string' && v.trim())?.trim() || '';
const extractWorkerCredentials = (root = {}, payload = {}, fallback = {}) => {
  const sources = [
    root.credentials, root.data?.credentials, root.loginCredentials, root.data?.loginCredentials,
    root.account, root.data?.account, payload.credentials, payload.loginCredentials, payload.account,
    payload, root, root.data
  ].filter(Boolean);
  const read = (keys) => {
    for (const src of sources) {
      if (!src || typeof src !== 'object') continue;
      for (const key of keys) {
        const value = src[key];
        if (typeof value === 'string' && value.trim()) return value.trim();
      }
    }
    return '';
  };
  const email = String(read(['email','workerEmail','loginEmail']) || fallback.email || '').trim().toLowerCase();
  const username = String(read(['username','userName','login','identifier']) || email || fallback.username || '').trim().toLowerCase();
  const password = read(['password','plainPassword','rawPassword','generatedPassword','temporaryPassword','tempPassword','workerPassword','loginPassword','initialPassword']) || fallback.password || '';
  return { email, username, password };
};
const verifyWorkerCredentials = async ({ email, username, password }) => {
  const identifiers = [...new Set([email, username].filter(Boolean))];
  if (!identifiers.length || !password) return false;
  for (const identifier of identifiers) {
    try {
      await authAPI.login({ identifier, email: identifier, username: identifier, password });
      return true;
    } catch (_) {
      // Try the next identifier without showing noisy validation messages.
    }
  }
  return false;
};
const normalizeIncomingAuthToken = (rawToken='') => {
  let token = String(rawToken || '').trim();
  if (!token) return '';
  try { token = decodeURIComponent(token); } catch {}
  token = token.trim().replace(/^['"]|['"]$/g, '');
  token = token.replace(/^Bearer\s+/i, '').trim();
  token = token.replace(/^token[:=]/i, '').trim();
  return token;
};

const setLoggedInUser = (user, token='') => {
  const role = normalizeRole(user.role || user.jobTitle || user.accountType, user.email || user.username);
  const account = {...user, role};
  const oldRaw = localStorage.getItem('ecosense_user');
  let oldKey = '';
  try { const oldUser = JSON.parse(oldRaw || '{}'); oldKey = String(oldUser._id || oldUser.id || oldUser.email || '').toLowerCase().trim(); } catch {}
  const newKey = String(account._id || account.id || account.email || account.username || '').toLowerCase().trim();
  if (oldKey && newKey && oldKey !== newKey) {
    clearAccountDataCache({ includeAllAccounts: true });
    sessionStorage.clear();
  }
  localStorage.setItem('ecosense_user', JSON.stringify(account));
  localStorage.setItem('sph_auth','true');
  // A successful backend login means the user can enter the app immediately.
  // Onboarding can still be reopened from Settings, but it must not block OAuth/login callbacks.
  localStorage.setItem('sph_onboarded','true');
  const accountEmail = String(account.email || account.username || '').toLowerCase().trim();
  if (accountEmail) localStorage.setItem('sph_user_email', accountEmail);
  else localStorage.removeItem('sph_user_email');
  const cleanToken = normalizeIncomingAuthToken(token);
  if (cleanToken && !['local-demo-token','local-worker-token'].includes(cleanToken)) {
    localStorage.setItem('ecosense_token', cleanToken);
    localStorage.setItem('sph_token', cleanToken);
  }
  localStorage.setItem('sph_role', role);
};
const routeForRole = (user={}) => {
  const role = normalizeRole(user.role || user.jobTitle, user.email || user.username);
  if (role === 'worker') return '/tasks';
  if (role === 'farm_manager') return '/farm-management';
  return '/dashboard';
};

const extractOAuthTokenFromUrl = () => {
  const tokenKeys = [
    'token','access_token','accessToken','authToken','auth_token','jwt','jwtToken',
    'id_token','idToken','bearer','Authorization','authorization','googleToken','google_token'
  ];

  const readDirect = (params) => {
    for (const key of tokenKeys) {
      const value = params.get(key);
      const token = normalizeIncomingAuthToken(value);
      if (token) return token;
    }
    // Some backends return JSON in a query param such as data/session/user.
    for (const key of ['data','session','auth','user','payload']) {
      const raw = params.get(key);
      if (!raw) continue;
      try {
        const decoded = decodeURIComponent(raw);
        const obj = JSON.parse(decoded);
        const nested = obj?.token || obj?.accessToken || obj?.access_token || obj?.jwt || obj?.authToken || obj?.data?.token || obj?.user?.token;
        const token = normalizeIncomingAuthToken(nested);
        if (token) return token;
      } catch {}
    }
    // Last-resort scan for a JWT-looking value in any param.
    for (const [, value] of params.entries()) {
      const token = normalizeIncomingAuthToken(value);
      if (/^eyJ[a-zA-Z0-9_-]+\./.test(token)) return token;
    }
    return '';
  };

  const query = new URLSearchParams(window.location.search || '');
  let token = readDirect(query);
  if (token) return token;

  const rawHash = String(window.location.hash || '').replace(/^#/, '');
  const hashQuery = rawHash.includes('?') ? rawHash.slice(rawHash.indexOf('?') + 1) : rawHash;
  token = readDirect(new URLSearchParams(hashQuery));
  if (token) return token;

  const pathParts = String(window.location.pathname || '').split('/').filter(Boolean);
  const possiblePathToken = pathParts.find(part => /^eyJ[a-zA-Z0-9_-]+\./.test(normalizeIncomingAuthToken(part)));
  return normalizeIncomingAuthToken(possiblePathToken || '');
};

const extractOAuthErrorFromUrl = () => {
  const params = new URLSearchParams(window.location.search || '');
  const hash = String(window.location.hash || '').replace(/^#/, '');
  const hashParams = new URLSearchParams(hash.includes('?') ? hash.slice(hash.indexOf('?') + 1) : hash);
  return params.get('error') || params.get('message') || hashParams.get('error') || hashParams.get('message') || '';
};
const taskPersonMarkers = (value) => {
  if (!value) return [];
  if (typeof value === 'object') {
    return [value._id, value.id, value.email, value.username, value.name, value.fullName, value.firstName && value.lastName ? `${value.firstName} ${value.lastName}` : value.firstName].filter(Boolean).map(v=>String(v).toLowerCase().trim());
  }
  return [String(value).toLowerCase().trim()];
};
const currentTaskAssigneeMarkers = () => {
  const user = currentUser() || {};
  return new Set([
    user._id, user.id, user.email, user.username, user.name, user.fullName, currentUserEmail(), currentUserName(),
    user.firstName && user.lastName ? `${user.firstName} ${user.lastName}` : user.firstName
  ].filter(Boolean).map(v=>String(v).toLowerCase().trim()));
};
const taskAssignedToCurrent = (task = {}) => {
  const current = currentTaskAssigneeMarkers();
  const markers = [
    ...taskPersonMarkers(task.assignedTo), ...taskPersonMarkers(task.assignee), ...taskPersonMarkers(task.worker),
    ...taskPersonMarkers(task.assignedWorker), ...taskPersonMarkers(task.assignedWorkerId), ...taskPersonMarkers(task.assignedToId),
    ...taskPersonMarkers(task.workerId), ...taskPersonMarkers(task.userId), ...taskPersonMarkers(task.assignedToEmail),
    ...taskPersonMarkers(task.workerEmail), ...taskPersonMarkers(task.email)
  ].filter(Boolean);
  return markers.some(m => current.has(m));
};
const normalizeTaskRecord = (task = {}, idx = 0) => {
  const assignedPerson = task.assignedTo || task.assignee || task.worker || task.assignedWorker || {};
  const sectorObj = task.sector || task.sectorId || task.assignedSector || {};
  const backendId = task._id || task.id || task.taskId || task.task_id || task.uuid || task.mongoId || task.backendId || '';
  return {
    ...task,
    backendId: looksLikeBackendId(backendId) ? backendId : '',
    id: looksLikeBackendId(backendId) ? backendId : `task-${idx + 1}`,
    title: task.title || task.name || task.taskName || '-',
    details: task.details || task.description || task.note || '',
    status: normalizeTaskStatus(task.status),
    priority: task.priority || task.level || 'Medium',
    assignedTo: assignedPerson.name || assignedPerson.fullName || task.assignedToName || task.workerName || (assignedPerson.firstName && assignedPerson.lastName ? `${assignedPerson.firstName} ${assignedPerson.lastName}` : assignedPerson.firstName) || task.assignedTo || '-',
    assignedToEmail: assignedPerson.email || task.assignedToEmail || task.workerEmail || task.email || '',
    workerId: task.workerId || task.assignedWorkerId || task.assignedToId || assignedPerson._id || assignedPerson.id || '',
    sectorId: sectorObj._id || sectorObj.id || task.sectorId || task.assignedSector || '',
    sector: sectorObj.name || task.sectorName || task.sector || '',
    createdAt: task.createdAt || task.created_at || '',
    dueDate: task.dueDate || task.due_date || '',
    service: true,
  };
};
const latestAutoReading = () => {
  const readings = getStore('sph_readings', initialReadings);
  return readings[0] || {};
};
const notificationStore = (email=currentUserEmail()) => getUserStore('sph_notifications', [], email);
const setNotificationStore = (items, email=currentUserEmail()) => setUserStore('sph_notifications', items, email);
const playNotificationSound = (type='warning') => {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    const gain = ctx.createGain();
    gain.connect(ctx.destination);
    const patterns = {
      success: [520, 660],
      warning: [620, 520, 620],
      danger: [740, 560, 740],
    };
    const pattern = patterns[type] || patterns.warning;
    const volume = type === 'danger' ? 0.16 : type === 'warning' ? 0.12 : 0.09;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(volume, ctx.currentTime + 0.025);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.72);
    pattern.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.14);
      osc.connect(gain);
      osc.start(ctx.currentTime + i * 0.14);
      osc.stop(ctx.currentTime + i * 0.14 + 0.10);
    });
    if (navigator.vibrate && type === 'danger') navigator.vibrate([120,80,120]);
  } catch {}
};
const sanitizeUserMessage = (message, lang='ar') => {
  const raw = typeof message === 'object' ? displayText(message, { lang }) : String(message || '');
  if (/deviceSerial|sectorId|معرف القطاع|تزويد السيرفر|server/i.test(raw)) {
    return lang === 'ar'
      ? 'تم التحليل بدون ربط بقطاع. اختر قطاعًا فقط إذا كنت تريد حفظ التشخيص داخل سجل القطاع.'
      : 'Diagnosis can run without a sector. Choose a sector only if you want to save it to sector history.';
  }
  if (/status code 429|too many requests|Request failed with status code 429/i.test(raw)) {
    return lang === 'ar'
      ? 'الخدمة مشغولة مؤقتًا بسبب كثرة الطلبات. انتظر لحظات ثم حاول مرة أخرى.'
      : 'The service is temporarily busy because of too many requests. Please wait and try again.';
  }
  if (/Unsupported cropType/i.test(raw)) {
    return lang === 'ar'
      ? 'الموديل الحالي يدعم فقط: نعناع، ذرة، فلفل، طماطم. اختر نوع النبات من القائمة.'
      : 'The model currently supports only Mint, Corn, Pepper, and Tomato. Choose the plant type from the list.';
  }
  if (/route not found|not found|404/i.test(raw)) {
    return lang === 'ar'
      ? 'هذه العملية تحتاج endpoint متاح من الباك إند. راجع الربط ثم حاول مرة أخرى.'
      : 'This action needs an available backend endpoint. Check the integration and try again.';
  }
  if (/Cannot read properties|undefined is not|is not a function|TypeError/i.test(raw)) {
    return lang === 'ar'
      ? 'حدث خطأ في عرض البيانات. حدّث الصفحة وحاول مرة أخرى.'
      : 'A display error occurred. Refresh the page and try again.';
  }
  if (/Network Error/i.test(raw)) {
    return lang === 'ar'
      ? 'تعذر الاتصال بالخدمة حاليًا. تأكد من الاتصال ثم حاول مرة أخرى.'
      : 'Could not connect to the service right now. Check your connection and try again.';
  }
  return raw || (lang === 'ar' ? 'تم تنفيذ الطلب.' : 'Request completed.');
};
const addUserNotification = (email, message, type='warning') => {
  const target = String(email || currentUserEmail()).toLowerCase();
  const uiMessage = sanitizeUserMessage(message, langFromDocument());
  const all = [{ id: Date.now(), time: new Date().toLocaleString(), message: uiMessage, type, read:false }, ...notificationStore(target)].slice(0, 40);
  setNotificationStore(all, target);
  if (target === currentUserEmail()) {
    if (getStore('sph_settings', {soundAlerts:true}).soundAlerts !== false) playNotificationSound(type);
    try { window.dispatchEvent(new CustomEvent('sph-notification', { detail: { message: uiMessage, type } })); } catch {}
  }
  return all;
};
const addNotification = (message, type='warning') => addUserNotification(currentUserEmail(), message, type);
const latestHomeAlert = (t) => {
  const n = notificationStore()[0];
  return n ? n.message : t('noCriticalAlerts');
};
const isDanger = (status='') => String(status).includes('High') || String(status).toLowerCase().includes('critical') || String(status).includes('خطر');

const currentRole = () => normalizeRole(localStorage.getItem('sph_role') || currentUser()?.role || currentUser()?.accountType, currentUser()?.email || currentUser()?.username);
const isOwner = () => currentRole() === 'owner';
const isFarmManager = () => currentRole() === 'farm_manager';
const isWorker = () => currentRole() === 'worker';
const canManageFarm = () => isOwner() || isFarmManager();
const roleLabel = (t) => isOwner() ? t('owner') : isFarmManager() ? t('farmManager') : t('worker');

const displayUserName = () => {
  const user = currentUser() || {};
  const raw = user.name || user.fullName || user.displayName || user.username || '';
  const blocked = ['Ecosense Owner','Ecosense User','Smart Plant Owner','Owner'];
  if(raw && !blocked.includes(String(raw).trim())) return raw;
  if(user.email) return String(user.email).split('@')[0].replace(/[._-]+/g,' ').replace(/\b\w/g, c=>c.toUpperCase());
  return localStorage.getItem('sph_profile_name') || 'Mahmoud';
};
const greetingKeyByTime = () => {
  const h = new Date().getHours();
  if (h < 12) return 'goodMorning';
  if (h < 17) return 'goodAfternoon';
  return 'goodEvening';
};
const signOutUser = () => {
  try { disconnectSocket(); } catch {}
  clearAccountDataCache({includeAllAccounts:true});
  ['sph_auth','ecosense_token','sph_token','sph_role','sph_user_email','ecosense_user','token','user_token','accessToken','sph_profile_name'].forEach(k => localStorage.removeItem(k));
  sessionStorage.clear();
  window.location.href = '/login';
};

const rolePermissions = {
  owner: { diagnose:true, reports:true, exportReports:true, deleteReports:true, deleteTasks:true, sensors:true, devices:true, alerts:true, tasks:true, assignTasks:true, workers:true, sectors:true, editOperationalSector:true, deleteSector:true, deleteDevice:true, managePermissions:true, ownerSensitive:true, settingsFarm:true },
  farm_manager: { diagnose:true, reports:true, exportReports:true, deleteReports:false, deleteTasks:true, sensors:true, devices:true, alerts:true, tasks:true, assignTasks:true, workersView:true, sectors:true, editOperationalSector:true, deleteSector:false, deleteDevice:false, managePermissions:false, ownerSensitive:false, settingsFarm:false },
  worker: { diagnose:false, reports:false, exportReports:false, deleteReports:false, deleteTasks:false, sensors:false, devices:false, alerts:true, tasks:true, assignTasks:false, workers:false, sectors:true, editOperationalSector:false, deleteSector:false, deleteDevice:false, managePermissions:false, ownerSensitive:false, settingsFarm:false }
};
const hasPermission = (permission) => !!(rolePermissions[currentRole()] || rolePermissions.worker)[permission];
const normalizeRole = (role, email='') => {
  const r = String(role || '').toLowerCase().trim();
  if (['farm_manager','farm manager','manager','مدير مزرعة','farm-manager'].includes(r)) return 'farm_manager';
  if (['worker','field_worker','field worker','farm_worker','farm worker','عامل','عامل زراعي','employee','staff','plant care worker','irrigation worker'].includes(r)) return 'worker';
  if (['owner','مالك','admin','administrator'].includes(r)) return 'owner';
  const e = String(email || '').toLowerCase();
  if(e.includes('manager')) return 'farm_manager';
  if(e.includes('worker')) return 'worker';
  return 'owner';
};

const asArray = (payload, keys=[]) => {
  if (Array.isArray(payload)) return payload;
  const knownKeys = [...new Set([...keys, 'data', 'docs', 'documents', 'list', 'values'])];
  for (const key of knownKeys) {
    const value = payload?.[key];
    if (Array.isArray(value)) return value;
  }
  if (Array.isArray(payload?.data)) return payload.data;
  for (const key of knownKeys) {
    const value = payload?.data?.[key];
    if (Array.isArray(value)) return value;
  }
  // Some APIs wrap arrays one level deeper, e.g. { data: { result: { history: [] } } }.
  const stack = [payload, payload?.data].filter(isPlainObject);
  const seen = new Set();
  while (stack.length) {
    const obj = stack.shift();
    if (!obj || seen.has(obj)) continue;
    seen.add(obj);
    for (const key of knownKeys) {
      const value = obj[key];
      if (Array.isArray(value)) return value;
      if (isPlainObject(value)) stack.push(value);
    }
  }
  return [];
};

const diagnosisRecordsFromPayload = (payload = {}) => asArray(payload, ['diagnoses','history','items','rows','records','reports','logs','sensorData','sensor_data','images','imageLogs','image_logs','results']);
// Backend getImageHistory returns: { success, totalRecords, currentPage, totalPages, data: images }.
// Keep image history parsing explicit so Sector Details / My Diagnoses never treat pagination metadata as records.
const imageHistoryRecordsFromPayload = (payload = {}) => asArray(payload, ['data','images','imageLogs','image_logs','items','rows','records','results']);
const diagnosisCountFromPayload = (payload = {}) => {
  const n = payload?.count ?? payload?.total ?? payload?.totalCount ?? payload?.totalRecords ?? payload?.data?.count ?? payload?.data?.total ?? payload?.data?.totalCount ?? payload?.data?.totalRecords;
  if (n !== undefined && n !== null) return Number(n) || 0;
  return diagnosisRecordsFromPayload(payload).length;
};
const normalizeDiagnosisType = (value='') => {
  const v = String(value || '').toLowerCase();
  if(v.includes('combined') || v.includes('with_image') || v.includes('image +') || v.includes('صورة +') || v.includes('دمج')) return 'Image + Sensors';
  if(v.includes('sensor') || v.includes('حساس') || v.includes('قيم')) return 'Sensors';
  if(v.includes('image') || v.includes('photo') || v.includes('صورة')) return 'Image';
  return value || 'Image + Sensors';
};
const firstNonEmpty = (...values) => values.find(v => {
  if (v === undefined || v === null || v === '') return false;
  if (typeof v === 'string') {
    const cleaned = v.trim().toLowerCase();
    if (!cleaned || ['unknown','not specified','undefined','null','غير محدد','غير معروف','-','—'].includes(cleaned)) return false;
  }
  return true;
});
const isPlainObject = (value) => value && typeof value === 'object' && !Array.isArray(value);
const pickObject = (...values) => values.find(isPlainObject) || {};
const cleanConfidenceValue = (value) => {
  if (isPlainObject(value)) {
    const nums = Object.values(value).map(Number).filter(v => !Number.isNaN(v));
    return nums.length ? Math.max(...nums) : 0;
  }
  const num = Number(value ?? 0);
  return Number.isNaN(num) ? 0 : (num > 1 ? num / 100 : num);
};
const normalizeSensorInput = (data = {}) => {
  const air = pickObject(data.air);
  const soil = pickObject(data.soil);
  const sectorObj = pickObject(data.sector, data.sectorId);
  const temperature = firstNonEmpty(data.temperature, data.airTemperature, data.air_temperature, data.temp, air.temperature, air.temp);
  // Current hardware sends air temperature, humidity, Soil moisture, and light only.
  // When the backend has no dedicated soil-temperature field, use the air temperature as a transparent fallback
  // so the AI model still receives a complete sensor payload instead of an empty soilTemp value.
  const soilTemp = firstNonEmpty(
    data.soilTemp, data.soil_temp, data.soilTemperature, data.soil_temperature, data.soilTempC, data.soil_temperature_c,
    data.rootZoneTemp, data.root_zone_temp, data.root_temp,
    soil.temperature, soil.temp, soil.temperatureC, soil.tempC,
    isPlainObject(soil.temperature) ? soil.temperature.value : undefined,
    isPlainObject(soil.temp) ? soil.temp.value : undefined,
    temperature
  );
  return {
    cropType: firstNonEmpty(data.cropType, data.crop_type, data.crop, data.plantType, data.plant_type, sectorObj.cropType, sectorObj.crop),
    temperature,
    humidity: firstNonEmpty(data.humidity, data.hum, air.humidity, air.hum),
    soilMoisture: firstNonEmpty(data.soilMoisture, data.soil_moisture, data.Soil, data.soil, data.moisture, soil.moisture, soil.soilMoisture, soil.Soil),
    soilTemp,
    soilTempEstimated: !firstNonEmpty(data.soilTemp, data.soil_temp, data.soilTemperature, data.soil_temperature, data.soilTempC, data.soil_temperature_c, data.rootZoneTemp, data.root_zone_temp, data.root_temp, soil.temperature, soil.temp, soil.temperatureC, soil.tempC) && firstNonEmpty(temperature, ''),
    light: firstNonEmpty(data.light, data.lightLevel, data.light_level),
    sectorId: firstNonEmpty(isPlainObject(data.sectorId) ? data.sectorId._id || data.sectorId.id : data.sectorId, data.sector_id, data.assignedSectorId, sectorObj._id, sectorObj.id),
    sectorName: firstNonEmpty(data.sectorName, data.sector_name, data.sector, data.assignedSector, sectorObj.name, sectorObj.sectorName),
  };
};
const mergeDiagnosisData = (raw = {}) => {
  const root = isPlainObject(raw?.data) ? raw.data : raw;
  const nested = pickObject(
    raw?.result, raw?.prediction, raw?.analysis, raw?.diagnosisResult, raw?.aiResult, raw?.rawResult,
    root?.result, root?.prediction, root?.analysis, root?.analysisResult, root?.analysis_result, root?.diagnosisResult, root?.aiResult, root?.rawResult,
    root?.imageAnalysis, root?.image_analysis
  );
  return { root, nested, data: { ...root, ...nested } };
};
const normalizeDiagnosisRecord = (raw = {}, index = 0) => {
  const { root, nested, data } = mergeDiagnosisData(raw);
  const directInput = pickObject(data.sensorReadings, data.sensor_readings, data.readings, data.input_data, data.inputData, data.sensors, root.sensorReadings, root.readings, root.sensors);
  const input = { ...normalizeSensorInput(data), ...normalizeSensorInput(directInput), ...directInput };
  const diagnosisObj = isPlainObject(data.diagnosis) ? data.diagnosis : pickObject(data.diagnosis_details, data.diagnosisDetails, nested.diagnosis);
  const diagnosis = diagnosisObj && Object.keys(diagnosisObj).length
    ? (diagnosisObj.explanation || diagnosisObj.summary || diagnosisObj.primary_issue || diagnosisObj.secondary_issue || diagnosisObj.text || '')
    : (data.diagnosis || data.diagnosisSummary || data.summary || data.message || data.description || '');
  const sectorObj = pickObject(
    data.sector, data.sectorId, data.assignedSector, data.farmSector,
    root.sector, root.sectorId, root.assignedSector
  );
  const imageObj = pickObject(data.image, data.imageData, data.plantImage, root.image, root.file);
  const imageUrl = firstNonEmpty(
    data.imageUrl, data.image_url, data.photoUrl, data.photo_url, data.url, data.secure_url,
    data.snapshotUrl, data.snapshot_url, data.fileUrl, data.file_url,
    imageObj.url, imageObj.imageUrl, imageObj.image_url, imageObj.secure_url,
    data.image_preview, root.imageUrl, root.image_url
  ) || '';
  const createdAt = firstNonEmpty(
    data.createdAt, data.created_at, data.timestamp, data.dateTime, data.date_time,
    data.diagnosedAt, data.diagnosed_at, data.updatedAt, data.time, data.date, root.createdAt, root.created_at
  ) || new Date().toISOString();
  const hasInput = Object.values(input || {}).some(v => v !== undefined && v !== null && v !== '');
  const type = normalizeDiagnosisType(
    data.analysisType || data.analysis_type || data.diagnosis_mode || data.mode || data.type || data.sourceType ||
    (imageUrl && hasInput ? 'Image + Sensors' : imageUrl ? 'Image' : 'Sensors')
  );
  const confidence = cleanConfidenceValue(firstNonEmpty(
    data.confidence, data.final_confidence, data.finalConfidence, data.confidence_breakdown,
    data.probability, data.score, data.health_score, nested.confidence
  ));
  return {
    id: firstNonEmpty(
      data.imageLogId, data.image_log_id, data.imageLog?._id, data.imageLog?.id,
      data._id, data.id, data.diagnosisId, data.diagnosis_id, data.reportId, data.logId,
      data.imageId, data.image_id, data.sensorLogId, data.sensor_log_id, `diagnosis-${index}`
    ),
    backendId: firstNonEmpty(
      data.imageLogId, data.image_log_id, data.imageLog?._id, data.imageLog?.id,
      data._id, data.id, data.diagnosisId, data.diagnosis_id, data.reportId, data.logId,
      data.imageId, data.image_id, data.sensorLogId, data.sensor_log_id, ''
    ),
    imageLogId: firstNonEmpty(data.imageLogId, data.image_log_id, data.imageLog?._id, data.imageLog?.id, data.imageId, data.image_id, ''),
    sensorLogId: firstNonEmpty(data.sensorLogId, data.sensor_log_id, data.sensorId, data.sensor_id, ''),
    number: data.number || data.serial || index + 1,
    date: createdAt,
    analysisType: type,
    sectorId: firstNonEmpty(
      isPlainObject(data.sectorId) ? data.sectorId._id || data.sectorId.id : data.sectorId,
      data.sector_id, sectorObj._id, sectorObj.id, input.sectorId, ''
    ),
    sectorName: firstNonEmpty(data.sectorName, data.sector_name, sectorObj.name, sectorObj.sectorName, input.sectorName, input.sector, '-'),
    cropType: firstNonEmpty(
      data.cropType, data.crop_type, data.plantType, data.plant_type, data.crop,
      sectorObj.cropType, sectorObj.crop, sectorObj.plantType,
      root.cropType, root.crop_type, root.plantType, root.plant_type, root.crop,
      nested.cropType, nested.crop_type, nested.plantType, nested.plant_type, nested.crop, ''
    ),
    imageUrl,
    final_status: firstNonEmpty(data.final_status, data.finalStatus, data.finalStatusPrimary, data.status, data.condition, data.healthStatus, nested.final_status, '-'),
    confidence,
    disease_name: firstNonEmpty(data.disease_name, data.diseaseName, data.disease, data.predicted_disease, data.predictedDisease, data.visual_problem, data.problem, '-'),
    diagnosis: diagnosis || '-',
    recommendations: Array.isArray(data.recommendations) ? data.recommendations : Array.isArray(data.recommendations_ar) ? data.recommendations_ar : data.recommendations ? [data.recommendations] : [],
    actions: Array.isArray(data.actions) ? data.actions : data.actions ? [data.actions] : [],
    sensor_status: firstNonEmpty(data.sensor_status, data.sensorStatus, data.sensors_status, data.sensorExplanation, '-'),
    image_status: firstNonEmpty(data.image_status, data.imageStatus, data.visual_status, data.imageExplanation, '-'),
    input_data: input,
    image_analysis: data.image_analysis || data.imageAnalysis || nested.image_analysis || nested.imageAnalysis || {},
    historySource: data._historySource || data.historySource || data.source || '',
    raw: data,
  };
};
const normalizeSensorHistoryRecord = (raw = {}, index = 0) => {
  const { root, nested, data } = mergeDiagnosisData(raw);
  return normalizeDiagnosisRecord({
    ...root,
    ...nested,
    ...data,
    _historySource: raw._historySource || raw.historySource || 'sensors',
    analysisType: 'Sensors',
    type: 'Sensors',
    sensorReadings: pickObject(data.sensorReadings, data.sensor_readings, data.readings, data.input_data, data.sensors, data),
    imageUrl: '',
  }, index);
};
const normalizeImageHistoryRecord = (raw = {}, index = 0) => {
  const { root, nested, data } = mergeDiagnosisData(raw);
  const input = pickObject(data.sensorReadings, data.sensor_readings, data.readings, data.input_data, data.sensors);
  const populatedSector = pickObject(data.sectorId, root.sectorId, data.sector, root.sector);
  const imageUrl = firstNonEmpty(data.imageUrl, data.image_url, data.url, data.secure_url, data.fileUrl, data.file_url, data.snapshotUrl, data.snapshot_url, root.imageUrl, root.image_url);
  return normalizeDiagnosisRecord({
    ...root,
    ...nested,
    ...data,
    sectorId: isPlainObject(data.sectorId) ? data.sectorId : (populatedSector._id || populatedSector.id ? populatedSector : data.sectorId),
    sectorName: firstNonEmpty(data.sectorName, data.sector_name, populatedSector.name, populatedSector.sectorName),
    cropType: firstNonEmpty(data.cropType, data.plantType, data.crop, populatedSector.cropType, populatedSector.crop, populatedSector.plantType),
    imageLogId: firstNonEmpty(data.imageLogId, data.image_log_id, root.imageLogId, root.image_log_id, root._id, root.id, data._id, data.id),
    _historySource: raw._historySource || raw.historySource || 'images',
    analysisType: Object.keys(input).length ? 'Image + Sensors' : 'Image',
    type: Object.keys(input).length ? 'Image + Sensors' : 'Image',
    imageUrl,
    sensorReadings: input,
  }, index);
};
const diagnosisHistoryRuntimeCache = { time: 0, key: '', value: null, promise: null };
const diagnosisRecordKey = (row = {}) => {
  const id = firstNonEmpty(row.imageLogId, row.image_log_id, row.imageLog?._id, row.imageLog?.id, row._id, row.id, row.diagnosisId, row.diagnosis_id, row.reportId, row.imageId, row.image_id, row.sensorLogId, row.sensor_log_id);
  if (id) return `id:${id}`;
  return [
    row.analysisType || row.type || '',
    row.date || row.createdAt || '',
    row.final_status || '',
    row.disease_name || '',
    row.sectorId || row.sectorName || '',
    row.imageUrl || '',
  ].map(v => String(v || '').trim().toLowerCase()).join('|');
};

const loadCombinedDiagnosisHistory = async (params = {}) => {
  const key = JSON.stringify(params || {});
  const now = Date.now();
  const force = !!params.force || !!params._force;
  const safeParams = { ...(params || {}) };
  delete safeParams.force;
  delete safeParams._force;

  // Shared throttle across dashboard, My Diagnoses, details, and save verification.
  // This prevents several components from hitting history endpoints at the same time.
  if (!force && diagnosisHistoryRuntimeCache.value && diagnosisHistoryRuntimeCache.key === key && now - diagnosisHistoryRuntimeCache.time < 12000) {
    return diagnosisHistoryRuntimeCache.value;
  }
  if (!force && diagnosisHistoryRuntimeCache.promise && diagnosisHistoryRuntimeCache.key === key) {
    return diagnosisHistoryRuntimeCache.promise;
  }

  const promise = (async()=>{
    // Image diagnoses are persisted by POST /images/upload and read from GET /images/history.
    // Sensor diagnoses are persisted by POST /sensors/analyze/:sectorId and read from GET /sensors/history.
    const buildRowsFromResponses = (sensorRes, imageRes, offset = 0) => {
      const sensorRows = sensorRes.status === 'fulfilled'
        ? diagnosisRecordsFromPayload(sensorRes.value.data).map((item, index) => normalizeSensorHistoryRecord({ ...item, historySource: 'sensors' }, index + offset))
        : [];
      const imageRows = imageRes.status === 'fulfilled'
        ? imageHistoryRecordsFromPayload(imageRes.value.data).map((item, index) => normalizeImageHistoryRecord({ ...item, historySource: 'images' }, index + offset + sensorRows.length))
        : [];
      return { sensorRows, imageRows };
    };

    const [sensorRes, imageRes] = await Promise.allSettled([
      sensorsAPI.getHistory(safeParams),
      imagesAPI.getHistory(safeParams),
    ]);
    let { sensorRows, imageRows } = buildRowsFromResponses(sensorRes, imageRes);
    const errors = [sensorRes, imageRes].filter(r => r.status === 'rejected').map(r => r.reason);

    // Do not fan-out history requests across every sector here. The backend already supports:
    // GET /sensors/history, GET /images/history, and sector-scoped ?sectorId=... when a sector is selected.
    // Keeping history to these direct calls prevents repeated request bursts and 429 rate-limit errors.

    const seen = new Set();
    const rows = [...sensorRows, ...imageRows]
      .filter(Boolean)
      .filter((row) => {
        const k = diagnosisRecordKey(row);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      })
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
    const value = {
      rows,
      count: rows.length,
      errors,
      live: sensorRes.status === 'fulfilled' || imageRes.status === 'fulfilled' || rows.length > 0,
      endpoints: {
        sensors: sensorRes.status === 'fulfilled' ? `GET /sensors/history${safeParams.sectorId ? '?sectorId=SECTOR_ID' : ''}` : null,
        images: imageRes.status === 'fulfilled' ? `GET /images/history${safeParams.sectorId ? '?sectorId=SECTOR_ID' : ''}` : null,
      },
    };
    diagnosisHistoryRuntimeCache.time = Date.now();
    diagnosisHistoryRuntimeCache.value = value;
    diagnosisHistoryRuntimeCache.key = key;
    return value;
  })();

  diagnosisHistoryRuntimeCache.promise = promise;
  diagnosisHistoryRuntimeCache.key = key;
  try {
    return await promise;
  } finally {
    diagnosisHistoryRuntimeCache.promise = null;
  }
};

const diagnosisRecordToResult = (record = {}) => ({
  final_status: record.final_status,
  confidence: record.confidence,
  disease_name: record.disease_name,
  diagnosis: record.diagnosis,
  diagnosis_mode: record.analysisType,
  image_preview: record.imageUrl,
  image_status: record.image_status,
  sensor_status: record.sensor_status,
  input_data: record.input_data || {},
  image_analysis: record.image_analysis || {},
  recommendations: record.recommendations || [],
  actions: record.actions || [],
});
const hasDiagnosisShape = (payload = {}) => {
  const { data } = mergeDiagnosisData(payload);
  return Boolean(
    data.final_status || data.finalStatus || data.status || data.condition ||
    data.diagnosis || data.diagnosisResult || data.analysis || data.result || data.prediction ||
    data.disease_name || data.diseaseName || data.predicted_disease || data.confidence || data.health_score
  );
};
const extractImageUrlFromPayload = (payload = {}) => {
  const { root, data } = mergeDiagnosisData(payload);
  const imageObj = pickObject(data.image, data.imageData, root.image, root.file, data.file);
  return firstNonEmpty(
    data.imageUrl, data.image_url, data.url, data.secure_url, data.fileUrl, data.file_url,
    data.snapshotUrl, data.snapshot_url, data.photoUrl, data.photo_url,
    imageObj.url, imageObj.imageUrl, imageObj.image_url, imageObj.secure_url,
    root.imageUrl, root.image_url
  ) || '';
};
const sensorPayloadForBackend = (form = {}) => ({
  ...(firstNonEmpty(form.deviceSerial, form.device_serial, '') ? { deviceSerial: String(firstNonEmpty(form.deviceSerial, form.device_serial, '')), device_serial: String(firstNonEmpty(form.deviceSerial, form.device_serial, '')) } : {}),
  temp: firstNonEmpty(form.temperature, form.temp, ''),
  temperature: firstNonEmpty(form.temperature, form.temp, ''),
  hum: firstNonEmpty(form.humidity, form.hum, ''),
  humidity: firstNonEmpty(form.humidity, form.hum, ''),
  Soil: firstNonEmpty(form.soilMoisture, form.Soil, form.soil, ''),
  soil: firstNonEmpty(form.soilMoisture, form.Soil, form.soil, ''),
  soilMoisture: firstNonEmpty(form.soilMoisture, form.Soil, form.soil, ''),
  soil_moisture: firstNonEmpty(form.soilMoisture, form.Soil, form.soil, ''),
  soilTemp: firstNonEmpty(form.soilTemp, form.soil_temp, form.soilTemperature, ''),
  soilTemperature: firstNonEmpty(form.soilTemp, form.soil_temp, form.soilTemperature, ''),
  soil_temp: firstNonEmpty(form.soilTemp, form.soil_temp, form.soilTemperature, ''),
  light: firstNonEmpty(form.light, form.lightLevel, ''),
  ...(firstNonEmpty(form.cropType, form.crop, '') ? { cropType: firstNonEmpty(form.cropType, form.crop, ''), crop: firstNonEmpty(form.cropType, form.crop, '') } : {}),
  sectorId: firstNonEmpty(form.sectorId, form.sector_id, ''),
  sector_id: firstNonEmpty(form.sectorId, form.sector_id, ''),
  sectorName: firstNonEmpty(form.sectorName, form.sector, ''),
  sector_name: firstNonEmpty(form.sectorName, form.sector, ''),
});
const buildSensorUploadFormData = (form = {}) => {
  const fd = new FormData();
  const payload = sensorPayloadForBackend(form);
  Object.entries(payload).forEach(([key,value]) => {
    if (value !== undefined && value !== null && value !== '') fd.append(key, String(value));
  });
  return fd;
};
const uploadManualSensorReading = async (form = {}) => {
  const response = await sensorsAPI.upload(buildSensorUploadFormData(form));
  return { response, method: 'POST /sensors/upload multipart/form-data', ok: true };
};


const SUPPORTED_CROP_OPTIONS = [
  { value: 'Mint', ar: 'نعناع', en: 'Mint' },
  { value: 'Corn', ar: 'ذرة', en: 'Corn' },
  { value: 'Pepper', ar: 'فلفل', en: 'Pepper' },
  { value: 'Tomato', ar: 'طماطم', en: 'Tomato' },
];
const normalizeCropTypeForModel = (value = '') => {
  const raw = String(value || '').trim();
  const v = raw.toLowerCase();
  const map = {
    mint: 'Mint', 'نعناع': 'Mint', 'النعناع': 'Mint', 'مينت': 'Mint',
    corn: 'Corn', 'ذرة': 'Corn', 'الذرة': 'Corn', 'درة': 'Corn',
    pepper: 'Pepper', 'فلفل': 'Pepper', 'الفلفل': 'Pepper',
    tomato: 'Tomato', 'طماطم': 'Tomato', 'الطماطم': 'Tomato', 'بندورة': 'Tomato',
  };
  return map[v] || SUPPORTED_CROP_OPTIONS.find(c => c.value.toLowerCase() === v)?.value || '';
};
const localCropLabel = (value = '', lang = 'ar') => {
  const normalized = normalizeCropTypeForModel(value) || String(value || '');
  const found = SUPPORTED_CROP_OPTIONS.find(c => c.value === normalized);
  return found ? (lang === 'ar' ? found.ar : found.en) : (value || '');
};
function CropTypeSelect({ value, onChange, lang='ar', required=false, allowEmpty=true }) {
  const normalized = normalizeCropTypeForModel(value);
  return <select value={normalized} required={required} onChange={e=>onChange(e.target.value)}>
    {allowEmpty && <option value="">{lang==='ar'?'اختر نوع النبات':'Choose crop type'}</option>}
    {SUPPORTED_CROP_OPTIONS.map(c=><option key={c.value} value={c.value}>{lang==='ar'?c.ar:c.en}</option>)}
  </select>;
}

const numberOrRaw = (value) => {
  if (value === undefined || value === null || value === '') return '';
  const n = Number(value);
  return Number.isFinite(n) ? n : value;
};

const manualSensorDiagnosisBody = (form = {}) => {
  const sectorId = firstNonEmpty(form.sectorId, form.sector_id, '');
  const deviceSerial = firstNonEmpty(
    form.deviceSerial,
    form.device_serial,
    form.deviceId,
    form.device_id,
    ''
  );
  const cropType = normalizeCropTypeForModel(firstNonEmpty(form.cropType, form.crop, ''));
  return {
    ...(sectorId ? { sectorId: String(sectorId), sector_id: String(sectorId) } : {}),
    ...(deviceSerial ? { deviceSerial: String(deviceSerial), device_serial: String(deviceSerial) } : {}),
    ...(cropType ? { cropType: String(cropType) } : {}),
    temperature: numberOrRaw(firstNonEmpty(form.temperature, form.temp, '')),
    humidity: numberOrRaw(firstNonEmpty(form.humidity, form.hum, '')),
    soilMoisture: numberOrRaw(firstNonEmpty(form.soilMoisture, form.Soil, form.soil, '')),
    soilTemp: numberOrRaw(firstNonEmpty(form.soilTemp, form.soil_temp, form.soilTemperature, form.temperature, form.temp, '')),
    light: String(firstNonEmpty(form.light, form.lightLevel, '')),
  };
};

const isMissingEndpointError = (err) => [404, 405, 501].includes(Number(err?.response?.status));
const manualSensorsEndpointMessage = (lang='ar') => lang === 'ar'
  ? 'تحليل القيم اليدوية غير متاح حاليًا من الخدمة. استخدم قراءة محفوظة من القطاع أو تواصل مع مسؤول النظام.'
  : 'Manual sensor diagnosis is not available from the service right now. Use a saved sector reading or contact the system owner.';
const combinedEndpointMessage = (lang='ar') => lang === 'ar'
  ? 'تحليل الصورة مع القراءات غير متاح حاليًا من الخدمة.'
  : 'Combined image and sensor diagnosis is not available from the service right now.';

const buildBackendCombinedFormData = (imageFile, form = {}) => {
  const fd = new FormData();
  fd.append('image', imageFile);
  const sectorId = firstNonEmpty(form.sectorId, form.sector_id, '');
  if (sectorId) fd.append('sectorId', String(sectorId));
  const deviceSerial = firstNonEmpty(form.deviceSerial, form.device_serial, '');
  if (deviceSerial) fd.append('deviceSerial', String(deviceSerial));
  const cropType = normalizeCropTypeForModel(firstNonEmpty(form.cropType, form.crop, ''));
  if (cropType) fd.append('cropType', String(cropType));
  fd.append('temperature', String(firstNonEmpty(form.temperature, form.temp, '')));
  fd.append('humidity', String(firstNonEmpty(form.humidity, form.hum, '')));
  fd.append('soilMoisture', String(firstNonEmpty(form.soilMoisture, form.Soil, form.soil, '')));
  fd.append('soilTemp', String(firstNonEmpty(form.soilTemp, form.soil_temp, form.soilTemperature, '')));
  const lightValue = firstNonEmpty(form.light, form.lightLevel, '');
  if (lightValue) fd.append('light', String(lightValue));
  return fd;
};

const buildModelImageFormData = (imageFile, form = {}) => {
  const fd = new FormData();
  // External model compatibility only. Backend upload/combined endpoints must use "image".
  fd.append('file', imageFile);
  const cropType = normalizeCropTypeForModel(firstNonEmpty(form.cropType, form.crop, ''));
  if (cropType) fd.append('cropType', String(cropType));
  fd.append('temperature', String(firstNonEmpty(form.temperature, form.temp, '')));
  fd.append('humidity', String(firstNonEmpty(form.humidity, form.hum, '')));
  fd.append('soilMoisture', String(firstNonEmpty(form.soilMoisture, form.Soil, form.soil, '')));
  fd.append('soilTemp', String(firstNonEmpty(form.soilTemp, form.soil_temp, form.soilTemperature, '')));
  fd.append('light', String(firstNonEmpty(form.light, form.lightLevel, '')));
  return fd;
};
const buildDiagnosisHistoryPayload = (result, form = {}, diagnosisMode = 'combined', imageUrl = '') => {
  const sectors = getStore('sph_sectors', []);
  const matchedSector = sectors.find(s => String(s.id) === String(form.sectorId || '')) ||
    sectors.find(s => String(s.name || '').toLowerCase() === String(form.sectorName || '').toLowerCase()) ||
    {};
  return {
    sectorId: form.sectorId || matchedSector.id || '',
    sectorName: form.sectorName || matchedSector.name || '',
    cropType: firstNonEmpty(result.cropType, result.crop_type, result.plantType, result.plant_type, ''),
    analysisType: normalizeDiagnosisType(diagnosisMode),
    final_status: result.final_status,
    confidence: result.confidence,
    disease_name: result.disease_name,
    diagnosis: result.diagnosis,
    recommendations: result.recommendations || [],
    actions: result.actions || [],
    sensor_status: result.sensor_status,
    image_status: result.image_status,
    sensorReadings: result.input_data || form,
    input_data: result.input_data || form,
    imageUrl: imageUrl || result.image_preview || result.imageUrl || '',
    image_analysis: result.image_analysis || {},
    rawResult: result.raw_response || result.raw || result,
    createdAt: new Date().toISOString(),
  };
};
const historyHasLikelyRecord = (rows = [], payload = {}) => {
  const targetTime = new Date(payload.createdAt || Date.now()).getTime();
  const payloadType = normalizeDiagnosisType(payload.analysisType || payload.diagnosis_mode || '');
  const payloadStatus = String(payload.final_status || payload.status || '').toLowerCase().trim();
  const payloadDisease = String(payload.disease_name || payload.diseaseName || '').toLowerCase().trim();
  const payloadSectorId = String(payload.sectorId || '').toLowerCase().trim();
  const payloadSectorName = String(payload.sectorName || '').toLowerCase().trim();
  const payloadImage = String(payload.imageUrl || '').trim();
  return rows.some(row => {
    const rowTime = new Date(row.date || row.createdAt || row.updatedAt || 0).getTime();
    const closeTime = Number.isFinite(rowTime) && Math.abs(rowTime - targetTime) < 1000 * 60 * 45;
    const sameStatus = !payloadStatus || String(row.final_status || row.status || '').toLowerCase().trim() === payloadStatus;
    const rowType = normalizeDiagnosisType(row.analysisType || row.type || row.historySource || '');
    const payloadTypeText = String(payloadType || '').toLowerCase();
    const rowTypeText = String(rowType || '').toLowerCase();
    const sameType = !payloadTypeText || !rowTypeText || rowTypeText === payloadTypeText ||
      (payloadTypeText.includes('image') && rowTypeText.includes('image')) ||
      (payloadTypeText.includes('sensor') && rowTypeText.includes('sensor')) ||
      (payloadTypeText.includes('combined') && (rowTypeText.includes('image') || rowTypeText.includes('sensor')));
    const sameSector = !payloadSectorId && !payloadSectorName
      ? true
      : String(row.sectorId || '').toLowerCase() === payloadSectorId ||
        String(row.sectorName || '').toLowerCase() === payloadSectorName ||
        (payloadSectorName && String(row.sectorName || '').toLowerCase().includes(payloadSectorName));
    const sameDisease = !payloadDisease || String(row.disease_name || row.diseaseName || '').toLowerCase().trim() === payloadDisease || String(row.disease_name || row.diseaseName || '').toLowerCase().includes(payloadDisease);
    const sameImage = !payloadImage || String(row.imageUrl || '').trim() === payloadImage || Boolean(row.imageUrl && payloadImage);
    // Backend history is the source of truth. For sector-linked saves, a fresh row from the same sector is enough
    // even if the backend uses different field names for final_status/disease.
    const freshSectorRow = sameSector && closeTime && sameType;
    return (closeTime && sameStatus && sameType && sameSector && sameDisease && sameImage) || freshSectorRow;
  });
};
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const verifyHistoryContainsDiagnosis = async (payload = {}, { retries = 3, delay = 900, minCount = null } = {}) => {
  let lastRows = [];
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const verifyParams = payload.sectorId ? { _force: true, sectorId: payload.sectorId, limit: 200 } : { _force: true, limit: 200 };
      const { rows, count, live } = await loadCombinedDiagnosisHistory(verifyParams);
      lastRows = rows || [];
      const countLooksUpdated = Number.isFinite(Number(minCount)) && Number(count || rows.length || 0) > Number(minCount);
      if (historyHasLikelyRecord(rows, payload) || countLooksUpdated) return { ok: true, rows, count, live, matchedBy: countLooksUpdated ? 'count-increased' : 'record-match' };
    } catch (err) {
      if (attempt === retries) return { ok: false, error: getApiError(err), rows: lastRows };
    }
    if (attempt < retries) await wait(delay * (attempt + 1));
  }
  return { ok: false, rows: lastRows };
};

const buildImageSaveFormData = async (result = {}, form = {}, diagnosisMode = 'image', imageUrl = '') => {
  const fd = new FormData();
  const imageFile = await imageFileFromDiagnosisResult(result, imageUrl);
  fd.append('image', imageFile);
  const sectorId = firstNonEmpty(form.sectorId, form.sector_id, result.sectorId, result.sector_id, '');
  if (sectorId) fd.append('sectorId', String(sectorId));
  const deviceSerial = firstNonEmpty(form.deviceSerial, form.device_serial, result.deviceSerial, result.device_serial, '');
  if (deviceSerial) fd.append('deviceSerial', String(deviceSerial));
  return fd;
};

const persistViaDocumentedHistoryEndpoints = async (result = {}, form = {}, diagnosisMode = 'combined', imageUrl = '') => {
  const mode = normalizeDiagnosisType(diagnosisMode || result.__diagnosisMode || result.diagnosis_mode || 'combined');
  const lower = String(mode).toLowerCase();
  const saves = [];

  if (lower.includes('sensor')) {
    if (!form.sectorId) throw new Error('missing-sector-id-for-sensor-analysis-save');
    const response = await sensorsAPI.analyze(form.sectorId);
    saves.push({ response, endpoint: 'POST /sensors/analyze/:sectorId', historyEndpoint: 'GET /sensors/history' });
  }

  if (lower.includes('image')) {
    const fd = await buildImageSaveFormData(result, form, mode, imageUrl);
    const response = await imagesAPI.upload(fd);
    saves.push({ response, endpoint: 'POST /images/upload', historyEndpoint: 'GET /images/history' });
  }

  if (!saves.length) throw new Error('unsupported-diagnosis-mode-for-history-save');
  return {
    response: saves[saves.length - 1].response,
    saves,
    endpoint: saves.map(s => s.endpoint).join(' + '),
    historyEndpoint: [...new Set(saves.map(s => s.historyEndpoint))].join(' + '),
  };
};

const callBackendSavedEvent = () => {
  try { window.dispatchEvent(new Event('ecosense-diagnoses-updated')); } catch {}
};
const saveDiagnosisToService = async (result, form = {}, diagnosisMode = 'combined', imageUrl = '') => {
  const payload = buildDiagnosisHistoryPayload(result, form, diagnosisMode, imageUrl);
  const mode = normalizeDiagnosisType(diagnosisMode || result.__diagnosisMode || result.diagnosis_mode || 'combined');

  const acceptDocumentedSave = async (extra = {}) => {
    // The backend endpoints themselves are the save operation:
    // Image -> POST /images/upload, Sensors -> POST /sensors/analyze/:sectorId.
    // Do not create a local/fake diagnosis record; My Diagnoses must read from backend history only.
    diagnosisHistoryRuntimeCache.time = 0;
    diagnosisHistoryRuntimeCache.value = null;
    diagnosisHistoryRuntimeCache.promise = null;

    let verification = { ok: true, skipped: true, rows: [] };
    if (payload.sectorId) {
      verification = await verifyHistoryContainsDiagnosis(payload, { retries: 4, delay: 700 });
    }

    diagnosisHistoryRuntimeCache.time = 0;
    diagnosisHistoryRuntimeCache.value = null;
    diagnosisHistoryRuntimeCache.promise = null;
    callBackendSavedEvent();
    const params = payload.sectorId ? { _force: true, sectorId: payload.sectorId, limit: 200 } : { _force: true, limit: 200 };
    loadCombinedDiagnosisHistory(params).catch(() => {});

    if (payload.sectorId && !verification.ok) {
      return { ok: false, data: payload, verified: false, backendAccepted: true, verification, ...extra };
    }
    return { ok: true, data: payload, verified: Boolean(payload.sectorId ? verification.ok : true), backendAccepted: true, verification, ...extra };
  };

  // If Analyze already used the saving backend endpoint successfully, treat it as saved.
  if (result.__backendSaved) {
    return acceptDocumentedSave({
      alreadySaved: true,
      endpoint: result.__backendEndpoint,
      historyEndpoint: result.__historyEndpoint,
    });
  }

  const usedDirectAiModel = String(result.__backendEndpoint || '').includes('amr2004-ecosense-ai.hf.space');
  if (usedDirectAiModel) {
    // Do not call /images/upload or /sensors/analyze after a direct model diagnosis.
    // Those endpoints are real backend analysis/save flows, not generic save-only endpoints,
    // and retrying them here can duplicate analysis requests and trigger 429 rate limits.
    return {
      ok: false,
      modelOnly: true,
      error: {
        status: 'MODEL_ONLY',
        message: 'Direct model results are displayed immediately. Backend history is read from GET /sensors/history and GET /images/history only; saving direct model results needs a dedicated backend save endpoint.',
      },
    };
  }

  // Match the working site: Image -> POST /images/upload, Sensors -> POST /sensors/analyze/:sectorId,
  // History -> GET /images/history + GET /sensors/history.
  try {
    const documentedSave = await persistViaDocumentedHistoryEndpoints(result, form, mode, imageUrl);
    return acceptDocumentedSave({
      explicitSave: true,
      endpoint: documentedSave.endpoint,
      historyEndpoint: documentedSave.historyEndpoint,
      response: documentedSave.response?.data,
    });
  } catch (documentedErr) {
    void documentedErr;
    return { ok: false, error: getApiError(documentedErr) };
  }
};

const pickId = (item={}) => item._id || item.id || item.notificationId || item.deviceId || item.sectorId || Date.now();
const looksLikeBackendId = (value) => {
  const id = String(value || '').trim();
  if (!id) return false;
  if (/^\d{10,}$/.test(id)) return false; // local Date.now() IDs must never be sent to backend DELETE routes.
  if (/^[a-f0-9]{24}$/i.test(id)) return true;
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) return true;
  return id.length >= 12 && /^[A-Za-z0-9_-]+$/.test(id);
};
const backendEntityId = (item={}) => {
  const candidates = [item.backendId, item.imageLogId, item.image_log_id, item.imageLog?._id, item.imageLog?.id, item.imageId, item.image_id, item._id, item.reportId, item.taskId, item.report_id, item.task_id, item.uuid, item.mongoId, item.id];
  return candidates.find(looksLikeBackendId) || '';
};
const serviceFriendlyError = (err, lang='ar') => {
  const info = getApiError(err);
  const rawMessage = String(info.message || '');
  const fallback = lang==='ar'
    ? 'تعذر تنفيذ الطلب حاليًا. تأكد من الاتصال وحاول مرة أخرى.'
    : 'The request could not be completed right now. Check your connection and try again.';
  if (Number(info.status) === 401) {
    return lang === 'ar'
      ? 'انتهت الجلسة أو التوكن غير صالح. سجل الدخول مرة أخرى ثم حدّث بيانات الهاردوير.'
      : 'Your session expired or the token is invalid. Sign in again, then refresh hardware data.';
  }
  if (Number(info.status) === 403) {
    return lang === 'ar'
      ? 'القطاع المطلوب غير تابع لهذا الحساب أو لا تملك صلاحية الوصول له. تم إلغاء الاعتماد على القطاع الثابت؛ سجل الدخول بالحساب الصحيح أو افتح قطاعًا موجودًا من الباك إند.'
      : 'This sector does not belong to this account or you do not have access. The fixed sector filter was removed; sign in with the correct account or open a backend sector.';
  }
  if (/لا توجد قراءات مسجلة|no readings|no sensor readings|not found readings|reading.*not.*found/i.test(rawMessage)) {
    return lang === 'ar'
      ? 'الـ endpoint شغال لكن لا توجد قراءة محفوظة لهذا القطاع بعد. تأكد أن الجهاز مربوط بقطاع ويرسل deviceSerial صحيح.'
      : 'The endpoint is working, but no reading is saved for this sector yet. Check the device-sector link and deviceSerial.';
  }
  if (/deviceSerial|sectorId|معرف القطاع|تزويد السيرفر|server/i.test(rawMessage)) {
    return lang === 'ar'
      ? 'يمكن تشغيل التحليل بدون ربط بقطاع. اختر قطاعًا فقط إذا أردت حفظ التشخيص داخل سجل القطاع.'
      : 'Analysis can run without a sector. Choose a sector only if you want to save it to sector history.';
  }
  if (/Unsupported cropType/i.test(rawMessage)) {
    return lang === 'ar'
      ? 'الموديل الحالي يدعم فقط: نعناع، ذرة، فلفل، طماطم. اختر نوع النبات من القائمة.'
      : 'The model currently supports only Mint, Corn, Pepper, and Tomato. Choose the plant type from the list.';
  }
  if (/Cannot read properties|undefined is not|is not a function|TypeError/i.test(rawMessage)) {
    return lang === 'ar'
      ? 'حدث خطأ في عرض البيانات. حدّث الصفحة وحاول مرة أخرى.'
      : 'A display error occurred. Refresh the page and try again.';
  }
  if ([404,405,501].includes(Number(info.status)) || /route not found|endpoint|not found/i.test(rawMessage)) {
    return lang === 'ar'
      ? 'هذه الميزة تحتاج endpoint شغال من الباك إند. لن يتم تنفيذ زرار شكلي بدون ربط حقيقي.'
      : 'This feature needs a working backend endpoint. The action will not run as a fake button.';
  }
  if (Number(info.status) === 429) {
    return lang === 'ar'
      ? 'الخدمة مشغولة مؤقتًا بسبب كثرة الطلبات. انتظر لحظات ثم حاول مرة أخرى.'
      : 'The service is temporarily busy because of too many requests. Please wait and try again.';
  }
  if (!info.status) return fallback;
  return rawMessage || fallback;
};
const userFriendlyServiceError = (err, lang='ar') => serviceFriendlyError(err, lang);
const langFromDocument = () => document.documentElement.lang === 'ar' ? 'ar' : 'en';

function OwnerOnly({children}) {
  const { t } = useLang();
  return isOwner() ? children : <Unauthorized />;
}
function RoleGate({children, permission, any=[]}) {
  const allowed = permission ? hasPermission(permission) : any.some(p=>hasPermission(p));
  return allowed ? children : <Unauthorized />;
}

function App() {
  const [lang, setLang] = useState(localStorage.getItem('sph_lang') || 'ar');
  const [theme, setTheme] = useState(localStorage.getItem('sph_theme') || 'light');
  const base = dictionaries[lang] || dictionaries.en;
  const dict = { ...dictionaries.en, ...(extraDictionaries.en || {}), ...(productionTranslations.en || {}), ...(finalPolishTranslations.en || {}), ...(finalDeclutterTranslations.en || {}), ...(roleDashboardTranslations.en || {}), ...(finalProductOrganizationTranslations.en || {}), ...base, ...(extraDictionaries[lang] || {}), ...(productionTranslations[lang] || {}), ...(finalPolishTranslations[lang] || {}), ...(finalDeclutterTranslations[lang] || {}), ...(roleDashboardTranslations[lang] || {}), ...(finalProductOrganizationTranslations[lang] || {}), ...(lang === 'ar' ? arText : {}) };
  const value = useMemo(() => ({ lang, setLang, t: k => dict[k] ?? dictionaries.en[k] ?? extraDictionaries.en[k] ?? finalPolishTranslations.en[k] ?? humanizeKey(k), dict }), [lang, dict]);

  useEffect(() => { document.documentElement.lang = lang; document.documentElement.dir = dict.dir; document.body.dir = dict.dir; document.body.classList.toggle('rtl', dict.dir === 'rtl'); localStorage.setItem('sph_lang', lang); }, [lang, dict.dir]);
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('sph_theme', theme); }, [theme]);

  useEffect(() => {
    const oauthError = extractOAuthErrorFromUrl();
    const oauthToken = extractOAuthTokenFromUrl();

    if (oauthError && !oauthToken) {
      window.history.replaceState({}, document.title, '/login');
      localStorage.removeItem('ecosense_token');
      localStorage.removeItem('sph_token');
      localStorage.removeItem('sph_auth');
      sessionStorage.setItem('ecosense_google_error', oauthError);
      window.location.replace('/login');
      return undefined;
    }

    if (!oauthToken) return undefined;

    let mounted = true;
    const cleanToken = normalizeIncomingAuthToken(oauthToken);
    window.history.replaceState({}, document.title, '/auth/callback');
    localStorage.setItem('ecosense_token', cleanToken);
    localStorage.setItem('sph_token', cleanToken);
    localStorage.setItem('sph_auth', 'true');
    localStorage.setItem('sph_onboarded', 'true');

    authAPI.getMe()
      .then((res) => {
        if (!mounted) return;
        const user = res?.data?.user || res?.data?.data?.user || res?.data?.data || res?.data || { role: 'owner' };
        setLoggedInUser(user, cleanToken);
        window.location.replace(routeForRole(user));
      })
      .catch((err) => {
        if (!mounted) return;
        const status = err?.response?.status;
        // If auth/me rejects the returned token, do not fake a session. Show a clear login error.
        if (status === 401 || status === 403) {
          clearAccountDataCache({includeAllAccounts:true}); ['sph_auth','ecosense_token','sph_token','sph_role','sph_user_email','ecosense_user','token','user_token','accessToken'].forEach(k=>localStorage.removeItem(k));
          sessionStorage.setItem('ecosense_google_error', lang === 'ar' ? 'تسجيل Google رجع Token غير صالح. جرّب مرة أخرى.' : 'Google returned an invalid session token. Please try again.');
          window.location.replace('/login');
          return;
        }
        const fallbackUser = { name: 'Google User', role: 'owner', authProvider: 'google' };
        setLoggedInUser(fallbackUser, cleanToken);
        window.location.replace('/dashboard');
      });

    return () => { mounted = false; };
  }, [lang]);

  useEffect(() => {
    const token = localStorage.getItem('ecosense_token');
    const user = currentUser();
    const userKey = user?._id || user?.id || user?.email;
    if (!token || ['local-demo-token','local-worker-token'].includes(token) || !userKey) return undefined;
    try {
      const socketInstance = connectSocket(userKey);
      if (!socketInstance) return undefined;
      return onNotification((notification) => {
        const message = notification?.message || notification?.title || (lang === 'ar' ? 'إشعار جديد' : 'New notification');
        const type = notification?.type === 'danger' || notification?.type === 'critical' ? 'danger' : notification?.type || 'warning';
        addNotification(message, type);
      });
    } catch {
      return undefined;
    }
  }, [lang]);

  return <LangContext.Provider value={value}>
    <Routes>
      <Route path="/" element={<Landing theme={theme} setTheme={setTheme} />} />
      <Route path="/login" element={<Auth mode="login" theme={theme} setTheme={setTheme} />} />
      <Route path="/register" element={<Auth mode="register" theme={theme} setTheme={setTheme} />} />
      <Route path="/forgot-password" element={<ResetAuthFlow mode="forgot" theme={theme} setTheme={setTheme} />} />
      <Route path="/reset-password/verify" element={<Navigate to="/reset-password" replace />} />
      <Route path="/reset-password" element={<ResetAuthFlow mode="reset" theme={theme} setTheme={setTheme} />} />
      <Route path="/auth/callback" element={<AuthCallback />} />
      <Route element={<ProtectedRoute><AppShell theme={theme} setTheme={setTheme} /></ProtectedRoute>}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/diagnosis" element={<RoleGate permission="diagnose"><Diagnosis /></RoleGate>} />
        <Route path="/my-diagnoses" element={<RoleGate any={["reports","diagnose"]}><MyDiagnoses /></RoleGate>} />
        <Route path="/sensors" element={<RoleGate permission="sensors"><Sensors /></RoleGate>} />
        <Route path="/alerts" element={<RoleGate permission="alerts"><Alerts /></RoleGate>} />
        <Route path="/reports" element={<RoleGate permission="reports"><Reports /></RoleGate>} />
        <Route path="/library" element={<Navigate to="/diagnosis" replace />} />
        <Route path="/farm-management" element={<RoleGate permission="sectors"><FarmManagement /></RoleGate>} />
        <Route path="/sectors" element={<RoleGate permission="sectors"><Sectors /></RoleGate>} />
        <Route path="/sectors/:id" element={<SectorDetails />} />
        <Route path="/workers" element={<RoleGate any={["workers","workersView"]}><WorkerAccountsPage /></RoleGate>} />
        <Route path="/devices" element={<RoleGate permission="devices"><DevicesPage /></RoleGate>} />
        <Route path="/plants" element={<OwnerOnly><PlantProfiles /></OwnerOnly>} />
        <Route path="/tasks" element={<RoleGate permission="tasks"><MyTasks /></RoleGate>} />
        <Route path="/admin" element={<OwnerOnly><AdminPanel /></OwnerOnly>} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/plants/:id" element={<OwnerOnly><PlantProfileDetails /></OwnerOnly>} />
        <Route path="/chat" element={<AgricultureChat />} />
        <Route path="/more" element={<MoreMenu />} />
        <Route path="/settings" element={<SettingsPage theme={theme} setTheme={setTheme} />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </LangContext.Provider>;
}

function AuthCallback() {
  const { lang } = useLang();
  return <main className="auth-page compact-auth-page"><section className="auth-card compact-auth-card auth-real-card" autoComplete="off" data-form-type="other"><LogoBlock/><h2>{lang === 'ar' ? 'جاري تسجيل الدخول باستخدام Google...' : 'Signing in with Google...'}</h2><p className="muted-copy">{lang === 'ar' ? 'يتم حفظ الجلسة والتحقق من بيانات الحساب.' : 'Saving your session and checking your account.'}</p></section></main>;
}

function ProtectedRoute({children}) {
  const [checking,setChecking]=useState(false);
  const token = localStorage.getItem('ecosense_token');
  const authed = localStorage.getItem('sph_auth') === 'true' && !!token && !['local-demo-token','local-worker-token'].includes(token);

  useEffect(()=>{
    if(!authed) return undefined;
    let mounted=true;
    const needsRefresh = !currentUser()?.id && !currentUser()?._id;
    if(!needsRefresh) return undefined;
    setChecking(true);
    authAPI.getMe().then((res)=>{
      if(!mounted) return;
      const user=res?.data?.user || res?.data?.data?.user || res?.data?.data || res?.data;
      if(user && (user.email || user._id || user.id)) setLoggedInUser(user, token);
    }).catch(()=>{
      // Do not logout on temporary network/connection failure; interceptor handles real 401.
    }).finally(()=>mounted && setChecking(false));
    return()=>{mounted=false};
  },[authed, token]);

  if (!authed) return <Navigate to="/login" replace />;
  if (checking) return <div className="page"><Panel title="Loading"><p className="muted-copy">Checking current user...</p></Panel></div>;
  return children;
}

function LogoBlock() { const { t } = useLang(); return <div className="logo-block"><img src="/project-logo.png" alt={t('appName') || 'Ecosense AI'} onError={e => e.currentTarget.style.display='none'} /><div><b>{t('appName') || 'Ecosense AI'}</b><span>{t('brand')}</span></div></div>; }
function LanguageTheme({ theme, setTheme, showNotifications=false }) {
  const { lang, setLang, t } = useLang();
  return <div className="tools header-tools">
    <select value={lang} onChange={e=>setLang(e.target.value)} aria-label={t('language')}>
      {Object.entries(dictionaries).map(([k,v])=><option key={k} value={k}>{v.name}</option>)}
    </select>
    <button className="icon-btn theme-toggle-btn" onClick={()=>setTheme(theme==='dark'?'light':'dark')} aria-label={t('theme')}>
      {theme==='dark'?<Sun/>:<Moon/>}
    </button>
    {showNotifications && <NotificationBell compact />}
  </div>;
}

function Landing({ theme, setTheme }) {
  const { t, lang } = useLang();
  const openLabel = t('openPlatform');
  return <main className="landing clean-landing">
    <nav className="landing-nav clean-landing-nav">
      <LogoBlock/>
      <div className="landing-actions clean-landing-actions">
        <LanguageTheme theme={theme} setTheme={setTheme}/>
        <NavLink className="ghost-btn clean-login-btn" to="/login"><LogIn size={18}/>{openLabel}</NavLink>
      </div>
    </nav>
    <section className="hero clean-hero">
      <div className="hero-copy clean-hero-copy">
        <span className="pill"><Sparkles size={16}/>{t('heroBadge')}</span>
        <h1>{t('heroTitle')}</h1>
        <p>{t('heroDesc')}</p>
        <div className="hero-actions clean-hero-actions">
          <NavLink to="/login" className="primary-btn"><ScanSearch size={19}/>{openLabel}</NavLink>
        </div>
      </div>
      <div className="hero-panel">
        <div className="plant-visual-pro">
          <div className="orbit o1"><ThermometerSun size={22}/><span>{t('temperature')} 25°C</span></div>
          <div className="orbit o2"><Activity size={22}/><span>{t('soilMoisture')} 45%</span></div>
          <div className="orbit o3"><Bell size={22}/><span>{t('safe')}</span></div>
          <div className="scan-card pro">
            <div className="scan-area"><Leaf size={132}/><i/></div>
            <div className="preview-grid">
              <Metric label={t('condition')} value={t('healthy')} tone="green"/>
              <Metric label={t('confidence')} value="91%"/>
              <Metric label={t('diseaseProblem')} value={t('noDiseaseDetected')}/>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section className="features clean-features">
      <Feature icon={<ScanSearch/>} title={t('core1')} text={t('core1Text')}/>
      <Feature icon={<Cpu/>} title={t('core2')} text={t('core2Text')}/>
      <Feature icon={<Activity/>} title={t('core3')} text={t('core3Text')}/>
    </section>
    <footer>{t('footer')}</footer>
  </main>;
}
function Feature({ icon, title, text }) { return <article className="feature-card"><div className="feature-icon">{icon}</div><h3>{title}</h3><p>{text}</p></article>; }
function Metric({ label, value, tone='' }) { return <div className={`metric ${tone}`}><span>{label}</span><b>{value}</b></div>; }

function NotificationBell({compact=false}={}){
  const { t } = useLang();
  const [count,setCount]=useState(notificationStore().filter(n=>!n.read).length);
  const nav=useNavigate();
  useEffect(()=>{
    let mounted = true;
    const sync=()=>setCount(notificationStore().filter(n=>!n.read).length);
    const loadServiceNotifications = async()=>{
      const token = localStorage.getItem('ecosense_token');
      if(!token) return;
      try{
        const { data } = await notificationsAPI.getAll();
        const list = Array.isArray(data) ? data : (data.notifications || data.data || []);
        if(Array.isArray(list) && list.length){
          const mapped = list.map(n=>({
            id: n._id || n.id || Date.now(),
            time: n.createdAt || n.time || new Date().toLocaleString(),
            message: n.message || n.title || n.body || 'Notification',
            type: n.type || 'warning',
            read: !!(n.read || n.isRead)
          }));
          setNotificationStore(mapped);
          if(mounted) setCount(mapped.filter(n=>!n.read).length);
        }
      }catch(err){
        /* service notifications unavailable: local notifications stay active */
      }
    };
    sync(); loadServiceNotifications();
    window.addEventListener('sph-notification', sync);
    window.addEventListener('storage', sync);
    const timer = setInterval(loadServiceNotifications, 60000);
    return ()=>{mounted=false; clearInterval(timer); window.removeEventListener('sph-notification', sync); window.removeEventListener('storage', sync);};
  },[]);
  return <button className={`top-notification-btn ${compact?'compact':''}`} onClick={()=>nav('/alerts')} title={t('notifications')} aria-label={t('notifications')}>
    <Bell size={20}/>
  </button>;
}



function AccountRoleBadge(){
  const { t, lang } = useLang();
  const role = currentRole();
  const title = role === 'owner' ? t('owner') : role === 'farm_manager' ? t('farmManager') : t('worker');
  return <div className={`role-chip-pro ${role}`} title={title}>
    <span>{role === 'owner' ? <ShieldCheck size={15}/> : role === 'farm_manager' ? <Layers3 size={15}/> : <UserPlus size={15}/>}</span>
    <b>{title}</b>
  </div>;
}

function AppShell({ theme, setTheme }) {
  const [open,setOpen]=useState(false);
  const { t } = useLang();
  const loc=useLocation();
  useEffect(()=>{
    document.body.classList.toggle('sidebar-open', open);
    document.documentElement.setAttribute('data-sidebar-open', open ? 'true' : 'false');
    return ()=>{ document.body.classList.remove('sidebar-open'); document.documentElement.removeAttribute('data-sidebar-open'); };
  },[open]);
  const page = [
    ['/diagnosis','diagnosisCenter'], ['/my-diagnoses','myDiagnosis'], ['/dashboard','dashboard'], ['/plants','myDiagnosis'], ['/alerts','alerts'], ['/farm-management','farmManagementTitle'], ['/workers','workers'], ['/sectors','farmManagementTitle'], ['/admin','adminPanel'], ['/more','more'], ['/settings','settings']
  ].find(([p])=>loc.pathname===p || loc.pathname.startsWith(p + '/'))?.[1] || 'diagnosisCenter';
  return <div className="app-shell">
    <Sidebar open={open} close={()=>setOpen(false)}/>
    <div className="overlay" data-open={open} onClick={()=>setOpen(false)}/>
    <section className="content">
      <header className="topbar polished-topbar">
        <button className="icon-btn menu-btn" onClick={()=>setOpen(true)}><Menu/></button>
        <div className="topbar-title brand-page-title header-logo-title"><img src="/project-logo.png" alt={t('appName') || 'Ecosense AI'} onError={e=>e.currentTarget.style.display='none'}/><div><h2>{t(page)}</h2><p>{t('roleText')}</p></div></div>
        <div className="topbar-actions pro-header-actions"><LanguageTheme theme={theme} setTheme={setTheme} showNotifications={true}/></div>
      </header>
      <div className="page"><Outlet /></div>
    </section>
    <FloatingAssistant/><NotificationToast/>
  </div>
}

function Sidebar({ open, close }) {
  const { t } = useLang();
  const [diagnosisCount,setDiagnosisCount]=useState(0);
  useEffect(()=>{
    let mounted=true;
    const loadCount=()=>loadCombinedDiagnosisHistory({limit:1}).then(({count})=>{ if(mounted) setDiagnosisCount(count); }).catch(()=>{ if(mounted) setDiagnosisCount(0); });
    loadCount();
    window.addEventListener('ecosense-diagnoses-updated', loadCount);
    return ()=>{ mounted=false; window.removeEventListener('ecosense-diagnoses-updated', loadCount); };
  },[]);
  const ownerItems = [
    ['/dashboard', Home, 'dashboard'], ['/diagnosis', ScanSearch, 'diagnosisCenter'], ['/my-diagnoses', FileText, 'myDiagnosis'],
    ['/farm-management', Layers3, 'farmManagementTitle'], ['/workers', UserPlus, 'workers'], ['/tasks', CheckCircle2, 'myTasks'],
    ['/alerts', Bell, 'alerts'], ['/reports', FileText, 'reports'], ['/devices', Cpu, 'devices'], ['/sensors', Activity, 'connectedSensors'],
    ['/admin', ShieldCheck, 'adminPanel'], ['/settings', ShieldCheck, 'settings']
  ];
  const managerItems = [
    ['/dashboard', Home, 'dashboard'], ['/diagnosis', ScanSearch, 'diagnosisCenter'], ['/my-diagnoses', FileText, 'myDiagnosis'],
    ['/farm-management', Layers3, 'farmManagementTitle'], ['/tasks', CheckCircle2, 'myTasks'], ['/alerts', Bell, 'alerts'],
    ['/reports', FileText, 'reports'], ['/devices', Cpu, 'devices'], ['/sensors', Activity, 'connectedSensors'], ['/workers', UserPlus, 'workers'], ['/settings', ShieldCheck, 'settings']
  ];
  const workerItems = [
    ['/dashboard', Home, 'dashboard'], ['/tasks', CheckCircle2, 'myTasks'], ['/farm-management', Layers3, 'assignedSector'],
    ['/alerts', Bell, 'alerts'], ['/my-diagnoses', FileText, 'myDiagnosis'], ['/settings', ShieldCheck, 'settings']
  ];
  const items = isOwner() ? ownerItems : isFarmManager() ? managerItems : workerItems;
  return <aside className={`sidebar ${open?'open':''}`} data-open={open}>
    <button className="side-close" onClick={close}><X/></button>
    <LogoBlock/>
    <nav>{items.map(([to,Icon,key])=><NavLink key={to} to={to} onClick={close} className={({isActive})=>`side-link ${isActive?'active':''}`}><Icon size={20}/><span>{t(key)}{key==='myDiagnosis' && diagnosisCount ? ` (${diagnosisCount})` : ''}</span></NavLink>)}</nav>
    <div className={`side-account-card ${currentRole()}`}>
      <div className="side-account-avatar"><span>{displayUserName().slice(0,1).toUpperCase()}</span></div>
      <div className="side-account-meta">
        <b>{displayUserName()}</b>
        <span>{roleLabel(t)}</span>
        <small><i />{t('online')}</small>
      </div>
      <button className="side-signout" onClick={signOutUser} title={t('signOut')}><LogOut size={16}/></button>
    </div>
  </aside>;
}


function MobileBottomNav(){
  const { t } = useLang();
  const ownerItems = [
    ['/dashboard', Home, 'dashboard'],
    ['/diagnosis', ScanSearch, 'diagnosis'],
    ['/farm-management', Layers3, 'farm'],
    ['/workers', UserPlus, 'workers'],
    ['/settings', ShieldCheck, 'settings']
  ];
  const managerItems = [
    ['/dashboard', Home, 'dashboard'],
    ['/diagnosis', ScanSearch, 'diagnosis'],
    ['/farm-management', Layers3, 'farm'],
    ['/tasks', CheckCircle2, 'myTasks'],
    ['/alerts', Bell, 'alerts']
  ];
  const workerItems = [
    ['/dashboard', Home, 'dashboard'],
    ['/tasks', CheckCircle2, 'myTasks'],
    ['/farm-management', Layers3, 'assignedSector'],
    ['/alerts', Bell, 'alerts'],
    ['/settings', ShieldCheck, 'settings']
  ];
  const baseItems = isOwner() ? ownerItems : isFarmManager() ? managerItems : workerItems;
  return <nav className="mobile-bottom-nav compact-five" aria-label="Mobile navigation">
    {baseItems.map(([to,Icon,key])=><NavLink key={to} to={to} className={({isActive})=>isActive?'active':''}><Icon size={19}/><span>{t(key)}</span></NavLink>)}
  </nav>;
}
function NotificationToast(){
  const [note,setNote]=useState(null);
  useEffect(()=>{
    const handler=(e)=>{ setNote(e.detail); setTimeout(()=>setNote(null), 6500); };
    window.addEventListener('sph-notification', handler);
    return ()=>window.removeEventListener('sph-notification', handler);
  },[]);
  if(!note) return null;
  return <div className={`alarm-toast ${note.type||'warning'}`}>
    <AlertTriangle size={26}/>
    <div><b>{note.type==='danger'?'ALARM':'Alert'}</b><span>{note.message}</span></div>
    <button onClick={()=>setNote(null)}><X size={16}/></button>
  </div>
}

function Dashboard(){ 
  const { t, lang }=useLang(); 
  if(isWorker()) return <WorkerDashboard/>; 
  const [serviceDashboard,setServiceDashboard]=useState(null);
  const [hardwareFeed,setHardwareFeed]=useState({latest:null,history:[],imageHistory:[],error:''});
  useEffect(()=>{
    let mounted=true;
    dashboardAPI.get().then(({data})=>{
      if(!mounted) return;
      const payload=data?.data || data?.dashboard || data || null;
      setServiceDashboard(payload);
    }).catch(()=>setServiceDashboard(null));
    return()=>{mounted=false};
  },[]);
  useEffect(()=>{
    let mounted=true;
    Promise.allSettled([
      sensorsAPI.getLatestHardware(),
      sensorsAPI.getHardwareHistory({limit:8}),
      imagesAPI.getHardwareHistory({limit:4}),
    ]).then(([latestRes,historyRes,imageRes])=>{
      if(!mounted) return;
      const latest = latestRes.status==='fulfilled' ? normalizeServiceSensorPayload(latestRes.value.data || {}).reading : null;
      const history = historyRes.status==='fulfilled' ? diagnosisRecordsFromPayload(historyRes.value.data).map((item,index)=>normalizeServiceSensorPayload(item).reading).filter(hasUsefulSensorReading) : [];
      const imageHistory = imageRes.status==='fulfilled' ? imageHistoryRecordsFromPayload(imageRes.value.data).map((item,index)=>normalizeImageHistoryRecord({...item,historySource:'images'},index)) : [];
      const failed = latestRes.status==='rejected' && historyRes.status==='rejected';
      setHardwareFeed({latest,history,imageHistory,error: failed ? serviceFriendlyError(latestRes.reason, lang) : ''});
    }).catch(err=>mounted && setHardwareFeed({latest:null,history:[],imageHistory:[],error:serviceFriendlyError(err,lang)}));
    return()=>{mounted=false};
  },[lang]);
  const localReadings=getStore('sph_readings',[]); 
  const remoteLatestRaw=serviceDashboard?.latestReading || serviceDashboard?.latest_reading || serviceDashboard?.latest || serviceDashboard?.sensorSnapshot || serviceDashboard?.sensors;
  const remoteLatest=remoteLatestRaw ? normalizeServiceSensorPayload(remoteLatestRaw).reading : hardwareFeed.latest;
  const hardwareHistory = hardwareFeed.history || [];
  const readings=remoteLatest ? [remoteLatest,...hardwareHistory.filter(r=>String(r.id)!==String(remoteLatest.id)),...localReadings.filter(r=>String(r.id)!==String(remoteLatest.id))] : (hardwareHistory.length ? hardwareHistory : localReadings);
  const latest=readings[0] || latestAutoReading();
  const alerts=Number(serviceDashboard?.alertsCount ?? serviceDashboard?.notificationsCount ?? notificationStore().length);
  const risky=readings.filter(r=>r.final_status!=='Healthy').length;
  const latestDiagnosis = latest.diagnosis || (latest.final_status==='Healthy' ? 'Stable plant condition with balanced readings.' : 'Water or heat stress indicators detected.');
  const dashSectors = getStore('sph_sectors', []);
  const dashDevices = dashSectors.flatMap(s => (s.devices || []));
  const onlineDevices = dashDevices.filter(d => String(d.status || 'Online').toLowerCase() !== 'offline').length;
  const healthyCount = dashSectors.filter(s => String(s.status || '').toLowerCase().includes('healthy')).length;
  const moderateCount = dashSectors.filter(s => String(s.status || '').toLowerCase().includes('moderate')).length;
  const highCount = dashSectors.filter(s => isDanger(s.status)).length;
  const totalHealth = Math.max(1, healthyCount + moderateCount + highCount);
  const commandLabels = lang === 'ar'
    ? { title:'مركز قيادة المزرعة', sub:'ملخص سريع لحالة المزرعة والقطاعات والتنبيهات من أول نظرة.', farmStatus:'حالة المزرعة العامة', sectors:'القطاعات', onlineDevices:'الأجهزة المتصلة', latestSensor:'آخر قراءة حساسات', latestDiagnosis:'آخر تشخيص', criticalAlerts:'تنبيهات خطيرة', healthSummary:'توزيع حالة القطاعات', healthy:'سليم', moderate:'متوسط', high:'خطر' }
    : { title:'Farm Command Center', sub:'A quick operating view for farm health, sectors, alerts, and latest diagnosis.', farmStatus:'Overall Farm Status', sectors:'Sectors', onlineDevices:'Connected Devices', latestSensor:'Latest Sensor Reading', latestDiagnosis:'Latest Diagnosis', criticalAlerts:'Critical Alerts', healthSummary:'Sector Health Summary', healthy:'Healthy', moderate:'Moderate', high:'High Stress' };
  return <>
    <PageHead title={currentRole()==='owner'?t('ownerDashboard'):currentRole()==='farm_manager'?t('farmManagerDashboard'):t('workerDashboard')} sub={t('mainProductFocus')} action={<NavLink to="/diagnosis" className="primary-btn"><ScanSearch size={18}/>{t('startDiagnosis')}</NavLink>}/>
    {hardwareFeed.error && <p className="form-helper warning-text">{hardwareFeed.error}</p>}
    <section className="farm-greeting-banner">
      <div className="farm-greeting-content">
        <span className="farm-greeting-eyebrow"><Leaf size={16}/>{t('appName')}</span>
        <h2>{t(greetingKeyByTime())}, {displayUserName()} 👋</h2>
        <p>{t('overviewSubtitle')}</p>
      </div>
      <div className="farm-greeting-status">
        <span className={`role-badge role-${currentRole()}`}>{roleLabel(t)}</span>
        <small><i />{t('online')}</small>
      </div>
    </section>
    <section className={`dashboard-command-center ${tone(latest.final_status)}`}>
      <div className="command-main">
        <span className="command-eyebrow"><Leaf size={16}/>{commandLabels.title}</span>
        <h2>{commandLabels.farmStatus}: {labelStatus(latest.final_status,t)}</h2>
        <p>{commandLabels.sub}</p>
      </div>
      <div className="command-side">
        <span className={`tag ${tone(latest.final_status)}`}>{labelStatus(latest.final_status,t)}</span>
        <b>{Math.round((latest.confidence ?? .9)*100)}%</b>
      </div>
    </section>
    <div className="dashboard-command-kpis">
      <Kpi icon={<Layers3/>} label={commandLabels.sectors} value={dashSectors.length}/>
      <Kpi icon={<Cpu/>} label={commandLabels.onlineDevices} value={`${onlineDevices}/${dashDevices.length || 0}`}/>
      <Kpi icon={<Activity/>} label={commandLabels.latestSensor} value={latest.soilMoisture !== undefined && latest.soilMoisture !== null ? `${latest.soilMoisture}%` : '—'}/>
      <Kpi icon={<ScanSearch/>} label={commandLabels.latestDiagnosis} value={labelStatus(latest.final_status,t)}/>
      <Kpi icon={<AlertTriangle/>} label={commandLabels.criticalAlerts} value={alerts + highCount} tone={alerts + highCount ? 'red' : 'green'}/>
    </div>
    <section className="health-summary-card">
      <div className="health-summary-head"><b>{commandLabels.healthSummary}</b><span>{dashSectors.length} {commandLabels.sectors}</span></div>
      <div className="health-summary-bar" aria-label={commandLabels.healthSummary}>
        <i className="healthy" style={{width:`${(healthyCount/totalHealth)*100}%`}}/>
        <i className="moderate" style={{width:`${(moderateCount/totalHealth)*100}%`}}/>
        <i className="high" style={{width:`${(highCount/totalHealth)*100}%`}}/>
      </div>
      <div className="health-summary-legend"><span><i className="healthy"/>{commandLabels.healthy}: {healthyCount}</span><span><i className="moderate"/>{commandLabels.moderate}: {moderateCount}</span><span><i className="high"/>{commandLabels.high}: {highCount}</span></div>
    </section>
    <section className={`dashboard-focus-card ${tone(latest.final_status)}`}>
      <div className="dashboard-focus-icon"><Leaf size={30}/></div>
      <div className="dashboard-focus-copy">
        <span>{t('finalStatus')}</span>
        <h2>{labelStatus(latest.final_status,t)}</h2>
        <p>{translateModelText(latestDiagnosis,t)}</p>
      </div>
      <div className="dashboard-focus-actions">
        <NavLink className="primary-btn" to="/diagnosis"><ScanSearch size={17}/>{t('startDiagnosis')}</NavLink>
        <NavLink className="secondary-btn" to="/farm-management"><FileText size={17}/>{t('farmReports')}</NavLink>
      </div>
    </section>
    <div className="kpis dashboard-quiet-kpis">
      <Kpi icon={<ScanSearch/>} label={t('myDiagnosis')} value={readings.length}/>
      <Kpi icon={<Activity/>} label={t('soilMoisture')} value={latest.soilMoisture !== undefined && latest.soilMoisture !== null ? `${latest.soilMoisture}%` : '—'}/>
      <Kpi icon={<Bell/>} label={t('alerts')} value={alerts + risky}/>
    </div>
    <details className="calm-details dashboard-extra-details">
      <summary><Eye size={18}/>{t('viewDetails')}</summary>
      <div className="grid-2 soft-hidden-grid">
        <Panel title={t('imageAndSensorResults')}>
          <p className="muted-copy">{t('uploadOrConnect')}</p>
          <ReadingGrid r={latest}/>
        </Panel>
        <Panel title={t('statistics')}>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={readings.slice().reverse()}>
              <CartesianGrid vertical={false} stroke="var(--line)"/><XAxis dataKey="time"/><YAxis/><Tooltip/>
              <Area dataKey="soilMoisture" stroke="var(--primary)" fill="var(--primarySoft)" strokeWidth={3}/>
              <Area dataKey="temperature" stroke="var(--amber)" fill="rgba(245,158,11,.13)" strokeWidth={3}/>
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
      </div>
      <div className="dashboard-secondary-links">
        <NavLink className="secondary-btn" to="/farm-management"><Layers3 size={17}/>{t('farmManagementTitle')}</NavLink>
        <button className="secondary-btn" type="button" onClick={()=>window.dispatchEvent(new Event('ecosense-open-assistant'))}><Leaf size={17}/>{t('askAssistant')}</button>
      </div>
    </details>
  </> 
}
function PageHead({title,sub,action}){ return <div className="page-head"><div><h1>{title}</h1><p>{sub}</p></div>{action}</div> }
function Kpi({icon,label,value,tone:tn=''}){ return <div className={`kpi ${tn}`}>{icon}<span>{label}</span><b>{value}</b></div> }
function Panel({title,children}){ return <section className="panel"><h3>{title}</h3>{children}</section> }
function ReadingGrid({r={}}){
  const {t}=useLang();
  const cropLabel=localCropLabel(r.cropType, langFromDocument());
  const sensorValue=(value,suffix='') => (value === undefined || value === null || value === '' ? '-' : `${value}${suffix}`);
  return <div className="reading-grid">
    {cropLabel && <Box label={t('cropType')} value={cropLabel}/>}<Box label={t('temperature')} value={sensorValue(r.temperature,'°C')}/><Box label={t('humidity')} value={sensorValue(r.humidity,'%')}/><Box label={t('soilMoisture')} value={sensorValue(r.soilMoisture,'%')}/><Box label={t('soilTemp')} value={sensorValue(r.soilTemp,'°C')}/><Box label={t('light')} value={sensorValue(r.light)}/>
  </div>
}
function Box({label,value}){ const {t}=useLang(); return <div className="box"><span>{label}</span><b>{localizeValue(value,t)}</b></div> }
function EmptyActionState({icon:Icon=Leaf,title,description,action}){
  return <section className="empty-action-state">
    <span className="empty-action-icon"><Icon size={28}/></span>
    <div><h3>{title}</h3>{description && <p>{description}</p>}</div>
    {action && <div className="empty-action-cta">{action}</div>}
  </section>;
}
function SkeletonGrid({cards=3}){
  return <div className="skeleton-grid">{Array.from({length:cards}).map((_,i)=><article className="skeleton-card" key={i}><span/><b/><em/><i/></article>)}</div>;
}


const DIAGNOSIS_HISTORY_MIN_INTERVAL = 12000;

function useDiagnoses(limit, params = {}){
  const [state,setState]=useState({loading:true,error:'',rows:[],count:0,live:false});
  const loadingRef = useRef(false);
  const lastFetchRef = useRef(0);
  const mountedRef = useRef(true);
  const limitRef = useRef(limit);
  const paramsRef = useRef(params || {});

  const paramsKey = JSON.stringify(params || {});

  useEffect(()=>{
    mountedRef.current = true;
    limitRef.current = limit;
    paramsRef.current = params || {};
    return()=>{ mountedRef.current = false; };
  },[limit, paramsKey]);

  const load=async(options={})=>{
    const force = !!options.force;
    const now = Date.now();

    // Prevent request storms that trigger backend 429 Too Many Requests.
    // Filters use the already-loaded state locally, so they should never refetch history.
    if (loadingRef.current) return;
    if (!force && now - lastFetchRef.current < DIAGNOSIS_HISTORY_MIN_INTERVAL) return;

    loadingRef.current = true;
    lastFetchRef.current = now;
    setState(s=>({...s,loading:!s.rows.length,error:''}));
    try{
      const queryParams = { ...(paramsRef.current || {}) };
      if (limitRef.current && !queryParams.limit) queryParams.limit = limitRef.current;
      const {rows,count,live,errors}=await loadCombinedDiagnosisHistory(queryParams);
      if(!mountedRef.current) return;
      const backendUnavailable = errors.length === 2;
      const finalRows = limitRef.current ? rows.slice(0,limitRef.current) : rows;
      const friendly = backendUnavailable
        ? (document.documentElement.lang==='ar'?'تعذر تحميل سجل التشخيصات من الباك إند حاليًا.':'Could not load diagnosis history from the backend right now.')
        : '';
      setState({loading:false,error:friendly,rows:finalRows,count:count || rows.length,live});
    }catch(err){
      if(!mountedRef.current) return;
      setState({loading:false,error:(document.documentElement.lang==='ar'?'تعذر تحميل سجل التشخيصات من الباك إند حاليًا.':'Could not load diagnosis history from the backend right now.'),rows:[],count:0,live:false});
    } finally {
      loadingRef.current = false;
    }
  };

  useEffect(()=>{
    load({force:true});
    const onUpdated=()=>load({force:true});
    window.addEventListener('ecosense-diagnoses-updated', onUpdated);
    return()=>window.removeEventListener('ecosense-diagnoses-updated', onUpdated);
  },[limit, paramsKey]);

  return {...state, reload:()=>load({force:true})};
}

function LatestDiagnosesPanel(){
  const {t,lang}=useLang();
  const {rows,loading,count}=useDiagnoses(5);
  const labels=lang==='ar'?{title:'آخر التشخيصات',sub:'آخر نتائج محفوظة من الباك إند.',all:'عرض كل التشخيصات'}:{title:'Latest Diagnoses',sub:'Recent saved results from the backend history.',all:'View all diagnoses'};
  return <section className="latest-diagnoses-panel panel">
    <div className="panel-head-row"><div><h3>{labels.title}</h3><p>{labels.sub}</p></div><NavLink className="secondary-btn" to="/my-diagnoses"><FileText size={16}/>{labels.all}{count?` (${count})`:''}</NavLink></div>
    {loading ? <SkeletonGrid cards={3}/> : rows.length ? <div className="latest-diagnosis-list">
      {rows.map(r=><article key={r.id} className={`latest-diagnosis-item ${tone(r.final_status)}`}>
        <span>{r.analysisType}</span><b>{labelStatus(r.final_status,t)}</b><small>{formatDateTime(r.date)}</small>
      </article>)}
    </div> : <EmptyActionState icon={ScanSearch} title={lang==='ar'?'لا توجد تشخيصات محفوظة':'No saved diagnoses'} description={lang==='ar'?'ابدأ أول تشخيص واربطه بقطاع ليظهر هنا من سجل الباك إند.':'Start your first diagnosis and link it to a sector so it appears here from backend history.'} action={<NavLink className="primary-btn" to="/diagnosis"><ScanSearch size={16}/>{lang==='ar'?'ابدأ تشخيص':'Start diagnosis'}</NavLink>}/>}
  </section>;
}

function MyDiagnoses(){
  const {t, lang}=useLang();
  const location=useLocation();
  const initialSectorFilter = new URLSearchParams(location.search).get('sectorId') || 'all';
  const [filters,setFilters]=useState({status:'all',type:'all',sector:initialSectorFilter,search:''});
  const [selected,setSelected]=useState(null);
  const historyParams = useMemo(() => {
    const params = { limit: 200 };
    if (filters.sector !== 'all') params.sectorId = filters.sector;
    return params;
  }, [filters.sector]);
  const {rows,loading,error,count,reload}=useDiagnoses(null, historyParams);
  const [deleteTarget,setDeleteTarget]=useState(null);
  const [deleting,setDeleting]=useState(false);
  const [sectors,setSectors]=useState(getStore('sph_sectors', []));
  useEffect(()=>{
    let mounted=true;
    sectorsAPI.getAll().then(({data})=>{
      const list=asArray(data,['sectors','items','rows','records']);
      if(!mounted || !list.length) return;
      const mapped=list.map((item,idx)=>({
        id:item._id || item.id || item.sectorId || idx+1,
        name:item.name || item.sectorName || item.sector_name || `Sector ${idx+1}`,
        location:item.location || item.area || '-',
        crop:item.crop || item.cropType || item.plantType || '',
        status:item.status || item.final_status || 'Healthy',
      }));
      setSectors(mapped); setStore('sph_sectors',mapped);
    }).catch(()=>{});
    return()=>{mounted=false};
  },[]);
  const labels=lang==='ar'?{
    title:'تشخيصاتي', sub:'كل التشخيصات المحفوظة من الصورة أو الحساسات في مكان واحد.',
    no:'لا توجد تشخيصات محفوظة حتى الآن.', start:'ابدأ أول تشخيص لحالة النبات وسيظهر هنا.',
    status:'الحالة', type:'نوع التشخيص', sector:'القطاع', search:'ابحث بالمرض أو التاريخ أو القطاع', all:'الكل', details:'عرض التفاصيل', report:'تحميل التقرير', saved:'تشخيص محفوظ', readings:'القراءات', recommendations:'التوصيات', actions:'الإجراءات', image:'الصورة', count:'عدد التشخيصات', refresh:'تحديث السجل', visible:'المعروض', newDiagnosis:'ابدأ تشخيص جديد', backend:'السجل المحفوظ من الباك إند', result:'النتيجة النهائية', disease:'المرض', summary:'ملخص التشخيص', scan:'تشخيص', clear:'مسح الفلاتر', total:'الإجمالي', delete:'حذف', cancel:'إلغاء', confirmDeleteTitle:'تأكيد الحذف', confirmDeleteImage:'هل تريد حذف تشخيص الصورة هذا من الباك إند؟', sensorDeleteMissing:'حذف تشخيصات الحساسات غير متاح من الباك إند حاليًا.', deleted:'تم حذف التشخيص.', deleteFailed:'تعذر حذف التشخيص من الباك إند.'
  }:{
    title:'My Diagnoses', sub:'All saved image and sensor diagnoses in one place.',
    no:'No saved diagnoses yet.', start:'Start your first plant diagnosis and it will appear here.',
    status:'Status', type:'Diagnosis Type', sector:'Sector', search:'Search disease, date, or sector', all:'All', details:'View Details', report:'Download Report', saved:'Saved diagnosis', readings:'Readings', recommendations:'Recommendations', actions:'Actions', image:'Image', count:'Diagnoses Count', refresh:'Refresh History', visible:'Visible', newDiagnosis:'Start New Diagnosis', backend:'Backend saved history', result:'Final Result', disease:'Disease', summary:'Diagnosis Summary', scan:'Scan', clear:'Clear filters', total:'Total', delete:'Delete', cancel:'Cancel', confirmDeleteTitle:'Confirm deletion', confirmDeleteImage:'Delete this image diagnosis from the backend?', sensorDeleteMissing:'Deleting sensor diagnoses is not available in the backend yet.', deleted:'Diagnosis deleted.', deleteFailed:'Could not delete this diagnosis from the backend.'
  };
  const filtered=useMemo(()=>rows.filter(r=>{
    const st=String(r.final_status||'').toLowerCase();
    const tp=String(r.analysisType||'').toLowerCase();
    const sec=String(`${r.sectorId||''} ${r.sectorName||''}`).toLowerCase();
    const q=String(filters.search||'').toLowerCase().trim();
    const okStatus=filters.status==='all' || st.includes(filters.status.toLowerCase());
    const okType=filters.type==='all' || tp.includes(filters.type.toLowerCase());
    const okSector=filters.sector==='all' || sec.includes(String(filters.sector).toLowerCase());
    const okSearch=!q || [r.disease_name,r.date,r.sectorName,r.diagnosis,r.analysisType,r.final_status].some(v=>String(v||'').toLowerCase().includes(q));
    return okStatus && okType && okSector && okSearch;
  }),[rows,filters]);
  const stats=useMemo(()=>{
    const image=rows.filter(r=>String(r.analysisType||'').toLowerCase().includes('image')).length;
    const sensors=rows.filter(r=>String(r.analysisType||'').toLowerCase().includes('sensor')).length;
    const danger=rows.filter(r=>tone(r.final_status)==='red').length;
    return {image,sensors,danger,total:count || rows.length};
  },[rows,count]);
  const resetFilters=()=>{
    setFilters({status:'all',type:'all',sector:'all',search:''});
  };
  const handleDeleteRecord=async()=>{
    if(!deleteTarget) return;
    const id = backendEntityId(deleteTarget) || backendEntityId(deleteTarget.raw || {}) || (!String(deleteTarget.id || '').startsWith('diagnosis-') ? deleteTarget.id : '');
    const isImage = String(deleteTarget.historySource || deleteTarget.analysisType || '').toLowerCase().includes('image') || Boolean(deleteTarget.imageUrl);
    if(!isImage){ addNotification(labels.sensorDeleteMissing,'warning'); setDeleteTarget(null); return; }
    if(!id){ addNotification(labels.deleteFailed,'danger'); setDeleteTarget(null); return; }
    setDeleting(true);
    try{
      await imagesAPI.delete(id);
      diagnosisHistoryRuntimeCache.time = 0;
      diagnosisHistoryRuntimeCache.value = null;
      diagnosisHistoryRuntimeCache.promise = null;
      addNotification(labels.deleted,'success');
      setDeleteTarget(null);
      await reload();
    }catch(err){
      addNotification(userFriendlyServiceError(err, lang) || labels.deleteFailed,'danger');
    }finally{ setDeleting(false); }
  };
  return <main className="mydiag-page mydiag-v2">
    <section className="mydiag-v2-header">
      <div className="mydiag-v2-titleblock">
        <div className="mydiag-v2-icon"><FileText size={24}/></div>
        <div>
          <span className="mydiag-v2-eyebrow">{labels.backend}</span>
          <h1>{labels.title}</h1>
          <p>{labels.sub}</p>
        </div>
      </div>
      <div className="mydiag-v2-side">
        <div className="mydiag-v2-stat strong"><small>{labels.count}</small><b>{stats.total}</b></div>
        <div className="mydiag-v2-stat"><small>{labels.visible}</small><b>{filtered.length}</b></div>
        <NavLink to="/diagnosis" className="primary-btn mydiag-v2-new"><ScanSearch size={17}/>{labels.newDiagnosis}</NavLink>
        <button className="secondary-btn mydiag-v2-refresh" onClick={reload} type="button"><Activity size={16}/>{labels.refresh}</button>
      </div>
    </section>

    <section className="mydiag-v2-mini-stats" aria-label={labels.total}>
      <div><span>{labels.total}</span><b>{stats.total}</b></div>
      <div><span>Image</span><b>{stats.image}</b></div>
      <div><span>Sensors</span><b>{stats.sensors}</b></div>
      <div className={stats.danger?'danger':''}><span>High Stress</span><b>{stats.danger}</b></div>
    </section>

    <section className="mydiag-v2-filterbar">
      <label><span>{labels.status}</span><select value={filters.status} onChange={e=>setFilters({...filters,status:e.target.value})}><option value="all">{labels.all}</option><option value="Healthy">{t('healthy')}</option><option value="Moderate">{t('moderate')}</option><option value="High">{t('high')}</option></select></label>
      <label><span>{labels.type}</span><select value={filters.type} onChange={e=>setFilters({...filters,type:e.target.value})}><option value="all">{labels.all}</option><option value="Image">Image</option><option value="Sensors">Sensors</option><option value="Image + Sensors">Image + Sensors</option></select></label>
      <label><span>{labels.sector}</span><select value={filters.sector} onChange={e=>setFilters({...filters,sector:e.target.value})}><option value="all">{labels.all}</option>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}</option>)}</select></label>
      <label className="mydiag-v2-search"><span>{labels.search}</span><div><Search size={15}/><input value={filters.search} onChange={e=>setFilters({...filters,search:e.target.value})} placeholder={labels.search}/></div></label>
      <button type="button" className="secondary-btn mydiag-v2-clear" onClick={resetFilters}>{labels.clear}</button>
    </section>

    {error && <section className="mydiag-v2-alert"><AlertTriangle size={18}/><span>{error}</span></section>}

    {loading ? <SkeletonGrid cards={6}/> : filtered.length ? <section className="mydiag-v2-grid">
      {filtered.map((r,i)=><DiagnosisHistoryCard key={`${r.analysisType}-${r.id}-${i}`} record={{...r,number:i+1}} labels={labels} onDetails={()=>setSelected(r)} onDelete={()=>setDeleteTarget(r)} />)}
    </section> : <EmptyActionState icon={FileText} title={labels.no} description={labels.start} action={<NavLink className="primary-btn" to="/diagnosis"><ScanSearch size={17}/>{labels.newDiagnosis}</NavLink>}/>}
    {selected && <DiagnosisDetailsModal record={selected} onClose={()=>setSelected(null)} labels={labels}/>} 
    {deleteTarget && <ConfirmModal title={labels.confirmDeleteTitle} message={labels.confirmDeleteImage} cancelLabel={labels.cancel} confirmLabel={labels.delete} loading={deleting} onCancel={()=>!deleting && setDeleteTarget(null)} onConfirm={handleDeleteRecord}/>}
  </main>;
}

function DiagnosisHistoryCard({record,labels,onDetails,onDelete}){
  const {t}=useLang();
  const ar=isArabicUI(t);
  const statusTone=tone(record.final_status);
  const confidence=record.confidence ? Math.round((record.confidence||0)*100) : null;
  const rows=readingRows(record,t).filter(r=>!['cropType'].includes(r.key));
  const hasImage=!!record.imageUrl;
  const sourceText = hasImage && rows.length ? (ar?'صورة + حساسات':'Image + Sensors') : hasImage ? (ar?'صورة':'Image') : (ar?'حساسات':'Sensors');
  const cleanDiseaseName = !record.disease_name || String(record.disease_name).trim()==='-' ? (ar?'لا يوجد مرض واضح':'No clear disease detected') : record.disease_name;
  const disease=translateModelText(cleanDiseaseName,t);
  const shortDiagnosis=displayText(record.diagnosis || diagnosisHeadline(record,t),t);
  const quickRows=rows.slice(0,3);

  return <article className={`history-compact-card ${statusTone}`}>
    <div className="history-compact-media">
      {hasImage ? <img src={record.imageUrl} alt="Plant diagnosis"/> : <div className="history-compact-sensor"><Cpu size={28}/><span>{ar?'حساسات':'Sensors'}</span></div>}
    </div>
    <div className="history-compact-body">
      <header className="history-compact-head">
        <div className="history-compact-title">
          <span className="history-compact-number">#{record.number}</span>
          <div>
            <b>{sourceText}</b>
            <small>{formatDateTime(record.date)} • {localizeValue(record.sectorName,t)}</small>
          </div>
        </div>
        <span className={`history-compact-status ${statusTone}`}>{labelStatus(record.final_status,t)}</span>
      </header>
      <div className="history-compact-facts">
        <span><small>{ar?'الثقة':'Confidence'}</small><b>{confidence !== null ? `${confidence}%` : '—'}</b></span>
        <span><small>{ar?'المرض / المشكلة':'Disease / Issue'}</small><b>{disease}</b></span>
        {displayPlantType(record,record,t) && <span><small>{ar?'نوع النبات':'Plant type'}</small><b>{displayPlantType(record,record,t)}</b></span>}
      </div>
      <p className="history-compact-summary">{shortDiagnosis}</p>
      {quickRows.length ? <div className="history-compact-readings">
        {quickRows.map(r=><span key={r.key}><small>{r.label}</small><b>{r.value}</b></span>)}
      </div> : null}
      <footer className="history-compact-actions">
        <button className="secondary-btn" onClick={onDetails}><Eye size={15}/>{labels.details}</button>
        <button className="primary-btn" onClick={()=>downloadDiagnosisReport(diagnosisRecordToResult(record),t)}><FileText size={15}/>{labels.report}</button>
        {hasImage ? <button className="danger-soft-btn" onClick={onDelete}><Trash2 size={15}/>{labels.delete}</button> : <small className="sensor-delete-note">{labels.sensorDeleteMissing}</small>}
      </footer>
    </div>
  </article>;
}

function DiagnosisDetailsModal({record,onClose,labels}){
  const {t}=useLang();
  const result=diagnosisRecordToResult(record);
  const statusTone=tone(record.final_status);
  const confidence=record.confidence ? Math.round((record.confidence||0)*100) : null;
  const rows=readingRows(record,t);
  const metrics=imageMetricRows(record,t);
  const treatments=treatmentItems(record,t);
  const actions=actionItems(record,t);
  const tips=captureTips(record,t);
  const headline=diagnosisHeadline(record,t);
  const note=modelNote(record,t);
  const ar=isArabicUI(t);
  const hasImage=!!record.imageUrl;
  const sourceText = hasImage && rows.length ? (ar?'صورة + حساسات':'Image + Sensors') : hasImage ? (ar?'صورة':'Image') : (ar?'حساسات':'Sensors');
  const cleanDiseaseName = !record.disease_name || String(record.disease_name).trim()==='-' ? (ar?'لا يوجد مرض واضح':'No clear disease detected') : record.disease_name;
  const disease=translateModelText(cleanDiseaseName,t);

  return <div className="modal-backdrop diagnosis-details-backdrop diag-v2-backdrop" role="dialog" aria-modal="true">
    <section className="modal-card diagnosis-detail-modal-pro">
      <button className="modal-close" onClick={onClose}><X size={18}/></button>

      <header className={`detail-modal-hero ${statusTone}`}>
        <div className="detail-modal-title">
          <span>{ar?'تقرير تشخيص محفوظ':'Saved Diagnosis Report'}</span>
          <h2>{labelStatus(record.final_status,t)}</h2>
          <p>{sourceText} • {formatDateTime(record.date)} • {localizeValue(record.sectorName,t)}</p>
        </div>
        <div className="detail-confidence-box"><b>{confidence}%</b><small>{ar?'نسبة الثقة':'Confidence'}</small></div>
      </header>

      <div className="detail-modal-grid">
        <aside className="detail-left-panel">
          {hasImage ? <img className="detail-plant-image" src={record.imageUrl} alt="Plant diagnosis"/> : <div className="detail-sensor-placeholder"><Cpu size={48}/><b>{ar?'تشخيص بالحساسات':'Sensor-only diagnosis'}</b><small>{ar?'لا توجد صورة محفوظة لهذا التشخيص.':'No image was saved for this diagnosis.'}</small></div>}
          <div className="detail-summary-facts">
            <span><small>{ar?'نوع التشخيص':'Analysis Type'}</small><b>{sourceText}</b></span>
            <span><small>{ar?'القطاع':'Sector'}</small><b>{localizeValue(record.sectorName,t)}</b></span>
            <span><small>{ar?'المرض أو المشكلة':'Disease / Issue'}</small><b>{disease}</b></span>
            {displayPlantType(record,record,t) && <span><small>{ar?'نوع النبات':'Plant type'}</small><b>{displayPlantType(record,record,t)}</b></span>}
          </div>
        </aside>

        <main className="detail-right-panel">
          <section className="detail-section model-section">
            <div className="detail-section-title"><Sparkles size={18}/><h3>{ar?'نتيجة النموذج بالعربي':'User-friendly model result'}</h3></div>
            <h4>{displayText(headline,t)}</h4>
            <p>{displayText(record.diagnosis,t)}</p>
            <div className="detail-explanation-grid">
              <span><small>{ar?'تفسير الحساسات':'Sensor Explanation'}</small><b>{displayText(record.sensor_status,t)}</b></span>
              <span><small>{ar?'تفسير الصورة':'Image Explanation'}</small><b>{displayText(record.image_status,t)}</b></span>
            </div>
          </section>

          <section className="detail-section">
            <div className="detail-section-title"><Cpu size={18}/><h3>{ar?'سجل القراءات':'Sensor Readings'}</h3></div>
            {rows.length ? <div className="detail-reading-grid">{rows.map(r=><span key={r.key}><small>{r.label}</small><b>{r.value}</b></span>)}</div> : <p>{ar?'لا توجد قراءات حساسات محفوظة مع هذا التشخيص.':'No sensor readings were saved with this diagnosis.'}</p>}
          </section>

          {metrics.length ? <section className="detail-section">
            <div className="detail-section-title"><BarChart3 size={18}/><h3>{ar?'تحليل الصورة':'Image Analysis'}</h3></div>
            <div className="detail-metric-bars">{metrics.map(m=><div key={m.key} className={`metric-row ${m.key}`}><div><b>{m.value.toFixed(1)}%</b><span>{m.label}</span></div><em><i style={{width:`${m.value}%`}}/></em></div>)}</div>
          </section> : null}

          <section className="detail-section">
            <div className="detail-section-title"><Zap size={18}/><h3>{ar?'التوصيات':'Recommendations'}</h3></div>
            <div className="detail-recommendation-list">{treatments.map((item,i)=><article key={i}><b>{ar?`توصية ${i+1}`:`Recommendation ${i+1}`}</b><p>{displayText(item,t)}</p></article>)}</div>
          </section>

          <section className="detail-section">
            <div className="detail-section-title"><CheckCircle2 size={18}/><h3>{ar?'الإجراءات المطلوبة':'Required Actions'}</h3></div>
            <ul className="detail-actions-list">{actions.map((item,i)=><li key={i}>{displayText(item,t)}</li>)}</ul>
          </section>

          {tips.length ? <section className="detail-section tips-section">
            <div className="detail-section-title"><Camera size={18}/><h3>{ar?'نصائح تصوير أفضل':'Better Capture Tips'}</h3></div>
            <ul className="detail-actions-list">{tips.map((item,i)=><li key={i}>{displayText(item,t)}</li>)}</ul>
          </section> : null}

          <section className="detail-ai-note"><AlertTriangle size={17}/><span>{note}</span></section>
        </main>
      </div>

      <footer className="detail-modal-actions"><button className="primary-btn" onClick={()=>downloadDiagnosisReport(result,t)}><FileText size={16}/>{labels.report}</button><button className="secondary-btn" onClick={onClose}>{t('closeAssistant')}</button></footer>
    </section>
  </div>;
}

function numericOrEmpty(value){
  if (value === undefined || value === null || value === '') return '';
  const n = Number(value);
  return Number.isFinite(n) ? n : '';
}

function formatDateTime(value){
  if(!value) return '-';
  const d=new Date(value);
  if(Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString([], {year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit'});
}

function normalizeServiceSensorPayload(payload={}){
  const root = payload || {};
  const data = root.data || root.reading || root.readings || root.latest || root.latestReading || root.sensor || root.sensorData || root.sensor_data || root;
  const nested = data.data || data.reading || data.latest || data.sensor || {};
  const air = pickObject(root.air, data.air, nested.air);
  const soil = pickObject(root.soil, data.soil, nested.soil);
  const analysis = pickObject(root.analysis, data.analysis, nested.analysis, root.analysisResult, data.analysisResult, nested.analysisResult);
  const sectorObject = pickObject(root.sector, data.sector, nested.sector, root.sectorId, data.sectorId, nested.sectorId);
  const deviceObject = pickObject(root.device, data.device, nested.device, root.deviceId, data.deviceId, nested.deviceId);
  const merged = { ...root, ...data, ...nested };
  const imageUrl = firstNonEmpty(
    root.image_url, root.imageUrl, root.latest_image, root.latestImage, root.snapshot_url, root.snapshotUrl,
    data.image_url, data.imageUrl, data.snapshot_url, data.snapshotUrl, nested.image_url, nested.imageUrl
  );
  const now = new Date().toLocaleString();
  const airTemperatureValue = numericOrEmpty(firstNonEmpty(merged.temperature, merged.airTemperature, merged.air_temperature, merged.temp, air.temperature, air.temp));
  const soilTempSource = firstNonEmpty(
    merged.soilTemp, merged.soil_temp, merged.soilTemperature, merged.soil_temperature, merged.soilTempC, merged.soil_temperature_c,
    merged.rootZoneTemp, merged.root_zone_temp, merged.root_temp,
    soil.temperature, soil.temp, soil.temperatureC, soil.tempC,
    isPlainObject(soil.temperature) ? soil.temperature.value : undefined,
    isPlainObject(soil.temp) ? soil.temp.value : undefined
  );
  const soilTemperatureValue = numericOrEmpty(firstNonEmpty(soilTempSource, airTemperatureValue));
  const soilTempEstimated = !firstNonEmpty(soilTempSource, '') && airTemperatureValue !== '';
  const reading = {
    id: firstNonEmpty(merged._id, merged.id, merged.sensorId, merged.readingId, Date.now()),
    time: firstNonEmpty(merged.time, merged.timestamp, merged.createdAt, merged.created_at, new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})),
    lastUpdated: firstNonEmpty(merged.lastUpdated, merged.last_updated, merged.updatedAt, merged.createdAt, merged.created_at, merged.timestamp, now),
    cropType: firstNonEmpty(merged.cropType, merged.crop_type, merged.crop, merged.plantType, merged.plant_type, sectorObject.cropType, sectorObject.crop, ''),
    temperature: airTemperatureValue,
    humidity: numericOrEmpty(firstNonEmpty(merged.humidity, merged.hum, merged.Hum, air.humidity, air.hum)),
    soilMoisture: numericOrEmpty(firstNonEmpty(merged.soilMoisture, merged.soil_moisture, merged.moisture, merged.Soil, soil.moisture, soil.soilMoisture, soil.Soil, merged.soil)),
    soilTemp: soilTemperatureValue,
    soilTempEstimated,
    light: firstNonEmpty(merged.light, merged.lightLevel, merged.light_level, merged.lux, ''),
    sectorId: firstNonEmpty(isPlainObject(merged.sectorId) ? merged.sectorId._id || merged.sectorId.id : merged.sectorId, merged.sector_id, sectorObject._id, sectorObject.id, ''),
    sectorName: firstNonEmpty(merged.sectorName, merged.sector_name, sectorObject.name, sectorObject.sectorName, ''),
    sector: firstNonEmpty(merged.sectorName, merged.sector_name, sectorObject.name, sectorObject.sectorName, ''),
    deviceSerial: firstNonEmpty(merged.deviceSerial, merged.device_serial, merged.serial, deviceObject.deviceSerial, deviceObject.serial, ''),
    deviceName: firstNonEmpty(merged.deviceName, merged.device_name, deviceObject.deviceName, deviceObject.name, ''),
    final_status: firstNonEmpty(merged.final_status, merged.finalStatus, analysis.final_status, analysis.finalStatus, analysis.status, merged.status, 'Healthy'),
    confidence: cleanConfidenceValue(firstNonEmpty(merged.confidence, merged.final_confidence, analysis.confidence, .91)),
    recommendation: firstNonEmpty(analysis.recommendation, analysis.recommendations, merged.recommendation, merged.message, ''),
    source: root.source || data.source || 'service'
  };
  return { reading, imageUrl };
}


function hasUsefulSensorReading(reading = {}){
  return ['temperature','humidity','soilMoisture','soilTemp','light'].some((key)=>{
    const value = reading?.[key];
    return value !== undefined && value !== null && String(value).trim() !== '';
  });
}

function extractFirstHistoryReading(payload = {}){
  const rows = asArray(payload, ['history','readings','records','rows','items','sensorData','sensor_data','data','results']);
  if (!rows.length) return null;
  return rows
    .slice()
    .sort((a,b)=> new Date(b.createdAt || b.created_at || b.updatedAt || b.timestamp || b.date || 0) - new Date(a.createdAt || a.created_at || a.updatedAt || a.timestamp || a.date || 0))[0];
}

const normalizeDiagnosisDevice = (item = {}, idx = 0) => {
  const sectorObj = typeof item.sector === 'object' && item.sector ? item.sector : {};
  return {
    id: firstNonEmpty(item._id, item.id, item.deviceId, item.device_id, idx + 1),
    name: firstNonEmpty(item.deviceName, item.name, item.device_name, item.label, 'EcoSense Device'),
    deviceSerial: firstNonEmpty(item.deviceSerial, item.device_serial, item.serial, item.serialNumber, item.device_serial_number, ''),
    sectorId: firstNonEmpty(item.sectorId, item.sector_id, item.assignedSectorId, sectorObj._id, sectorObj.id, typeof item.sector === 'string' ? item.sector : ''),
    sectorName: firstNonEmpty(item.sectorName, item.sector_name, sectorObj.name, sectorObj.sectorName, ''),
  };
};

const isRealDeviceSerial = (value = '') => {
  const serial = String(value || '').trim();
  return Boolean(serial) && !/^ESP32-GENERIC-UNIT$/i.test(serial) && !/^web-manual-/i.test(serial);
};

function makeSimulationReading(){
  const samples=[
    { temperature:25, humidity:60, soilMoisture:45, soilTemp:24, light:'Sufficient', final_status:'Healthy', confidence:.91 },
    { temperature:34, humidity:40, soilMoisture:28, soilTemp:30, light:'Medium', final_status:'Moderate Stress', confidence:.82 },
    { temperature:39, humidity:24, soilMoisture:17, soilTemp:35, light:'Low', final_status:'High Stress', confidence:.88 },
    { temperature:29, humidity:55, soilMoisture:36, soilTemp:27, light:'Sufficient', final_status:'Moderate Stress', confidence:.79 }
  ];
  const reading = samples[Math.floor(Math.random()*samples.length)];
  const now = new Date();
  return { id: Date.now(), time: now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}), lastUpdated: now.toLocaleString(), source:'simulation', ...reading };
}

async function fileFromImageUrl(url){
  const response = await fetch(url, { mode:'cors' });
  if(!response.ok) throw new Error('image-fetch-failed');
  const blob = await response.blob();
  return new File([blob], 'service-captured-plant-image.jpg', { type: blob.type || 'image/jpeg' });
}

async function fileFromDataUrl(dataUrl, filename = 'plant-diagnosis-image.jpg') {
  if (!dataUrl || !String(dataUrl).startsWith('data:')) throw new Error('not-a-data-url');
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const type = blob.type || 'image/jpeg';
  const extension = type.includes('png') ? 'png' : type.includes('webp') ? 'webp' : 'jpg';
  return new File([blob], filename.replace(/\.(jpg|jpeg|png|webp)$/i, '') + '.' + extension, { type });
}

async function imageFileFromDiagnosisResult(result = {}, imageUrl = '') {
  if (typeof File !== 'undefined' && result.__sourceFile instanceof File) return result.__sourceFile;
  if (typeof Blob !== 'undefined' && result.__sourceFile instanceof Blob) {
    return new File([result.__sourceFile], 'plant-diagnosis-image.jpg', { type: result.__sourceFile.type || 'image/jpeg' });
  }
  const src = imageUrl || result.image_preview || result.imageUrl || result.image_url || '';
  if (String(src).startsWith('data:')) return fileFromDataUrl(src);
  if (src) return fileFromImageUrl(src);
  throw new Error('missing-image-file-for-save');
}

function Diagnosis(){ 
  const {t, lang}=useLang(); 
  const cameraInputRef=useRef(null);
  const galleryInputRef=useRef(null);
  const [diagnosisMode,setDiagnosisMode]=useState('combined'); 
  const [sensorData,setSensorData]=useState({ cropType:'', temperature:'', humidity:'', soilMoisture:'', soilTemp:'', light:'', sectorId:'', sectorName:'', sector:'', deviceSerial:'', source:'service', lastUpdated:'' }); 
  const [file,setFile]=useState(null); 
  const [preview,setPreview]=useState(''); 
  const [serviceImageUrl,setServiceImageUrl]=useState('');
  const [result,setResult]=useState(null); 
  const [loading,setLoading]=useState(false); 
  const [mode,setMode]=useState('apiMode'); 
  const [sensorState,setSensorState]=useState({status:'loading', source:'service', lastUpdated:'', error:''});
  const [diagnosisSectors,setDiagnosisSectors]=useState([]);
  const [diagnosisDevices,setDiagnosisDevices]=useState([]);

  useEffect(()=>{
    let mounted=true;
    Promise.allSettled([sectorsAPI.getAll(), devicesAPI.getAll()]).then(([sectorsRes, devicesRes])=>{
      if(!mounted) return;
      const sectorPayload = sectorsRes.status === 'fulfilled' ? sectorsRes.value.data : {};
      const devicePayload = devicesRes.status === 'fulfilled' ? devicesRes.value.data : {};
      const list=asArray(sectorPayload,['sectors','items','rows','records']);
      const mapped=list.map((item,idx)=>({
        id:item._id || item.id || item.sectorId || idx+1,
        name:item.name || item.sectorName || item.sector_name || `Sector ${idx+1}`,
        crop:item.crop || item.cropType || item.plantType || '',
        status:item.status || item.final_status || 'Healthy',
      }));
      const deviceList = asArray(devicePayload,['devices','items','rows','records']).map(normalizeDiagnosisDevice).filter(d=>d.deviceSerial);
      if(mapped.length){ setDiagnosisSectors(mapped); setStore('sph_sectors',mapped); }
      if(deviceList.length) setDiagnosisDevices(deviceList);
      if(mapped.length) {
        setSensorData(prev=>{
          const selected = prev.sectorId ? mapped.find(s=>String(s.id)===String(prev.sectorId)) : null;
          if(!selected) return { ...prev, cropType: prev.cropType || '' };
          const device = deviceList.find(d=>String(d.sectorId)===String(selected.id)) || null;
          return {
            ...prev,
            sectorId: prev.sectorId,
            sectorName: prev.sectorName || selected.name,
            sector: prev.sector || selected.name,
            cropType: prev.cropType || '',
            deviceSerial: prev.deviceSerial || device?.deviceSerial || '',
            device_serial: prev.device_serial || device?.deviceSerial || '',
          };
        });
      }
    }).catch(()=>{});
    return()=>{mounted=false};
  },[]);

  const getDeviceForSector = (sectorId = '') =>
    diagnosisDevices.find(d=>String(d.sectorId)===String(sectorId) && d.deviceSerial) ||
    diagnosisDevices.find(d=>d.deviceSerial) || null;

  const effectiveCropTypeForDiagnosis = (sector = null) =>
    firstNonEmpty(sector?.crop, sensorData.cropType, sensorData.crop, '');

  const applyReading=(reading, source='service')=>{
    const selectedSector = diagnosisSectors.find(s=>String(s.id)===String(sensorData.sectorId)) || null;
    const clean={
      ...sensorData,
      ...reading,
      sectorId: firstNonEmpty(reading.sectorId, reading.sector_id, sensorData.sectorId, selectedSector?.id, ''),
      sector_id: firstNonEmpty(reading.sector_id, reading.sectorId, sensorData.sectorId, selectedSector?.id, ''),
      sectorName: firstNonEmpty(reading.sectorName, reading.sector_name, reading.sector, sensorData.sectorName, selectedSector?.name, ''),
      sector: firstNonEmpty(reading.sector, reading.sectorName, reading.sector_name, sensorData.sector, selectedSector?.name, ''),
      cropType: firstNonEmpty(reading.cropType, reading.crop, reading.plantType, sensorData.cropType, ''),
      deviceSerial: firstNonEmpty(reading.deviceSerial, reading.device_serial, sensorData.deviceSerial, sensorData.device_serial, getDeviceForSector(firstNonEmpty(reading.sectorId, reading.sector_id, sensorData.sectorId, selectedSector?.id, ''))?.deviceSerial, ''),
      device_serial: firstNonEmpty(reading.device_serial, reading.deviceSerial, sensorData.device_serial, sensorData.deviceSerial, getDeviceForSector(firstNonEmpty(reading.sectorId, reading.sector_id, sensorData.sectorId, selectedSector?.id, ''))?.deviceSerial, ''),
      lastUpdated: reading.lastUpdated || new Date().toLocaleString(),
      source
    };
    setSensorData(clean);
    const all=[clean,...getStore('sph_readings',[])].slice(0,30);
    setStore('sph_readings', all);
    try { window.dispatchEvent(new CustomEvent('sph-sensors-updated', { detail: clean })); } catch {}
  };

  const loadServiceSensors=async({silent=false, fallback=false}={})=>{
    if(!silent) setSensorState(s=>({...s,status:'loading', error:''}));
    // Diagnosis Center manual flow is independent from sectors.
    // Only backend readings may carry a sectorId from the server response.
    const requestedSectorId = '';
    const selectedSector = null;

    const mergeReading = (reading = {}) => ({
      ...sensorData,
      ...reading,
      sectorId: reading.sectorId || requestedSectorId || sensorData.sectorId || selectedSector?.id || '',
      sectorName: reading.sectorName || reading.sector || sensorData.sectorName || selectedSector?.name || '',
      sector: reading.sectorName || reading.sector || sensorData.sector || selectedSector?.name || '',
      cropType: firstNonEmpty(reading.cropType, reading.crop, reading.plantType, sensorData.cropType, ''),
      temperature: reading.temperature !== '' ? reading.temperature : sensorData.temperature,
      humidity: reading.humidity !== '' ? reading.humidity : sensorData.humidity,
      soilMoisture: reading.soilMoisture !== '' ? reading.soilMoisture : sensorData.soilMoisture,
      soilTemp: reading.soilTemp !== '' ? reading.soilTemp : (sensorData.soilTemp || reading.temperature || sensorData.temperature),
      soilTempEstimated: reading.soilTempEstimated || (!reading.soilTemp && Boolean(reading.temperature || sensorData.temperature)),
      light: reading.light || sensorData.light,
    });

    try{
      let response;
      try {
        response = await sensorsAPI.getLatest(requestedSectorId);
      } catch (sectorLatestError) {
        // Some backend builds expose /sensors/latest globally only. Retry without sectorId before showing offline.
        if (requestedSectorId) response = await sensorsAPI.getLatest();
        else throw sectorLatestError;
      }
      const {reading,imageUrl}=normalizeServiceSensorPayload(response.data || {});
      const mergedReading = mergeReading(reading);
      if(!hasUsefulSensorReading(mergedReading)) throw new Error('empty-latest-sensor-reading');
      applyReading(mergedReading,'service');
      if(imageUrl){ setServiceImageUrl(imageUrl); setPreview(imageUrl); }
      setSensorState({status:'connected', source:'service', lastUpdated: mergedReading.lastUpdated, error:''});
      if(!silent) addNotification(t('sensorsUpdated'),'success');
    }catch(err){
      try {
        // Fallback to backend history if /sensors/latest is empty/unavailable, because history is now the reliable saved source.
        const historyRes = await sensorsAPI.getHistory({ ...(requestedSectorId ? { sectorId: requestedSectorId } : {}), limit: 1 });
        const first = extractFirstHistoryReading(historyRes.data || {});
        if (first) {
          const { reading } = normalizeServiceSensorPayload(first);
          const mergedReading = mergeReading(reading);
          if (hasUsefulSensorReading(mergedReading)) {
            applyReading(mergedReading,'history');
            setSensorState({status:'cached', source:'history', lastUpdated: mergedReading.lastUpdated, error:''});
            if(!silent) addNotification(t('sensorsUpdated'),'success');
            return;
          }
        }
      } catch(historyErr) {
        // Keep the original error as the reason shown to the user.
      }

      if(fallback){
        const reading=makeSimulationReading();
        applyReading(reading,'simulation');
        setSensorState({status:'simulation', source:'simulation', lastUpdated: reading.lastUpdated, error:String(err?.message||'offline')});
        if(!silent) addNotification(t('sensorsSimulationUpdated'),'warning');
      }else{
        // If the user has manually entered values, do not show a misleading offline/service-required state.
        if (hasUsefulSensorReading(sensorData)) {
          setSensorState({status:'manual', source:'manual', lastUpdated: sensorData.lastUpdated || new Date().toLocaleString(), error:String(err?.message||'offline')});
        } else {
          setSensorState(s=>({...s,status:'offline', source:'service', error:String(err?.message||'offline')}));
        }
        if(!silent) addNotification(t('sensorsOfflineFallback'),'warning');
      }
    }
  };

  useEffect(()=>{
    loadServiceSensors({silent:true, fallback:false});
  },[]);

  const refreshServiceImage = async()=>{
    try{
      const sectorId = sensorData.sectorId || '';
      const response = sectorId
        ? await imagesAPI.getHistoryWithFallback({ sectorId, limit:1 })
        : await imagesAPI.getHardwareHistory({ limit:1 });
      const record = firstImageRecord(response.data || {});
      const info = imageInfoFromRecord(record);
      const latestUrl = info.url || extractImageUrlFromPayload(response.data || {});
      if(!latestUrl){
        addNotification(lang==='ar'?'لا توجد صورة محفوظة من الباك إند حاليًا.':'No saved backend image is available right now.','warning');
        return;
      }
      setFile(null);
      setServiceImageUrl(latestUrl);
      setPreview(latestUrl);
      addNotification(lang==='ar'?'تم جلب آخر صورة من الباك إند.':'Latest backend image loaded.','success');
    }catch(err){
      addNotification(userFriendlyServiceError(err, lang) || (lang==='ar'?'تعذر تحديث الصورة من الباك إند.':'Could not refresh the image from backend.'),'warning');
    }
  };

  const choose=e=>{ 
    const f=e.target.files?.[0]; 
    if(!f) return;
    setFile(f); setServiceImageUrl('');
    const reader=new FileReader(); 
    reader.onload=()=>setPreview(reader.result); 
    reader.readAsDataURL(f);
    addNotification(t('imageSelected'),'success');
    e.target.value='';
  };
  const openCamera=()=>{
    if(cameraInputRef.current){ cameraInputRef.current.value=''; cameraInputRef.current.click(); }
  };
  const openGallery=()=>{
    if(galleryInputRef.current){ galleryInputRef.current.value=''; galleryInputRef.current.click(); }
  };
  const clearImage=()=>{ setFile(null); setPreview(''); setServiceImageUrl(''); addNotification(t('imageRemoved'),'success'); };
  const change=(k,v)=>{
    const next={...sensorData,[k]:v,lastUpdated:new Date().toLocaleString(),source:'manual'};
    setSensorData(next);
    setSensorState({status:'manual', source:'manual', lastUpdated: next.lastUpdated, error:''});
  };
  const refresh=()=>loadServiceSensors({fallback:false});
  const simulate=()=>{
    const simulated=makeSimulationReading();
    // Simulation is a manual AI-model flow, not a sector/backend saved-reading flow.
    const next={...simulated, cropType:firstNonEmpty(sensorData.cropType, ''), sectorId:'', sector_id:'', sectorName:'', sector:'', deviceSerial:'', device_serial:''};
    applyReading(next,'simulation');
    setSensorState({status:'simulation',source:'simulation',lastUpdated:next.lastUpdated,error:''});
    addNotification(t('sensorsSimulationUpdated'),'success');
  };

  async function analyze(){ 
    setLoading(true); setResult(null); 
    const sensorInputSource = String(sensorState.source || sensorState.status || sensorData.source || '').toLowerCase();
    const hasSavedSensorSource = ['connected','cached','history','service'].some(key => sensorInputSource.includes(key)) && !['manual','simulation'].some(key => sensorInputSource.includes(key));
    const isManualLikeInput = ['manual','simulation'].some(key => sensorInputSource.includes(key)) || (hasUsefulSensorReading(sensorData) && !hasSavedSensorSource);
    const selectedSector = hasSavedSensorSource && sensorData.sectorId ? diagnosisSectors.find(s=>String(s.id)===String(sensorData.sectorId)) : null;
    const selectedDevice = hasSavedSensorSource && sensorData.sectorId ? getDeviceForSector(sensorData.sectorId) : null;
    const form = {
      // sector/device are sent only for real backend readings. Manual and simulated values are analyzed directly by the AI model.
      sectorId: hasSavedSensorSource ? (sensorData.sectorId || selectedSector?.id || '') : '',
      sectorName: hasSavedSensorSource ? (sensorData.sectorName || selectedSector?.name || sensorData.sector || '') : '',
      sector: hasSavedSensorSource ? (sensorData.sector || sensorData.sectorName || selectedSector?.name || '') : '',
      cropType: firstNonEmpty(sensorData.cropType, sensorData.crop, hasSavedSensorSource ? selectedSector?.crop : '', ''),
      deviceSerial: hasSavedSensorSource ? firstNonEmpty(sensorData.deviceSerial, sensorData.device_serial, selectedDevice?.deviceSerial, '') : '',
      device_serial: hasSavedSensorSource ? firstNonEmpty(sensorData.device_serial, sensorData.deviceSerial, selectedDevice?.deviceSerial, '') : '',
      temperature: sensorData.temperature, humidity: sensorData.humidity,
      soilMoisture: sensorData.soilMoisture, soilTemp: sensorData.soilTemp, light: sensorData.light
    };
    let apiOk=false, directData=null, backendSaved=false, endpointUsed='', historyEndpoint='', imageFileForSave=null, sensorAnalysisData=null; 
    try{ 
      const hasSector = Boolean(form.sectorId);
      const hasSensorValues = hasUsefulSensorReading(form);
      const wantsSensors = diagnosisMode === 'sensors' || diagnosisMode === 'combined';
      const wantsImage = diagnosisMode === 'image' || diagnosisMode === 'combined';
      const hasSavedSensorSource = Boolean(form.sectorId) && !isManualLikeInput;
      const isManualOrSimulatedSensors = Boolean(isManualLikeInput) || (hasUsefulSensorReading(sensorData) && !hasSavedSensorSource);

      if (wantsSensors && !hasSensorValues) {
        throw new Error(lang==='ar' ? 'لا توجد قراءات حساسات للتحليل. أدخل القيم يدويًا أو اضغط محاكاة القراءات أولًا.' : 'No sensor readings are available. Enter values manually or use simulation first.');
      }
      if (wantsSensors && isManualOrSimulatedSensors && !normalizeCropTypeForModel(firstNonEmpty(form.cropType, ''))) {
        throw new Error(lang==='ar' ? 'اختر نوع النبات أولًا. الموديل يدعم: نعناع، ذرة، فلفل، طماطم.' : 'Choose crop type first. The model supports: Mint, Corn, Pepper, Tomato.');
      }

      const resolveImageFile = async()=>{
        let imageFile=file;
        if(!imageFile && serviceImageUrl) imageFile = await fileFromImageUrl(serviceImageUrl);
        if (!imageFile) {
          addNotification(t('imageRequired'),'danger');
          throw new Error(lang==='ar' ? 'لا توجد صورة لتحليلها.' : 'No image is available for analysis.');
        }
        imageFileForSave = imageFile;
        return imageFile;
      };

      const analyzeImageWithModel = async(imageFile)=>{
        const fd = buildModelImageFormData(imageFile, form);
        const res = await diagnosisAPI.predictImageModel(fd);
        return { ...(res.data || res), __backendSaved: false, __backendEndpoint: 'POST https://amr2004-ecosense-ai.hf.space/api/predict_image' };
      };

      const uploadImageToBackend = async(imageFile)=>{
        if (!form.sectorId && !form.deviceSerial) {
          return analyzeImageWithModel(imageFile);
        }
        const fd = new FormData();
        fd.append('image', imageFile);
        if (form.sectorId) fd.append('sectorId', String(form.sectorId));
        if (form.deviceSerial) fd.append('deviceSerial', String(form.deviceSerial));
        try {
          const imageUploadRes = await imagesAPI.upload(fd);
          const uploadedPayload = imageUploadRes.data || imageUploadRes;
          const savedImageUrl = extractImageUrlFromPayload(uploadedPayload);
          if(savedImageUrl) setServiceImageUrl(savedImageUrl);
          return { ...(uploadedPayload || {}), __backendSaved: true, __backendEndpoint: 'POST /images/upload' };
        } catch (err) {
          if (!form.sectorId && !form.deviceSerial) {
            return analyzeImageWithModel(imageFile);
          }
          throw err;
        }
      };

      const analyzeSensorsWithModel = async()=>{
        const body = manualSensorDiagnosisBody(form);
        const canonicalCrop = normalizeCropTypeForModel(firstNonEmpty(form.cropType, sensorData.cropType, ''));
        if (canonicalCrop) body.cropType = canonicalCrop;
        delete body.sectorId; delete body.sector_id; delete body.deviceSerial; delete body.device_serial;
        const res = await diagnosisAPI.predictSensorsModel(body);
        return res.data || res;
      };

      const analyzeCombinedWithDirectModel = async(imageFile)=>{
        // Manual/simulated Image + Sensors goes directly to the combined model with field name "file".
        // This avoids backend rate-limit loops and prevents the 429 message from blocking the user result.
        const modelForm = { ...form, cropType: normalizeCropTypeForModel(firstNonEmpty(form.cropType, sensorData.cropType, '')) };
        const modelFd = () => buildModelImageFormData(imageFile, modelForm);
        try {
          const res = await diagnosisAPI.predictWithImageModel(modelFd());
          return { ...(res.data || res), __backendSaved: false, __backendEndpoint: 'POST https://Amrkhaled2004.pythonanywhere.com/api/predict_with_image' };
        } catch (pythonErr) {
          const res = await diagnosisAPI.predictWithImageModelHF(modelFd());
          return { ...(res.data || res), __backendSaved: false, __backendEndpoint: 'POST https://amr2004-ecosense-ai.hf.space/api/predict_with_image' };
        }
      };

      const analyzeBackendSensors = async () => {
        if (!form.sectorId) {
          return analyzeSensorsWithModel();
        }
        try {
          const res = await sensorsAPI.analyze(form.sectorId);
          return { ...(res.data || res), __backendSaved: true, __backendEndpoint: 'POST /sensors/analyze/:sectorId' };
        } catch (err) {
          const raw = String(err?.response?.data?.message || err?.response?.data?.error || err?.message || '');
          if (/لا توجد قراءات مسجلة|no readings|no sensor readings|not found readings|reading.*not.*found/i.test(raw)) {
            const modelResult = await analyzeSensorsWithModel();
            return { ...(modelResult || {}), __backendSaved: false, __backendEndpoint: 'POST https://amr2004-ecosense-ai.hf.space/api/predict_sensors' };
          }
          throw err;
        }
      };

      const analyzeManualSensorsWithBackend = async () => {
        // Manual and simulated sensor values must use the direct AI model.
        // /sensors/analyze/:sectorId is reserved for saved backend readings only.
        return analyzeSensorsWithModel();
      };

      const splitBackendImageAndSensors = async (imageFile) => {
        const parts = [];
        let imagePayload = null;
        let sensorPayload = null;
        if (wantsImage) {
          imagePayload = await uploadImageToBackend(imageFile);
          parts.push({ type: 'image', payload: imagePayload });
        }
        if (wantsSensors) {
          if (isManualOrSimulatedSensors) {
            sensorPayload = await analyzeManualSensorsWithBackend();
          } else {
            sensorPayload = await analyzeBackendSensors();
          }
          parts.push({ type: 'sensors', payload: sensorPayload });
        }
        return { imagePayload, sensorPayload, parts };
      };

      // Saved/backend-driven diagnosis only. Do not invent cropType and do not send default crop values.
      if (wantsImage && wantsSensors) {
        const imageFile = await resolveImageFile();
        if (isManualOrSimulatedSensors) {
          directData = await analyzeCombinedWithDirectModel(imageFile);
          endpointUsed = directData?.__backendEndpoint || (directData?.__backendSaved ? 'POST /diagnosis/analyze-combined or /predict_with_image' : 'POST https://Amrkhaled2004.pythonanywhere.com/api/predict_with_image');
          historyEndpoint = directData?.__backendSaved ? 'GET /images/history?sectorId=SECTOR_ID + GET /sensors/history?sectorId=SECTOR_ID' : '';
          backendSaved = Boolean(directData?.__backendSaved);
        } else {
          if (!form.sectorId) {
            directData = await analyzeCombinedWithDirectModel(imageFile);
            endpointUsed = directData?.__backendEndpoint || 'POST https://Amrkhaled2004.pythonanywhere.com/api/predict_with_image';
            historyEndpoint = '';
            backendSaved = false;
          } else {
            try {
              const combinedRes = await diagnosisAPI.analyzeCombined(buildBackendCombinedFormData(imageFile, form));
              directData = combinedRes.data || combinedRes;
              endpointUsed = 'POST /diagnosis/analyze-combined or /predict_with_image';
              historyEndpoint = 'GET /images/history?sectorId=SECTOR_ID + GET /sensors/history?sectorId=SECTOR_ID';
              backendSaved = true;
            } catch (combinedErr) {
              const { imagePayload, sensorPayload } = await splitBackendImageAndSensors(imageFile);
              directData = imagePayload && sensorPayload
                ? { ...(sensorPayload || {}), imageResult: imagePayload, sensorResult: sensorPayload, combined: true }
                : (imagePayload || sensorPayload);
              endpointUsed = 'POST /images/upload + POST /sensors/analyze/:sectorId';
              historyEndpoint = 'GET /images/history?sectorId=SECTOR_ID + GET /sensors/history?sectorId=SECTOR_ID';
              backendSaved = true;
            }
          }
        }
      } else if (wantsSensors) {
        directData = isManualOrSimulatedSensors ? await analyzeSensorsWithModel() : await analyzeBackendSensors();
        sensorAnalysisData = directData;
        endpointUsed = directData?.__backendEndpoint || (isManualOrSimulatedSensors ? 'POST https://amr2004-ecosense-ai.hf.space/api/predict_sensors' : (form.sectorId ? 'POST /sensors/analyze/:sectorId' : 'POST https://amr2004-ecosense-ai.hf.space/api/predict_sensors'));
        historyEndpoint = form.sectorId && !isManualOrSimulatedSensors ? 'GET /sensors/history?sectorId=SECTOR_ID' : '';
        backendSaved = Boolean(form.sectorId && !isManualOrSimulatedSensors && directData?.__backendSaved);
      } else if (wantsImage) {
        const imageFile = await resolveImageFile();
        directData = await uploadImageToBackend(imageFile);
        endpointUsed = directData?.__backendEndpoint || (form.sectorId ? 'POST /images/upload' : 'POST https://amr2004-ecosense-ai.hf.space/api/predict_image');
        historyEndpoint = form.sectorId && directData?.__backendSaved ? 'GET /images/history?sectorId=SECTOR_ID' : '';
        backendSaved = Boolean(form.sectorId && directData?.__backendSaved);
      }

      const data=directData;
      if(!data) throw new Error('Backend analysis returned no data.');
      const normalized=normalize(data, form, diagnosisMode !== 'sensors' && (!!file || !!serviceImageUrl || !!imageFileForSave));
      if (diagnosisMode === 'combined' && sensorAnalysisData) {
        const sensorNormalized = normalize(sensorAnalysisData, form, false);
        if (!normalized.actions?.length && sensorNormalized.actions?.length) normalized.actions = sensorNormalized.actions;
        if (!normalized.recommendations?.length && sensorNormalized.recommendations?.length) normalized.recommendations = sensorNormalized.recommendations;
        if (!normalized.sensor_status || normalized.sensor_status === '-') normalized.sensor_status = sensorNormalized.sensor_status;
      }
      normalized.diagnosis_mode=data.source || diagnosisMode;
      normalized.image_preview=preview || extractImageUrlFromPayload(data) || serviceImageUrl;
      normalized.__backendEndpoint=endpointUsed;
      normalized.__historyEndpoint=historyEndpoint;
      normalized.__backendSaved=backendSaved;
      normalized.__diagnosisMode=diagnosisMode;
      normalized.__inputForm=form;
      normalized.__manualSensorInput=isManualOrSimulatedSensors;
      if (imageFileForSave) normalized.__sourceFile=imageFileForSave;
      setResult(normalized); apiOk=true; 
      triggerModelAlerts(normalized,t);
      const finalTone = tone(normalized.final_status);
      const healthy = finalTone === 'green';
      addNotification(
        healthy
          ? (lang==='ar' ? 'تم التشخيص بنجاح - النبات سليم' : 'Diagnosis completed successfully - plant is healthy')
          : (finalTone === 'red'
            ? (lang==='ar' ? 'تم التشخيص - توجد حالة إجهاد مرتفع' : 'Diagnosis completed - high stress detected')
            : (lang==='ar' ? 'تم التشخيص - توجد حالة تحتاج متابعة' : 'Diagnosis completed - follow-up recommended')),
        healthy ? 'success' : finalTone === 'red' ? 'danger' : 'warning'
      );
      if (backendSaved) {
        diagnosisHistoryRuntimeCache.time = 0;
        diagnosisHistoryRuntimeCache.value = null;
        diagnosisHistoryRuntimeCache.promise = null;
        callBackendSavedEvent();
      }
    } catch (error) { 
      setResult(null);
      addNotification(userFriendlyServiceError(error, lang) || (lang==='ar' ? 'تعذر تشغيل التشخيص من الموديل. تأكد من الصورة والقيم ثم حاول مرة أخرى.' : 'Model diagnosis failed. Check the image/readings and try again.'),'danger');
    } finally { setMode(apiOk?'apiMode':'apiMode'); setLoading(false); }
  } 

  const sensorTone = ['connected','cached','manual'].includes(sensorState.status) ? 'success' : sensorState.status==='simulation' ? 'warning' : sensorState.status==='loading' ? 'info' : 'danger';
  const sensorLabel = sensorState.status==='connected' ? t('latestBackendReading') : sensorState.status==='cached' ? t('latestSavedReading') : sensorState.status==='manual' ? t('manualReadingsMode') : sensorState.status==='simulation' ? t('simulationMode') : sensorState.status==='loading' ? t('sensorLoading') : t('sensorOfflineState');
  const imageRequired = (diagnosisMode === 'image' || diagnosisMode === 'combined') && !file && !serviceImageUrl;
  const diagnosisLabels = lang === 'ar'
    ? { step1:'اختر نوع التحليل', step2:'أدخل الصورة / القراءات', step3:'اعرض النتيجة النهائية' }
    : { step1:'Choose analysis type', step2:'Add image / readings', step3:'Review final result' };
  return <>
    <PageHead title={t('diagnosisCenter')} sub={t('modelDrivenNote')} action={<span className="mode-pill">{t(mode)}</span>}/>
    <section className="diagnosis-workflow-strip">
      {[diagnosisLabels.step1, diagnosisLabels.step2, diagnosisLabels.step3].map((label,index)=><div className={index===0?'active':''} key={label}><span>{index+1}</span><b>{label}</b></div>)}
    </section>
    <Panel title={t('diagnosisMode')}>
      <div className="diagnosis-modes">
        <button className={diagnosisMode==='image'?'active':''} onClick={()=>setDiagnosisMode('image')}><Camera size={18}/>{t('imageOnlyMode')}</button>
        <button className={diagnosisMode==='sensors'?'active':''} onClick={()=>setDiagnosisMode('sensors')}><ThermometerSun size={18}/>{t('sensorsOnlyMode')}</button>
        <button className={diagnosisMode==='combined'?'active':''} onClick={()=>setDiagnosisMode('combined')}><ScanSearch size={18}/>{t('combinedMode')}</button>
      </div>
    </Panel>
    <div className="grid-2 diagnosis-mobile-grid">
      {diagnosisMode !== 'image' && <Panel title={t('currentSensorReadings')}>
        <div className={`sensor-connection-banner ${sensorTone}`}>
          <Activity size={18}/>
          <div><b>{sensorLabel}</b><span>{t('lastUpdated')}: {sensorState.lastUpdated || sensorData.lastUpdated || '-'}</span><span>{t('sensorSource')}: {sensorState.status==='connected'?t('backendLatestSource'):sensorState.status==='cached'?t('backendHistorySource'):sensorState.status==='manual'?t('manualSource'):sensorState.status==='simulation'?t('simulationSource'):t('serviceRequiredSource')}</span></div>
        </div>
        <p className="muted-copy">{sensorState.status==='simulation' ? t('sensorsOfflineFallback') : t('sensorUserHint')}</p>
        <div className="form-grid">
          <Field label={lang==='ar'?'نوع النبات':'Crop type'}><CropTypeSelect lang={lang} value={sensorData.cropType || ''} onChange={value=>change('cropType', value)} /></Field>
          {['connected','cached'].includes(sensorState.status) && (sensorData.sectorName || sensorData.sectorId) && <div className="backend-reading-note backend-reading-note-mint"><CheckCircle2 size={15}/><span>{lang==='ar'?'القراءة الحالية جاية من الباك إند':'Current reading is from backend'}</span><b>{localizeValue(sensorData.sectorName || sensorData.sector || sensorData.sectorId, t)}</b>{sensorData.soilTempEstimated && <small>{lang==='ar'?'حرارة التربة تقديرية من حرارة الجو لأن الهاردوير لا يرسل حساسًا منفصلًا لها.':'Soil temperature is estimated from air temperature because the hardware does not send a separate soil-temperature sensor.'}</small>}</div>}
          <Field label={t('temperature')}><input type="number" value={sensorData.temperature} onChange={e=>change('temperature',Number(e.target.value))}/></Field>
          <Field label={t('humidity')}><input type="number" value={sensorData.humidity} onChange={e=>change('humidity',Number(e.target.value))}/></Field>
          <Field label={t('soilMoisture')}><input type="number" value={sensorData.soilMoisture} onChange={e=>change('soilMoisture',Number(e.target.value))}/></Field>
          <Field label={`${t('soilTemp')}${sensorData.soilTempEstimated ? (lang==='ar'?' تقديرية':' estimated') : ''}`}><input type="number" value={sensorData.soilTemp} onChange={e=>change('soilTemp',Number(e.target.value))}/></Field>
          <Field label={t('light')}><select value={sensorData.light} onChange={e=>change('light',e.target.value)}><option value="Low">{t('low')}</option><option value="Medium">{t('medium')}</option><option value="Sufficient">{t('sufficient')}</option></select></Field>
        </div>
        <div className="mobile-action-row">
          <button className="secondary-btn" onClick={refresh} disabled={sensorState.status==='loading'}><Activity size={18}/>{t('refreshSensors')}</button>
          <button className="secondary-btn" onClick={simulate}><Sparkles size={18}/>{t('simulateReadings')}</button>
        </div>
      </Panel>}
      {diagnosisMode !== 'sensors' && <Panel title={t('uploadImage')}>
        <p className="muted-copy">{t('mobileUploadHint')}</p>
        <input ref={cameraInputRef} className="hidden-file-input" type="file" accept="image/*" capture="environment" aria-label={t('takePhoto')} onChange={choose}/>
        <input ref={galleryInputRef} className="hidden-file-input" type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" aria-label={t('chooseFromGallery')} onChange={choose}/>
        <div className="mobile-upload-choice">
          <button type="button" className="secondary-btn camera-choice" onClick={openCamera}><Camera size={18}/><span>{t('takePhoto')}</span><small>{t('takePhotoHint')}</small></button>
          <button type="button" className="secondary-btn gallery-choice" onClick={openGallery}><ImagePlus size={18}/><span>{t('chooseFromGallery')}</span><small>{t('galleryHint')}</small></button>
        </div>
        <div className={`dropzone image-scan-zone ${preview?'has-preview':''}`}>{preview?<><img src={preview} alt={t('imagePreview')}/><span className="image-source-pill">{serviceImageUrl ? t('latestServiceImage') : t('imagePreview')}</span><span className="image-ready-pill"><CheckCircle2 size={14}/>{t('imageReady')}</span></>:<div><UploadCloud size={48}/><b>{t('chooseImage')}</b><small>{t('mobileImagePickerNote')}</small></div>}</div>
        <div className="mobile-action-row image-backend-refresh-row"><button type="button" className="secondary-btn" onClick={refreshServiceImage}><ImagePlus size={16}/>{lang==='ar'?'تحديث الصورة':'Refresh Image'}</button>{preview && <button type="button" className="secondary-btn" onClick={openGallery}>{t('changeImage')}</button>}{preview && <button type="button" className="danger-btn" onClick={clearImage}><X size={16}/>{t('removeImage')}</button>}</div>
        {serviceImageUrl && <p className="muted-copy"><CheckCircle2 size={14}/> {t('combinedReady')}</p>}
      </Panel>}
    </div>
    <button className="primary-btn wide" onClick={analyze} disabled={loading || imageRequired}>{loading?<><Activity className="spin" size={18}/>{t('analyzing')}</>:<><ScanSearch size={18}/>{t('sendToModel')}</>}</button>
    {imageRequired && <p className="form-helper warning-text">{t('imageRequired')}</p>}
    {result && <div className="diagnosis-result-focus"><Result result={result}/></div>}
  </>
}
function ConfirmModal({title,message,onCancel,onConfirm,loading=false,cancelLabel='Cancel',confirmLabel='Delete'}){
  return <div className="modal-backdrop compact-confirm-backdrop" role="dialog" aria-modal="true">
    <section className="modal-card compact-confirm-modal">
      <button className="modal-close" onClick={onCancel} disabled={loading}><X size={18}/></button>
      <div className="confirm-icon"><AlertTriangle size={24}/></div>
      <h3>{title}</h3>
      <p>{message}</p>
      <div className="confirm-actions"><button className="secondary-btn" onClick={onCancel} disabled={loading}>{cancelLabel}</button><button className="danger-soft-btn" onClick={onConfirm} disabled={loading}>{loading?<><Activity className="spin" size={15}/>{confirmLabel}</>:confirmLabel}</button></div>
    </section>
  </div>
}

function Field({label,children}){ return <label className="field"><span>{label}</span>{children}</label> }

function getReturnedPlantType(record = {}, result = {}) {
  const raw = record.raw || result.raw_response || result.raw || result || {};
  const inputOnly = record.input_data || result.input_data || {};
  return firstNonEmpty(
    record.cropType, record.crop_type, record.plantType, record.plant_type, record.crop,
    result.cropType, result.crop_type, result.plantType, result.plant_type, result.crop,
    raw.cropType, raw.crop_type, raw.plantType, raw.plant_type, raw.crop,
    raw.data?.cropType, raw.data?.crop_type, raw.data?.plantType, raw.data?.plant_type, raw.data?.crop,
    raw.result?.cropType, raw.result?.crop_type, raw.result?.plantType, raw.result?.plant_type, raw.result?.crop,
    raw.prediction?.cropType, raw.prediction?.crop_type, raw.prediction?.plantType, raw.prediction?.plant_type, raw.prediction?.crop,
    ''
  ) || '';
}
function displayPlantType(record = {}, result = {}, t = (v)=>v) {
  const value = getReturnedPlantType(record, result);
  if (!value) return '';
  return localizeValue(value, t);
}

function normalize(raw, inputData={}, hasImage=false){
  const r = raw?.result || raw?.prediction || raw?.data?.result || raw?.data?.prediction || raw?.data || raw || {};
  const ia = r.image_analysis || r.imageAnalysis || r.analysisResult || r.analysis_result || r.analysis || r.image || {};
  const diagnosisObj = r.diagnosis && typeof r.diagnosis === 'object' ? r.diagnosis : null;
  const disease =
    r.disease_name ||
    r.disease_name_ar ||
    r.diseaseName ||
    r.predicted_disease ||
    r.disease ||
    r.class_name ||
    diagnosisObj?.visual_problem_ar ||
    diagnosisObj?.visual_problem ||
    ia.disease_name_ar ||
    ia.disease_name ||
    ia.diseaseName ||
    ia.visual_problem_ar ||
    ia.visual_problem ||
    r.visual_problem ||
    r.visualProblem ||
    '';

  const flags = r.backend_flags || r.backendFlags || r.flags || {};
  const risks = r.risk_factors || r.riskFactors || r.risks || [];
  const actions = r.actions || r.actions_ar || r.immediate_actions || r.immediateActions || ia.treatment_plan || ia.treatment_plan_ar || [];

  const confidenceValue =
    r.final_confidence ??
    r.image_confidence ??
    ia.confidence ??
    (typeof r.confidence === 'object' && r.confidence !== null
      ? Math.max(...Object.values(r.confidence).map(Number).filter(v=>!Number.isNaN(v)))
      : r.confidence) ??
    r.model_confidence ??
    0;

  const diagnosisText = diagnosisObj
    ? (diagnosisObj.explanation || diagnosisObj.primary_issue || diagnosisObj.visual_problem_ar || diagnosisObj.visual_problem || '-')
    : (r.diagnosis || r.summary || r.message || '-');

  const rawFinalStatus = r.final_status || r.finalStatus || r.final_status_ar || ia.final_status || ia.finalStatus || ia.status;
  const statusFallback = ['healthy','moderate stress','high stress','سليم','إجهاد متوسط','اجهاد متوسط','إجهاد مرتفع','اجهاد مرتفع'].includes(String(r.status || '').toLowerCase()) ? r.status : '';
  const finalStatus = rawFinalStatus || statusFallback || '-';

  return {
    cropType: firstNonEmpty(r.cropType, r.crop_type, r.plantType, r.plant_type, r.crop, ia.cropType, ia.crop_type, ia.plantType, ia.plant_type, ia.crop, ''),
    final_status: finalStatus,
    sensor_status: r.sensor_status || r.sensorStatus || '-',
    image_status: r.image_status || r.imageStatus || ia.image_stress || ia.final_status || (hasImage ? 'Analyzed' : 'No Image'),
    disease_name: disease || '-',
    confidence: Number(confidenceValue || 0),
    final_confidence: r.final_confidence ?? Number(confidenceValue || 0),
    sensor_confidence: r.sensor_confidence || r.confidence || null,
    image_confidence: r.image_confidence ?? ia.confidence ?? null,
    severity: r.severity || '-',
    diagnosis: diagnosisText,
    diagnosis_details: diagnosisObj || null,
    recommendations: Array.isArray(r.recommendations) ? r.recommendations : (Array.isArray(r.recommendations_ar) ? r.recommendations_ar : (firstNonEmpty(r.recommendations, r.recommendations_ar, ia.recommendation, ia.recommendations, '') ? [firstNonEmpty(r.recommendations, r.recommendations_ar, ia.recommendation, ia.recommendations, '')] : [])),
    actions: Array.isArray(actions) ? actions : (actions ? [actions] : []),
    backend_flags: flags,
    risk_factors: Array.isArray(risks) ? risks : (risks ? [risks] : []),
    image_analysis: ia,
    safety_layer: r.safety_layer || null,
    notification: r.notification || null,
    monitoring: r.monitoring || [],
    suspicious_regions: r.suspicious_regions || r.suspiciousRegions || ia.suspicious_regions || ia.bounding_boxes || [],
    likely_cause: r.likely_cause || r.likelyCause || r.cause || diagnosisObj?.primary_issue || '',
    treatment_plan: r.treatment_plan || r.treatmentPlan || ia.treatment_plan || null,
    model_alerts: r.alerts || r.model_alerts || r.notifications || [],
    input_data: r.input_data || r.inputData || inputData,
    raw_response: r
  };
}
function triggerModelAlerts(result,t){
  const alerts = [];
  if(Array.isArray(result.model_alerts)) alerts.push(...result.model_alerts.map(a=>typeof a==='string'?a:(a.message||a.text||'')));
  const input = result.input_data || {};
  if(Number(input.soilMoisture) < 30) alerts.push(t('lowSoilMoistureAlert'));
  if(Number(input.temperature) > 34) alerts.push(t('highTemperatureAlert'));
  if(String(result.disease_name||'').toLowerCase().includes('fung') || String(result.disease_name||'').includes('فطر')) alerts.push(t('fungalRiskAlert'));
  if(isDanger(result.final_status)) alerts.push(`${t('alarmAlert')}: ${translateModelText(result.diagnosis,t)}`);
  [...new Set(alerts.filter(Boolean))].forEach(msg=>addNotification(msg, isDanger(result.final_status)?'danger':'warning'));
}


function displayText(value, t) {
  if (value == null || value === '') return '—';

  if (typeof value === 'string' || typeof value === 'number') {
    return translateModelText(value, t);
  }

  if (Array.isArray(value)) {
    return value.map(v => displayText(v, t)).filter(Boolean).join(' - ');
  }

  if (typeof value === 'object') {
    return translateModelText(
      value.title ||
      value.details ||
      value.message ||
      value.text ||
      value.primary_issue ||
      value.explanation ||
      value.name ||
      JSON.stringify(value),
      t
    );
  }

  return String(value);
}

function getActionTitle(action, t) {
  if (!action) return '—';
  if (typeof action === 'string') return translateModelText(action, t);
  return translateModelText(action.title || action.code || 'Action', t);
}

function getActionDetails(action, t) {
  if (!action || typeof action === 'string') return '';
  return translateModelText(action.details || action.message || action.text || '', t);
}

function Result({result}){
  const {t, lang}=useLang();
  const nav = useNavigate();
  const [savingHistory,setSavingHistory]=useState(false);
  const ar=isArabicUI(t);
  const resultRecord = normalizeDiagnosisRecord({
    ...result,
    analysisType: result.__diagnosisMode || result.diagnosis_mode || (result.image_preview ? 'Image + Sensors' : 'Sensors'),
    type: result.__diagnosisMode || result.diagnosis_mode,
    imageUrl: result.image_preview || result.imageUrl || '',
    sensorReadings: result.input_data || result.sensorReadings || {},
    date: result.createdAt || new Date().toISOString(),
    raw: result
  }, 0);
  const statusTone=tone(resultRecord.final_status);
  const confidence=Math.round((resultRecord.confidence||0)*100);
  const rows=readingRows(resultRecord,t).filter(r=>r.value !== '—');
  const metrics=imageMetricRows(resultRecord,t);
  const recommendations=treatmentItems(resultRecord,t);
  const actions=actionItems(resultRecord,t);
  const tips=captureTips(resultRecord,t);
  const headline=diagnosisHeadline(resultRecord,t);
  const disease=translateModelText(!resultRecord.disease_name || String(resultRecord.disease_name).trim()==='-' ? t('noDiseaseDetected') : resultRecord.disease_name,t);
  const sourceText = resultRecord.imageUrl && rows.length ? (ar?'صورة + حساسات':'Image + Sensors') : resultRecord.imageUrl ? (ar?'صورة':'Image') : (ar?'حساسات':'Sensors');
  const input = result.input_data || {};
  const [sectors,setSectors]=useState(getStore('sph_sectors', []));
  const [sectorId,setSectorId]=useState(result.__inputForm?.sectorId || input.sectorId || '');
  useEffect(()=>{
    let mounted=true;
    sectorsAPI.getAll().then(({data})=>{
      const list=asArray(data,['sectors','items','rows','records']);
      if(!mounted || !list.length) return;
      const mapped=list.map((item,idx)=>({
        id:item._id || item.id || item.sectorId || item.sector_id || '',
        name:item.name || item.sectorName || item.sector_name || `Sector ${idx+1}`,
        crop:item.crop || item.cropType || item.plantType || '',
      })).filter(s=>s.id);
      setSectors(mapped); setStore('sph_sectors',mapped);
    }).catch(()=>{});
    return()=>{mounted=false};
  },[]);
  const linkToSector=async()=>{
    const selectedSectorId = String(sectorId || '').trim();
    if(!selectedSectorId){ addNotification(lang==='ar'?'اختر قطاعًا صحيحًا أولًا.':'Choose a valid sector first.','warning'); return; }
    try{
      const resultMode = normalizeDiagnosisType(result.__diagnosisMode || result.diagnosis_mode || (result.image_preview ? 'image' : 'sensors'));
      const resultModeLower = String(resultMode || '').toLowerCase();
      const hasImage = Boolean(result.__sourceFile || result.image_preview || result.imageUrl || result.image_url);
      const sensorForm = { ...(result.__inputForm || input || {}), sectorId:selectedSectorId, sector_id:selectedSectorId };
      const hasSensors = hasUsefulSensorReading(sensorForm);
      let linked = false;

      if(hasSensors && (resultModeLower.includes('sensor') || resultModeLower.includes('combined') || !hasImage)){
        try{
          await diagnosisAPI.analyzeManualSensors(manualSensorDiagnosisBody(sensorForm));
        }catch(err){
          if(isMissingEndpointError(err)) throw new Error(manualSensorsEndpointMessage(lang));
          throw err;
        }
        linked = true;
      }

      if(hasImage && (resultModeLower.includes('image') || resultModeLower.includes('combined') || !hasSensors)){
        const imageFile = await imageFileFromDiagnosisResult(result, result.image_preview || result.imageUrl || result.image_url || '');
        const fd = new FormData();
        fd.append('image', imageFile);
        fd.append('sectorId', selectedSectorId);
        const linkDeviceSerial = firstNonEmpty(sensorForm.deviceSerial, sensorForm.device_serial, result.deviceSerial, result.device_serial, '');
        if (linkDeviceSerial) fd.append('deviceSerial', String(linkDeviceSerial));
        await imagesAPI.upload(fd);
        linked = true;
      }

      if(!linked){
        throw new Error(lang==='ar'
          ? 'لا يوجد ملف صورة أو قراءات قابلة لإعادة الإرسال. لو المطلوب ربط تشخيص قديم بقطاع بدون إعادة تحليل، الباك إند يحتاج endpoint مثل PATCH /api/diagnoses/:id/sector.'
          : 'No image file or readings are available to re-send. Linking an old diagnosis without re-analysis needs a backend endpoint like PATCH /api/diagnoses/:id/sector.');
      }

      diagnosisHistoryRuntimeCache.time = 0;
      diagnosisHistoryRuntimeCache.value = null;
      diagnosisHistoryRuntimeCache.promise = null;
      callBackendSavedEvent();
      addNotification(t('linkedToSector'),'success');
    }catch(err){
      addNotification(userFriendlyServiceError(err, lang) || (lang==='ar'?'تعذر ربط التشخيص بالقطاع من الباك إند.':'Could not link this diagnosis to the sector from the backend.'),'danger');
    }
  };
  const modelAlerts = [
    ...(Array.isArray(result.model_alerts) ? result.model_alerts : []),
    ...(isDanger(result.final_status) ? [t('alarmAlert')] : [])
  ].filter(Boolean);

  return <section className="diagnosis-center-result pro-result">
    <div className={`diagnosis-center-hero ${statusTone}`}>
      <div className="diagnosis-center-hero-copy">
        <span className="diagnosis-center-kicker">{ar?'نتيجة النموذج الذكي':'AI model result'}</span>
        <h2>{labelStatus(resultRecord.final_status,t)}</h2>
        <p>{displayText(headline,t)}</p>
        <div className="diagnosis-center-meta">
          <span>{ar?'نوع التحليل':'Analysis'}: <b>{sourceText}</b></span>
          <span>{ar?'الثقة':'Confidence'}: <b>{confidence}%</b></span>
          <span>{ar?'المرض / المشكلة':'Disease / Issue'}: <b>{disease}</b></span>
          {displayPlantType(resultRecord,result,t) && <span>{ar?'نوع النبات':'Plant type'}: <b>{displayPlantType(resultRecord,result,t)}</b></span>}
        </div>
      </div>
      <div className="diagnosis-center-confidence">
        <b>{confidence}%</b>
        <small>{ar?'نسبة الثقة':'Confidence'}</small>
        <em><i style={{width:`${confidence}%`}}/></em>
      </div>
    </div>

    {modelAlerts.length ? <div className={`model-alert-strip ${statusTone}`}><Bell size={20}/><div><b>{t('modelNotificationBar')}</b><span>{modelAlerts.map(a=>displayText(a,t)).join(' • ')}</span></div></div> : null}

    <div className="diagnosis-center-layout">
      <aside className="diagnosis-center-left">
        {resultRecord.imageUrl ? <div className="diagnosis-center-image-card">
          <img src={resultRecord.imageUrl} alt="Plant diagnosis result"/>
          {(result.suspicious_regions||[]).map((r,i)=><span key={i} className="suspect-box" style={{left:`${r.x||r.left||20}%`,top:`${r.y||r.top||20}%`,width:`${r.w||r.width||25}%`,height:`${r.h||r.height||20}%`}}>{r.label||t('possibleDisease')}</span>)}
        </div> : <div className="diagnosis-center-sensor-card"><Cpu size={48}/><b>{ar?'تشخيص اعتمادًا على القراءات':'Sensor-based diagnosis'}</b><small>{ar?'لم يتم رفع صورة في هذا التحليل.':'No image was uploaded for this analysis.'}</small></div>}
        <div className="diagnosis-center-small-facts">
          <span><small>{ar?'اسم المرض':'Disease Name'}</small><b>{disease}</b></span>
          <span><small>{ar?'الحالة النهائية':'Final Status'}</small><b>{labelStatus(resultRecord.final_status,t)}</b></span>
          <span><small>{ar?'القطاع':'Sector'}</small><b>{localizeValue(resultRecord.sectorName,t)}</b></span>
          {displayPlantType(resultRecord,result,t) && <span><small>{ar?'نوع النبات':'Plant type'}</small><b>{displayPlantType(resultRecord,result,t)}</b></span>}
        </div>
      </aside>

      <main className="diagnosis-center-main">
        <section className="diagnosis-center-section primary-section">
          <div className="diagnosis-center-section-title"><Sparkles size={18}/><h3>{ar?'تشخيص حالة النبات':'Plant diagnosis explanation'}</h3></div>
          <p>{displayText(resultRecord.diagnosis || headline,t)}</p>
          <div className="diagnosis-center-explain-grid">
            <span><small>{ar?'شرح الحساسات':'Sensor explanation'}</small><b>{displayText(resultRecord.sensor_status,t)}</b></span>
            <span><small>{ar?'شرح الصورة':'Image explanation'}</small><b>{displayText(resultRecord.image_status,t)}</b></span>
          </div>
        </section>

        {rows.length ? <section className="diagnosis-center-section">
          <div className="diagnosis-center-section-title"><Cpu size={18}/><h3>{ar?'سجل القراءات المستخدمة':'Sensor readings used'}</h3></div>
          <div className="diagnosis-center-reading-grid">{rows.map(r=><span key={r.key}><small>{r.label}</small><b>{r.value}</b></span>)}</div>
        </section> : null}

        {metrics.length ? <section className="diagnosis-center-section">
          <div className="diagnosis-center-section-title"><BarChart3 size={18}/><h3>{ar?'تحليل الصورة':'Image analysis'}</h3></div>
          <div className="diagnosis-center-metrics">{metrics.map(m=><div key={m.key}><div><b>{m.value.toFixed(1)}%</b><span>{m.label}</span></div><em><i style={{width:`${m.value}%`}}/></em></div>)}</div>
        </section> : null}

        <section className="diagnosis-center-section">
          <div className="diagnosis-center-section-title"><Zap size={18}/><h3>{ar?'التوصيات':'Recommendations'}</h3></div>
          {recommendations.length ? <div className="diagnosis-center-recs">{recommendations.slice(0,5).map((item,i)=><article key={i}><b>{ar?`توصية ${i+1}`:`Recommendation ${i+1}`}</b><p>{displayText(item,t)}</p></article>)}</div> : <p className="empty-copy">{ar?'لا توجد توصيات راجعة من الخدمة لهذا التشخيص.':'No recommendations were returned by the service for this diagnosis.'}</p>}
        </section>

        <section className="diagnosis-center-section">
          <div className="diagnosis-center-section-title"><CheckCircle2 size={18}/><h3>{ar?'الإجراءات المطلوبة':'Required actions'}</h3></div>
          {actions.length ? <ul className="diagnosis-center-actions">{actions.map((item,i)=><li key={i}>{displayText(item,t)}</li>)}</ul> : <p className="empty-copy">{ar?'لا توجد إجراءات راجعة من الخدمة لهذا التشخيص.':'No actions were returned by the service for this diagnosis.'}</p>}
        </section>

        {tips.length ? <section className="diagnosis-center-section tips-section">
          <div className="diagnosis-center-section-title"><Camera size={18}/><h3>{ar?'نصائح لتحسين نتيجة الصورة':'Capture tips for better results'}</h3></div>
          <ul className="diagnosis-center-actions">{tips.map((item,i)=><li key={i}>{displayText(item,t)}</li>)}</ul>
        </section> : null}
      </main>
    </div>

    <footer className="diagnosis-center-actions-bar">
      <button type="button" className="primary-btn" disabled={savingHistory} onClick={async()=>{ setSavingHistory(true); const saved=await saveDiagnosisResult(result,t); setSavingHistory(false); if(saved){ try{ window.dispatchEvent(new Event('ecosense-diagnoses-updated')); }catch{} const sid=result.__inputForm?.sectorId || result.input_data?.sectorId || ''; nav(sid ? `/my-diagnoses?sectorId=${encodeURIComponent(sid)}` : '/my-diagnoses'); } }}><CheckCircle2 size={18}/>{savingHistory ? (lang==='ar'?'جاري الحفظ...':'Saving...') : t('saveDiagnosis')}</button>
      <button type="button" className="secondary-btn" onClick={()=>downloadDiagnosisReport(result,t)}><FileText size={18}/>{t('downloadPdf')}</button>
      <select value={sectorId} onChange={e=>setSectorId(e.target.value)}><option value="">{lang==='ar'?'بدون قطاع - عرض النتيجة فقط':'No sector - show result only'}</option>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}</option>)}</select>
      <button type="button" className="secondary-btn" onClick={linkToSector}><Layers3 size={18}/>{t('linkDiagnosisToSector')}</button>
    </footer>
  </section>
}

function getLikelyCause(result,t){
  if(isDanger(result.final_status)) return t('riskReasonCritical');
  if(String(result.final_status).includes('Moderate')) return t('riskReasonWaterHeat');
  return t('riskReasonStable');
}
async function saveDiagnosisResult(result,t){
  const form = result.__inputForm || result.input_data || result.sensorReadings || {};
  const saved = await saveDiagnosisToService(result, form, result.__diagnosisMode || result.diagnosis_mode || 'combined', result.image_preview || result.imageUrl || '');
  const ar = document.documentElement.lang === 'ar';
  if(saved?.ok){
    // Backend endpoints are the source of truth. Do not create local diagnosis history.
    addNotification(ar ? 'تم حفظ التشخيص في السجل بنجاح' : 'Diagnosis saved to history successfully', 'success');
    return true;
  }
  if (saved?.modelOnly) {
    addNotification(ar
      ? 'تم عرض نتيجة الموديل. للحفظ الدائم في السجل اختر قطاعًا واستخدم قراءة محفوظة من الحساسات أو اطلب endpoint حفظ من الباك إند للتشخيص اليدوي.'
      : 'The model result is displayed. For permanent history, choose a sector and use a saved sensor reading, or add a backend save endpoint for manual diagnoses.', 'warning');
    return false;
  }
  if (saved?.backendAccepted && saved?.verified === false) {
    addNotification(ar
      ? 'تعذر ظهور التشخيص في سجل القطاع بعد الحفظ. حدّث صفحة تشخيصاتي أو حاول مرة أخرى.'
      : 'The diagnosis did not appear in the sector history after saving. Refresh My Diagnoses or try again.', 'danger');
    return false;
  }
  const details = saved?.error?.status ? ` (${saved.error.status})` : '';
  addNotification(ar
    ? `تعذر حفظ التشخيص في السجل${details}. حاول مرة أخرى بعد قليل.`
    : `Could not save this diagnosis to history${details}. Please try again shortly.`, 'danger');
  return false;
}

function createTreatmentTaskFromDiagnosis(result,t){
  const workers=workerAccounts();
  const worker=workers[0] || {};
  const task={id:Date.now(),title:result.final_status==='High Stress'?'Inspect leaves':'Check irrigation system',details:result.diagnosis || '',status:'pending',priority:isDanger(result.final_status)?'High':'Medium',assignedTo:worker.name || 'Plant Care Worker',assignedToEmail:worker.email || worker.username || '',createdAt:new Date().toLocaleString(),dueDate:new Date(Date.now()+86400000).toISOString().slice(0,10),sector:'Auto'};
  const tasks=getGlobalTasks();
  setGlobalTasks([task,...tasks]);
  if(task.assignedToEmail) addUserNotification(task.assignedToEmail, t('treatmentTaskCreated'), 'success');
  addNotification(t('treatmentTaskCreated'),'success');
}
function downloadDiagnosisReport(result,t){
  const input=result.input_data || {};
  const now=new Date();
  const row={
    id:Date.now(),
    date:now.toISOString().slice(0,10),
    time:now.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),
    sector: input.sectorName || input.sector || input.sectorId || '-',
    cropType:result.cropType || result.crop_type || result.plantType || result.plant_type || '-',
    final_status:result.final_status,
    confidence:result.confidence,
    disease_name:result.disease_name,
    analysisType:result.diagnosis_mode || input.analysisType || (result.image_preview ? 'Image + Sensors' : 'Sensors Only'),
    temperature:input.temperature,
    humidity:input.humidity,
    soilMoisture:input.soilMoisture,
    soilTemp:input.soilTemp,
    light:input.light,
    image:result.image_preview || result.imageUrl || result.image || '',
    image_status:result.image_status,
    diagnosis:result.diagnosis,
    recommendations:result.recommendations || [],
    actions:result.actions || [],
    workerTask:''
  };
  openProfessionalReport({ t, lang: (document.documentElement.dir==='rtl'?'ar':'en'), rows:[row], titleKey:'diagnosisReport' });
}
function formatInputValue(k,v){ if(v==null || v==='') return '—'; if(['temperature','soilTemp'].includes(k)) return `${v}°C`; if(['humidity','soilMoisture'].includes(k)) return `${v}%`; return v; }

function isArabicUI(t){ return t('name') === 'العربية' || document.documentElement.dir === 'rtl'; }
function firstUsefulText(...values){
  for(const v of values){
    if(v == null) continue;
    if(typeof v === 'string' && v.trim() && v.trim() !== '-' && v.trim() !== '—') return v.trim();
    if(typeof v === 'number') return String(v);
  }
  return '';
}
function percentFromAny(value){
  if(value == null || value === '') return null;
  const n = Number(value);
  if(!Number.isFinite(n)) return null;
  return Math.max(0, Math.min(100, n <= 1 ? n * 100 : n));
}
function readingRows(record={}, t){
  const input = record.input_data || record.sensorReadings || {};
  const rows = [
    ['temperature', t('temperature') || 'Temperature', input.temperature ?? input.temp],
    ['humidity', t('humidity') || 'Humidity', input.humidity ?? input.hum],
    ['soilMoisture', t('soilMoisture') || 'Soil Moisture', input.soilMoisture ?? input.Soil ?? input.soil],
    ['soilTemp', t('soilTemp') || 'Soil Temperature', input.soilTemp ?? input.soilTemperature],
    ['light', t('light') || 'Light', input.light ?? input.lightLevel]
  ];
  return rows.map(([key,label,value]) => ({key,label,value: formatInputValue(key, value)})).filter(r => r.value !== '—');
}
function imageMetricRows(record={}, t){
  const raw = record.raw || {};
  const a = record.image_analysis || raw.image_analysis || raw.imageAnalysis || raw.analysis || raw.result || {};
  const candidates = [
    ['green_ratio', t('greenRatio') || 'Green tissue', a.green_ratio ?? a.greenRatio ?? raw.green_ratio ?? raw.greenRatio],
    ['yellow_ratio', t('yellowRatio') || 'Yellowing', a.yellow_ratio ?? a.yellowRatio ?? raw.yellow_ratio ?? raw.yellowRatio],
    ['brown_ratio', t('brownRatio') || 'Brown tissue', a.brown_ratio ?? a.brownRatio ?? raw.brown_ratio ?? raw.brownRatio],
    ['dark_spot_ratio', t('darkSpotRatio') || 'Dark spots', a.dark_spot_ratio ?? a.darkSpotRatio ?? raw.dark_spot_ratio ?? raw.darkSpotRatio],
    ['damaged_ratio', t('damagedRatio') || 'Visible damage', a.damaged_ratio ?? a.damagedRatio ?? raw.damaged_ratio ?? raw.damagedRatio],
    ['health_score', t('healthScore') || 'Health score', a.health_score ?? a.healthScore ?? raw.health_score ?? raw.healthScore]
  ];
  return candidates.map(([key,label,value])=>({key,label,value:percentFromAny(value)})).filter(r=>r.value!=null);
}
function diagnosisHeadline(record={}, t){
  const ar = isArabicUI(t);
  const status = String(record.final_status || '').toLowerCase();
  const disease = String(record.disease_name || '').toLowerCase();
  const imageStatus = String(record.image_status || '').toLowerCase();
  const sensorStatus = String(record.sensor_status || '').toLowerCase();
  const hasFungal = disease.includes('leaf spot') || disease.includes('fung') || disease.includes('spot') || disease.includes('تبقع') || disease.includes('فطر') || imageStatus.includes('spot');
  const hasChlorosis = disease.includes('chlorosis') || disease.includes('yellow') || disease.includes('اصفرار');
  if(ar){
    if(status.includes('high') || status.includes('critical') || status.includes('مرتفع') || hasFungal) return hasFungal ? 'اشتباه تبقع أوراق أو إصابة فطرية' : 'إجهاد مرتفع يحتاج تدخل سريع';
    if(hasChlorosis) return 'اشتباه اصفرار أو نقص عناصر غذائية';
    if(status.includes('moderate') || status.includes('متوسط')) return 'إجهاد متوسط يحتاج متابعة وضبط للظروف';
    if(status.includes('healthy') || status.includes('سليم')) return 'النبات يبدو سليمًا ولا توجد مؤشرات خطر واضحة';
    return firstUsefulText(record.disease_name, record.diagnosis, sensorStatus) || 'تشخيص النموذج الذكي';
  }
  if(status.includes('high') || status.includes('critical') || hasFungal) return hasFungal ? 'Suspected leaf spot or fungal infection' : 'High stress requiring urgent action';
  if(hasChlorosis) return 'Possible chlorosis or nutrient deficiency';
  if(status.includes('moderate')) return 'Moderate stress needs follow-up';
  if(status.includes('healthy')) return 'Plant looks healthy with no clear risk indicators';
  return firstUsefulText(record.disease_name, record.diagnosis, sensorStatus) || 'AI diagnosis summary';
}
function cleanListItems(value){
  if(!value) return [];
  if(Array.isArray(value)) return value.flatMap(cleanListItems).filter(Boolean);
  if(typeof value === 'object') return [value.title || value.details || value.message || value.text || value.name].filter(Boolean);
  return [String(value)].filter(Boolean);
}
function treatmentItems(record={}, t){
  const ar = isArabicUI(t);
  const fromBackend = [...cleanListItems(record.recommendations), ...cleanListItems(record.raw?.image_recommendations), ...cleanListItems(record.raw?.recommendations_ar)].filter(Boolean);
  if(fromBackend.length) return [...new Set(fromBackend)].slice(0,6);
  const st = String(record.final_status || '').toLowerCase();
  const disease = String(record.disease_name || '').toLowerCase();
  const input = record.input_data || {};
  const items=[];
  const fungal = disease.includes('leaf spot') || disease.includes('fung') || disease.includes('spot') || disease.includes('تبقع') || disease.includes('فطر');
  if(ar){
    if(fungal){
      items.push('اعزل الأوراق المصابة أو راقبها بعناية لتقليل انتشار العدوى.');
      items.push('أزل الأوراق شديدة الإصابة التي تحتوي على بقع كثيرة أو تلف شديد.');
      items.push('حسّن حركة الهواء حول النبات وقلل الرطوبة الزائدة.');
      items.push('تجنب رش الماء على الأوراق واسقِ النبات عند التربة فقط.');
      items.push('استخدم معاملة فطرية مناسبة حسب نوع النبات أو استشر مختصًا زراعيًا.');
    } else if(st.includes('moderate') || st.includes('متوسط')){
      items.push('راجع رطوبة التربة واضبط الري تدريجيًا بدون إغراق.');
      items.push('تابع درجة الحرارة والإضاءة خلال اليومين القادمين.');
      items.push('افحص الأوراق بصريًا للتأكد من عدم ظهور بقع أو اصفرار جديد.');
    } else if(st.includes('high') || st.includes('critical') || st.includes('مرتفع')){
      items.push('تعامل مع الحالة كإنذار عاجل وافحص القطاع اليوم.');
      items.push('اضبط الري أو التهوية حسب القراءات غير الطبيعية.');
      items.push('أنشئ مهمة للعامل لمتابعة النبات وإرسال صورة جديدة بعد العلاج.');
    } else {
      items.push('استمر في المتابعة الدورية وسجل قراءة جديدة عند تغير حالة النبات.');
      items.push('حافظ على الري والإضاءة ضمن الحدود المناسبة للمحصول.');
    }
    if(Number(input.soilMoisture) < 30) items.push('رطوبة التربة منخفضة؛ زوّد الري تدريجيًا ثم أعد القياس.');
    if(Number(input.temperature) > 34) items.push('درجة الحرارة مرتفعة؛ فعّل التهوية أو التبريد إن وجد.');
    return [...new Set(items)].slice(0,6);
  }
  if(fungal){
    items.push('Isolate or closely monitor affected leaves to reduce disease spread.');
    items.push('Remove severely damaged leaves with many spots.');
    items.push('Improve airflow around the plant and reduce excess humidity.');
    items.push('Avoid spraying water on leaves; irrigate near the soil.');
    items.push('Use a suitable fungicide treatment or consult an agronomist.');
  } else if(st.includes('moderate')){
    items.push('Review soil moisture and adjust irrigation gradually.');
    items.push('Track temperature and light over the next two days.');
    items.push('Inspect leaves for new spots or yellowing.');
  } else if(st.includes('high') || st.includes('critical')){
    items.push('Treat this as an urgent warning and inspect the sector today.');
    items.push('Adjust irrigation or ventilation based on abnormal readings.');
    items.push('Create a worker task and capture a new image after treatment.');
  } else {
    items.push('Continue regular monitoring and save a new reading if plant condition changes.');
    items.push('Keep irrigation and light within the crop ideal range.');
  }
  return [...new Set(items)].slice(0,6);
}
function actionItems(record={}, t){
  const fromBackend = cleanListItems(record.actions);
  if(fromBackend.length) return fromBackend.slice(0,5);
  const ar=isArabicUI(t);
  const rows=readingRows(record,t);
  const input=record.input_data || {};
  const items=[];
  if(Number(input.soilMoisture) < 30) items.push(ar?'زيادة الري تدريجيًا ثم إعادة القياس.':'Increase irrigation gradually, then measure again.');
  if(Number(input.temperature) > 34) items.push(ar?'تشغيل التهوية أو التبريد لتقليل الإجهاد الحراري.':'Enable ventilation or cooling to reduce heat stress.');
  if(!items.length && rows.length) items.push(ar?'متابعة القراءات وتسجيل قراءة جديدة بعد 24 ساعة.':'Monitor readings and save a new reading after 24 hours.');
  return items.length ? items : [ar?'لا يوجد إجراء عاجل، استمر في المتابعة.':'No urgent action; continue monitoring.'];
}
function captureTips(record={}, t){
  const ar=isArabicUI(t);
  const raw = record.raw || {};
  const fromBackend = cleanListItems(raw.capture_tips || raw.captureTips || raw.image_capture_tips || raw.imageCaptureTips);
  if(fromBackend.length) return fromBackend.slice(0,5);
  if(!record.imageUrl) return [];
  return ar ? [
    'صوّر النبات في إضاءة طبيعية واضحة بدون ظلال قوية.',
    'خلّي الخلفية بسيطة وفاتحة قدر الإمكان.',
    'قرّب الورقة أو المنطقة المصابة من الكاميرا بدون اهتزاز.',
    'تجنب الظلال القوية لأنها قد تظهر كأنها بقع مرضية.',
    'ارفع صورة مع قراءات الحرارة والرطوبة ورطوبة التربة للحصول على نتيجة أدق.'
  ] : [
    'Capture the plant in clear natural light without strong shadows.',
    'Keep the background simple and bright when possible.',
    'Move the affected leaf close to the camera without blur.',
    'Avoid heavy shadows because they may look like disease spots.',
    'Combine the image with sensor readings for a more accurate result.'
  ];
}
function modelNote(record={}, t){
  const ar=isArabicUI(t);
  const hasImage=!!record.imageUrl;
  const rows=readingRows(record,t);
  if(hasImage && !rows.length) return ar ? 'هذه نتيجة تقديرية من الصورة فقط. يفضّل دمجها مع قراءات الحرارة والرطوبة ورطوبة التربة والإضاءة للحصول على قرار أدق.' : 'This is an image-only estimate. Combine it with temperature, humidity, soil moisture, and light readings for a stronger decision.';
  if(!hasImage && rows.length) return ar ? 'هذا التشخيص مبني على قراءات الحساسات. أضف صورة للنبات عند وجود أعراض على الأوراق لتحسين دقة التشخيص.' : 'This diagnosis is based on sensor readings. Add a plant image when leaf symptoms are visible for better accuracy.';
  return ar ? 'يعرض هذا الملخص نتيجة التشخيص النهائية مع أهم المؤشرات والتوصيات.' : 'This summary shows the final diagnosis with the key indicators and recommendations.';
}
function Bars({data}){ const {t}=useLang(); const keys=[['green_ratio','greenRatio'],['yellow_ratio','yellowRatio'],['brown_ratio','brownRatio'],['dark_spot_ratio','darkSpotRatio'],['damaged_ratio','damagedRatio']]; return <div className="bars">{keys.filter(([k])=>data[k]!=null).map(([k,l])=>{ const v=Math.round(Number(data[k])*100); return <div key={k}><span>{t(l)}</span><em><i style={{width:`${v}%`}}/></em><b>{v}%</b></div> })}</div> }
function Sensors(){ 
  const {t,lang}=useLang(); 
  const [readings,setReadings]=useState(getStore('sph_readings',[])); 
  const [device,setDevice]=useState({deviceId:'SPH-DEVICE-001',interval:'5'});
  const [form,setForm]=useState({cropType:'',temperature:'',humidity:'',soilMoisture:'',soilTemp:'',light:''});
  const [result,setResult]=useState(null);
  const [historyOpen,setHistoryOpen]=useState(false);
  const [serviceState,setServiceState]=useState({loading:false,error:'',live:false,analytics:null});
  const loadSensorService=async()=>{
    setServiceState(s=>({...s,loading:true,error:''}));
    try{
      const [latestRes,historyRes,analyticsRes]=await Promise.allSettled([sensorsAPI.getLatestHardware(),sensorsAPI.getHardwareHistory({limit:30}),sensorsAPI.getAnalytics()]);
      if(latestRes.status==='fulfilled'){
        const {reading}=normalizeServiceSensorPayload(latestRes.value.data || {});
        setForm({cropType:reading.cropType,temperature:reading.temperature,humidity:reading.humidity,soilMoisture:reading.soilMoisture,soilTemp:reading.soilTemp,light:reading.light});
      }
      if(historyRes.status==='fulfilled'){
        const list=asArray(historyRes.value.data,['history','readings','items']);
        if(list.length){
          const mapped=list.map((item)=>normalizeServiceSensorPayload(item).reading);
          setReadings(mapped); setStore('sph_readings',mapped);
        }
      }
      const analytics=analyticsRes.status==='fulfilled' ? (analyticsRes.value.data?.data || analyticsRes.value.data) : null;
      setServiceState({loading:false,error:'',live:latestRes.status==='fulfilled' || historyRes.status==='fulfilled',analytics});
    }catch(err){ setServiceState({loading:false,error:serviceFriendlyError(err,lang),live:false,analytics:null}); }
  };
  useEffect(()=>{ loadSensorService(); },[lang]);
  const change=(k,v)=>setForm({...form,[k]:v});
  const sendSensors=async()=>{ 
    let normalized;
    try{
      const sectorId = form.sectorId || form.sector_id || '';
      if(!sectorId) throw new Error(lang==='ar'?'اختر قطاعًا حقيقيًا من الباك إند قبل حفظ تحليل الحساسات.':'Choose a real backend sector before saving sensor analysis.');
      const response = await sensorsAPI.analyze(sectorId, sensorPayloadForBackend(form));
      normalized=normalize(response.data || response, form, false);
      normalized.__backendSaved=true;
      normalized.__backendEndpoint='POST /sensors/analyze/:sectorId';
      normalized.__historyEndpoint='GET /sensors/history?sectorId=SECTOR_ID';
    }catch(err){
      addNotification(userFriendlyServiceError(err,lang),'danger');
      return;
    }
    setResult(normalized);
    const r={...form,id:Date.now(),time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),final_status:normalized.final_status,confidence:normalized.confidence,solution:normalized.recommendations?.[0]};
    const all=[r,...readings].slice(0,30);
    setReadings(all); setStore('sph_readings',all);
    triggerModelAlerts(normalized,t);
  };
  const simulate=()=>{
    const next={cropType:'',temperature:Math.round(23+Math.random()*17),humidity:Math.round(25+Math.random()*45),soilMoisture:Math.round(16+Math.random()*45),soilTemp:Math.round(22+Math.random()*14),light:['Low','Medium','Sufficient'][Math.floor(Math.random()*3)]};
    setForm(next);
  };
  return <>
    <PageHead title={t('connectedSensors')} sub={t('sensorUserHint')} action={<div className="report-actions"><button className="secondary-btn" onClick={loadSensorService} disabled={serviceState.loading}><Activity size={18}/>{t('refreshReadings')}</button><button className="primary-btn" onClick={sendSensors}><ScanSearch size={18}/>{t('sendToModel')}</button></div>}/>
    {serviceState.error && <p className="form-helper warning-text">{serviceState.error}</p>}
    {serviceState.live && <p className="form-helper success-text">{lang==='ar'?'تم تحميل قراءات الحساسات.':'Sensor readings loaded successfully.'}</p>}
    <div className="grid-2">
      <Panel title={t('connectSensors')}>
        <div className="form-grid">
          <Field label={t('sensorDeviceId')}><input value={device.deviceId} onChange={e=>setDevice({...device,deviceId:e.target.value})}/></Field>
          <Field label={t('readingInterval')}><select value={device.interval} onChange={e=>setDevice({...device,interval:e.target.value})}><option value="5">{t('everyFiveMinutes')}</option></select></Field>
        </div>
        <div className="sensor-connect-card"><Cpu/><div><b>{device.deviceId}</b><span>{t('lastSync')}: {new Date().toLocaleTimeString()}</span></div></div>
      </Panel>
      <Panel title={t('manualSensorInput')}>
        <div className="form-grid">
          <Field label={t('cropType')}><CropTypeSelect lang={lang} value={form.cropType || ''} onChange={value=>change('cropType', value)} /></Field>
          <Field label={t('temperature')}><input type="number" value={form.temperature} onChange={e=>change('temperature',Number(e.target.value))}/></Field>
          <Field label={t('humidity')}><input type="number" value={form.humidity} onChange={e=>change('humidity',Number(e.target.value))}/></Field>
          <Field label={t('soilMoisture')}><input type="number" value={form.soilMoisture} onChange={e=>change('soilMoisture',Number(e.target.value))}/></Field>
          <Field label={t('soilTemp')}><input type="number" value={form.soilTemp} onChange={e=>change('soilTemp',Number(e.target.value))}/></Field>
          <Field label={t('light')}><select value={form.light} onChange={e=>change('light',e.target.value)}><option>Low</option><option>Medium</option><option>Sufficient</option></select></Field>
        </div>
        <button className="secondary-btn wide" onClick={simulate}><Activity size={18}/>{t('simulateFiveMinutes')}</button>
      </Panel>
    </div>
    {result && <Result result={{...result,diagnosis_mode:'sensors'}}/>}
    <section className="sensor-history-cta panel">
      <div>
        <h3>{t('sensorHistory')}</h3>
        <p>{t('readingsHistorySub')}</p>
      </div>
      <button className="secondary-btn" onClick={()=>setHistoryOpen(true)}><BarChart3 size={18}/>{t('viewSensorHistory')}</button>
    </section>
    {historyOpen && <SensorHistoryModal readings={readings} onClose={()=>setHistoryOpen(false)} />}
  </>
}

function SensorHistoryModal({readings,onClose}){
  const {t}=useLang();
  const sectors=getStore('sph_sectors',[]);
  const [sector,setSector]=useState('all');
  const [sensor,setSensor]=useState('all');
  const sensorOptions=[['temperature',t('temperature')],['humidity',t('humidity')],['soilMoisture',t('soilMoisture')],['soilTemp',t('soilTemp')],['light',t('light')]];
  const rows=readings.map((r,i)=>({
    ...r,
    dateTime: r.dateTime || `${new Date().toLocaleDateString()} ${r.time || ''}`,
    sector: r.sector || sectors[i % Math.max(sectors.length,1)]?.name || 'Greenhouse A'
  })).filter(r=>sector==='all' || String(r.sector)===String(sector));
  const chartSensor = sensor==='all' ? 'soilMoisture' : sensor;
  return <div className="modal-backdrop sensor-history-modal" role="dialog" aria-modal="true">
    <section className="modal-card sensor-history-card">
      <div className="modal-head">
        <div><h2>{t('sensorHistory')}</h2><p>{t('readingsHistorySub')}</p></div>
        <button className="icon-btn" onClick={onClose} aria-label={t('close')}><X size={18}/></button>
      </div>
      <div className="history-filters">
        <Field label={t('filterBySector')}><select value={sector} onChange={e=>setSector(e.target.value)}><option value="all">{t('allSectors')}</option>{sectors.map(s=><option key={s.id} value={s.name}>{localizeValue(s.name,t)}</option>)}</select></Field>
        <Field label={t('filterBySensor')}><select value={sensor} onChange={e=>setSensor(e.target.value)}><option value="all">{t('allSensors')}</option>{sensorOptions.map(([k,l])=><option key={k} value={k}>{l}</option>)}</select></Field>
      </div>
      <div className="history-chart-wrap">
        <ResponsiveContainer width="100%" height={230}>
          <LineChart data={rows.slice().reverse()}><CartesianGrid vertical={false} stroke="var(--line)"/><XAxis dataKey="time"/><YAxis/><Tooltip/><Line dataKey={chartSensor} stroke="var(--primary)" strokeWidth={3} dot={{r:3}}/></LineChart>
        </ResponsiveContainer>
      </div>
      <div className="history-cards-list">
        {rows.map(r=><article key={r.id} className="history-reading-card">
          <div><b>{localizeValue(r.sector,t)}</b><span>{t('dateTime')}: {r.dateTime}</span></div>
          <div className="history-reading-values">
            {(sensor==='all' ? sensorOptions : sensorOptions.filter(([k])=>k===sensor)).map(([k,l])=><span key={k}><small>{l}</small><strong>{formatInputValue(k,r[k])}</strong></span>)}
          </div>
          <em className={`tag ${tone(r.final_status)}`}>{labelStatus(r.final_status,t)}</em>
        </article>)}
      </div>
    </section>
  </div>
}

function DataTable({readings}){ const {t}=useLang(); return <div className="table-wrap"><table><thead><tr><th>{t('time')}</th><th>{t('cropType')}</th><th>{t('temperature')}</th><th>{t('humidity')}</th><th>{t('soilMoisture')}</th><th>{t('condition')}</th></tr></thead><tbody>{readings.map(r=><tr key={r.id}><td>{r.time}</td><td>{localCropLabel(r.cropType, langFromDocument())}</td><td>{r.temperature}°C</td><td>{r.humidity}%</td><td>{r.soilMoisture}%</td><td><span className={`tag ${tone(r.final_status)}`}>{labelStatus(r.final_status,t)}</span></td></tr>)}</tbody></table></div> }
function Sectors({embedded=false}){ 
  const {t,lang}=useLang(); 
  const [items,setItems]=useState(getStore('sph_sectors',[])); 
  const [serviceMsg,setServiceMsg]=useState('');
  const [form,setForm]=useState({name:'',location:'',crop:''});
  const [workerForm,setWorkerForm]=useState({name:'',role:'Plant Care Worker',phone:'',sectorId:''}); 
  useEffect(()=>{
    let mounted=true;
    sectorsAPI.getAll().then(({data})=>{
      const list=asArray(data,['sectors','items']);
      if(!mounted || !list.length) return;
      const mapped=list.map((item,idx)=>({
        id:item._id || item.id || idx+1,
        name:item.name || item.sectorName || `Sector ${idx+1}`,
        location:item.location || item.area || '-',
        crop:item.crop || item.cropType || item.plantType || '',
        status:item.status || item.final_status || item.healthStatus || '',
        sensors:item.sensors || {},
        diagnosis:item.diagnosis || {condition:item.status || '', disease:item.disease || '', confidence:item.confidence || 0, summary:item.summary || ''},
        workers:item.workers || [], tasks:item.tasks || [], notes:item.notes || [], rows:item.rows || [], equipment:item.equipment || [], devices:item.devices || []
      }));
      setItems(mapped); setStore('sph_sectors',mapped); setServiceMsg(lang==='ar'?'تم تحميل القطاعات بنجاح.':'Sectors loaded successfully.');
    }).catch(err=>setServiceMsg(serviceFriendlyError(err,lang)));
    return()=>{mounted=false};
  },[lang]);
  const save=async()=>{ 
    if(!form.name.trim()) return; 
    try{ const {data}=await sectorsAPI.create({name:form.name,location:form.location,...(form.crop ? {crop:form.crop,cropType:form.crop} : {})}); const saved=data?.sector || data?.data || data; const item={...saved,id:saved?._id || saved?.id || Date.now(),name:saved?.name || form.name,location:saved?.location || form.location,crop:saved?.crop || saved?.cropType || form.crop,workers:saved?.workers || [],tasks:saved?.tasks || [],notes:saved?.notes || [],rows:saved?.rows || [],equipment:saved?.equipment || [],devices:saved?.devices || []}; const all=[item,...items]; setItems(all); setStore('sph_sectors',all); setForm({name:'',location:'',crop:''}); setServiceMsg(lang==='ar'?'تم حفظ القطاع بنجاح.':'Sector saved successfully.'); }catch(err){ setServiceMsg(serviceFriendlyError(err,lang)); }
  }; 
  const del=async(id)=>{ if(!hasPermission('deleteSector')) return addNotification(t('ownerSensitiveBlocked'),'warning'); try{ await sectorsAPI.delete(id); }catch(err){ setServiceMsg(serviceFriendlyError(err,lang)); } const all=items.filter(x=>String(x.id)!==String(id)); setItems(all); setStore('sph_sectors',all); }; 
  const healthy=items.filter(s=>s.status==='Healthy').length;
  const warning=items.filter(s=>String(s.status).includes('Moderate')).length;
  const danger=items.filter(s=>isDanger(s.status)).length;
  return <>
    {!embedded && <PageHead title={t('sectorsAsService')} sub={t('sectorsAsServiceSub')} action={<span className="mode-pill">{serviceMsg || t('linkedServiceFeature')}</span>}/>}
    {serviceMsg && <p className="form-helper">{serviceMsg}</p>}
    <div className="kpis sector-kpis">
      <Kpi icon={<Layers3/>} label={t('totalSectors')} value={items.length}/>
      <Kpi icon={<Leaf/>} label={t('healthySectors')} value={healthy} tone="green"/>
      <Kpi icon={<AlertTriangle/>} label={t('warningSectors')} value={warning} tone="amber"/>
      <Kpi icon={<Bell/>} label={t('dangerousSectors')} value={danger} tone="red"/>
    </div>
    <div className="grid-2">
      <Panel title={t('addSector')}>
        <div className="form-grid">
          <Field label={t('sectorName')}><input value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></Field>
          <Field label={t('location')}><input value={form.location} onChange={e=>setForm({...form,location:e.target.value})}/></Field>
          <Field label={t('crop')}><CropTypeSelect lang={langFromDocument()} value={form.crop || ''} onChange={value=>setForm({...form,crop:value})} /></Field>
        </div>
        <button className="primary-btn" onClick={save}><Plus size={18}/>{t('save')}</button>
      </Panel>
      <Panel title={t('sectorStats')}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={items.map(s=>({name:s.name, moisture:s.sensors?.soilMoisture||0, temp:s.sensors?.temperature||0}))}>
            <CartesianGrid vertical={false} stroke="var(--line)"/><XAxis dataKey="name"/><YAxis/><Tooltip/>
            <Bar dataKey="moisture" fill="var(--primary)" radius={[12,12,0,0]}/>
            <Bar dataKey="temp" fill="var(--amber)" radius={[12,12,0,0]}/>
          </BarChart>
        </ResponsiveContainer>
      </Panel>
    </div>
    <div className="sector-grid farm">
      {items.map(s=><article className={`sector-card ${tone(s.status)}`} key={s.id}>
        <div className="sector-head"><Layers3/><span className={`tag ${tone(s.status)}`}>{labelStatus(s.status,t)}</span></div>
        <h3>{localizeValue(s.name,t)}</h3><p>{localizeValue(s.location,t)}</p><strong>{localizeValue(s.crop,t)}</strong>
        <div className="sector-mini">
          <Box label={t('temperature')} value={`${s.sensors?.temperature ?? '-'}°C`}/>
          <Box label={t('soilMoisture')} value={`${s.sensors?.soilMoisture ?? '-'}%`}/>
          <Box label={t('light')} value={s.sensors?.light ?? '-'}/>
          <Box label={t('confidence')} value={`${Math.round((s.diagnosis?.confidence ?? .8)*100)}%`}/>
        </div>
        <div className="sector-diagnosis">
          <b>{t('latestDiagnosis')}</b>
          <span>{translateModelText(s.diagnosis?.disease,t)}</span>
          <p>{translateModelText(s.diagnosis?.summary,t)}</p>
        </div>
        <div className="sector-card-counts">
          <span>{t('workers')}: {(s.workers||[]).length}</span>
          <span>{t('devices')}: {(s.devices||[]).length}</span>
          <span>{t('careTasks')}: {(s.tasks||[]).length}</span>
        </div>
        <div className="sector-actions"><NavLink className="secondary-btn" to={`/sectors/${s.id}`}>{t('enterSector')}</NavLink><button onClick={()=>del(s.id)}>{t('delete')}</button></div>
      </article>)}
    </div>
  </> 
}
function SectorDetails(){
  const {t}=useLang();
  const loc=useLocation();
  const id=loc.pathname.split('/').pop();
  const [sectors,setSectors]=useState(getStore('sph_sectors',[]));
  const sector=sectors.find(s=>String(s.id)===String(id)) || sectors[0];
  if(!sector){ return <><PageHead title={t('sectorDetails')} sub={langFromDocument()==='ar'?'لا توجد قطاعات متاحة حاليًا من الباك إند.':'No sectors are available from the backend right now.'} action={<NavLink className="secondary-btn" to="/farm-management">{t('backToSectors')}</NavLink>}/><section className="farm-empty-focus"><p>{langFromDocument()==='ar'?'لا توجد بيانات متاحة حاليًا':'No data is available right now'}</p></section></>; }
  const [sectorSensorState,setSectorSensorState]=useState({loading:false,error:'',latest:null,history:[],imageHistory:[]});
  const [serviceSectorDevices,setServiceSectorDevices]=useState([]);
  const [serviceWorkers,setServiceWorkers]=useState([]);
  const [sectorRefreshKey,setSectorRefreshKey]=useState(0);
  const [worker,setWorker]=useState({name:'',role:'',phone:''});
  const [workersOpen,setWorkersOpen]=useState(false);
  const [device,setDevice]=useState({name:'',type:'',value:'',unit:'',status:'Online'});
  const [equipmentOpen,setEquipmentOpen]=useState(false);
  const [sensorFormOpen,setSensorFormOpen]=useState(false);
  const [task,setTask]=useState({name:'',description:'',priority:'Medium',assignedToEmail:'',dueDate:''});
  const [note,setNote]=useState('');
  const [sectorAssistantOpen,setSectorAssistantOpen]=useState(false);
  const [selectedDiagnosis,setSelectedDiagnosis]=useState(null);
  const updateSector=(next)=>{
    const all=sectors.map(s=>s.id===sector.id?next:s);
    setSectors(all); setStore('sph_sectors',all);
  };
  useEffect(()=>{
    let mounted=true;
    const sectorId = sector?._id || sector?.id || sector?.sectorId;
    if(!sectorId) return()=>{mounted=false};
    setSectorSensorState(s=>({...s,loading:true,error:''}));
    Promise.allSettled([
      sensorsAPI.getLatestWithFallback(sectorId),
      sensorsAPI.getHistoryWithFallback({sectorId, limit:10}),
      imagesAPI.getHistoryWithFallback({sectorId, limit:10}),
      devicesAPI.getAll(),
      usersAPI.getWorkers(),
      sectorsAPI.getAll(),
    ]).then(([latestRes,historyRes,imageHistoryRes,devicesRes,workersRes,sectorsRes])=>{
      if(!mounted) return;
      const latest = latestRes.status === 'fulfilled' ? normalizeServiceSensorPayload(latestRes.value.data || {}).reading : null;
      const sensorHistoryRows = historyRes.status === 'fulfilled'
        ? diagnosisRecordsFromPayload(historyRes.value.data).map((item,index)=>normalizeSensorHistoryRecord({ ...item, sectorId, sectorName: sector.name, historySource:'sensors' }, index))
        : [];
      const imageHistoryRows = imageHistoryRes.status === 'fulfilled'
        ? imageHistoryRecordsFromPayload(imageHistoryRes.value.data).map((item,index)=>normalizeImageHistoryRecord({ ...item, historySource:'images' }, index))
        : [];
      const historyRows = [...sensorHistoryRows, ...imageHistoryRows]
        .sort((a,b)=> new Date(b.createdAt || b.date || 0) - new Date(a.createdAt || a.date || 0))
        .slice(0,10);

      const deviceRows = devicesRes.status === 'fulfilled'
        ? asArray(devicesRes.value.data, ['devices','items','rows','records']).map(normalizeDiagnosisDevice)
        : [];
      const devicesForSector = deviceRows.filter(d =>
        String(d.sectorId || '') === String(sectorId) ||
        String(d.sectorName || '') === String(sector.name || '')
      );
      setServiceSectorDevices(devicesForSector);

      const workerRows = workersRes.status === 'fulfilled'
        ? filterWorkersForCurrentOwner(asArray(workersRes.value.data, ['workers','users','items','rows','records'])).map((w,idx)=>({
            id: w._id || w.id || idx + 1,
            name: w.name || w.fullName || [w.firstName,w.lastName].filter(Boolean).join(' ') || w.email || `Worker ${idx+1}`,
            email: w.email || w.username || '',
            username: w.username || w.email || '',
            phone: w.phone || w.phoneNumber || '',
            role: normalizeRole(w.role || w.jobTitle, w.email),
            jobTitle: w.jobTitle || w.role || 'Plant Care Worker',
            status: w.status || (w.isActive === false ? 'Offline' : 'Online'),
            sectorId: w.sectorId || w.assignedSectorId || w.assignedSector?._id || w.sector?._id || w.sectorId,
            sector: w.sectorName || w.assignedSectorName || w.assignedSector?.name || w.sector?.name || w.sector || '',
            lastTask: w.lastTask || w.currentTask || '-',
          }))
        : [];
      setServiceWorkers(workerRows.filter(w => String(w.sectorId || '') === String(sectorId) || String(w.sector || '') === String(sector.name || '')));

      if (sectorsRes.status === 'fulfilled') {
        const sectorRows = asArray(sectorsRes.value.data, ['sectors','items','rows','records']).map((item,idx)=>({
          ...item,
          id: item._id || item.id || item.sectorId || idx + 1,
          name: item.name || item.sectorName || item.sector_name || `Sector ${idx+1}`,
          crop: item.crop || item.cropType || item.plantType || '',
          workers: item.workers || [],
          tasks: item.tasks || [],
          notes: item.notes || [],
          devices: item.devices || [],
        })).filter(x=>x.id);
        if (sectorRows.length) { setSectors(sectorRows); setStore('sph_sectors', sectorRows); }
      }

      setSectorSensorState({loading:false,error: latestRes.status==='rejected' && historyRes.status==='rejected' && imageHistoryRes.status==='rejected' ? serviceFriendlyError(latestRes.reason, langFromDocument()) : '',latest,history:historyRows,imageHistory:imageHistoryRows});
      if(latest){ updateSector({...sector, sensors:{...(sector.sensors||{}), ...latest}, devices: devicesForSector.length ? devicesForSector : (sector.devices || [])}); }
    }).catch(err=>{
      if(mounted) setSectorSensorState({loading:false,error:serviceFriendlyError(err, langFromDocument()),latest:null,history:[],imageHistory:[]});
    });
    return()=>{mounted=false};
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[sector?.id, sectorRefreshKey]);
  const addWorker=()=>{
    if(!worker.name.trim()) return;
    updateSector({...sector, workers:[...(sector.workers||[]),{...worker,id:Date.now()}]});
    setWorker({name:'',role:'',phone:''});
  };
  const addDevice=async()=>{
    if(!device.name.trim()) return;
    let saved={...device,id:Date.now(),sectorId:sector.id,sector:sector.name};
    try{ const {data}=await devicesAPI.create({name:device.name,type:device.type,value:device.value,unit:device.unit,status:device.status,sectorId:sector.id}); const payload=data?.device || data?.data || data; saved={...saved,...payload,id:payload?._id || payload?.id || saved.id}; }catch(err){ addNotification(serviceFriendlyError(err, document.documentElement.lang==='ar'?'ar':'en'),'warning'); }
    updateSector({...sector, devices:[...(sector.devices||[]),saved]});
    setDevice({name:'',type:'',value:'',unit:'',status:'Online'});
  };
  const sectorWorkers=[...(sector.workers||[]), ...serviceWorkers];
  const accountWorkers=workerAccounts();
  const assignedAccountWorkers=accountWorkers.filter(w=>String(w.sectorId || w.assignedSectorId || w.assignedSector || w.sector || '')===String(sector.id) || String(w.sector || w.assignedSectorName || '')===String(sector.name));
  const sectorWorkersTable=[...sectorWorkers,...assignedAccountWorkers.filter(w=>!sectorWorkers.some(sw=>String(sw.email||sw.username||sw.id||sw.name)===String(w.email||w.username||w.id||w.name)))];
  const availableWorkers=[...sectorWorkers,...accountWorkers.filter(w=>!sectorWorkers.some(sw=>String(sw.email||sw.username||sw.id)===String(w.email||w.username||w.id)))];
  const sectorDeviceList = [...serviceSectorDevices, ...(sector.devices||[])].filter((d,idx,arr)=>arr.findIndex(x=>String(x.id||x._id||x.deviceSerial||x.name)===String(d.id||d._id||d.deviceSerial||d.name))===idx);
  const addTask=async()=>{
    if(!task.name.trim()) return;
    const selectedWorker=availableWorkers.find(w=>String(w.email || w.username || w.id || w._id || w.name)===String(task.assignedToEmail)) || {};
    const workerId=selectedWorker._id || selectedWorker.id || '';
    const workerEmail=selectedWorker.email || selectedWorker.username || '';
    if(!workerId && !workerEmail){ addNotification(langFromDocument()==='ar'?'اختار عامل حقيقي قبل إرسال المهمة.':'Select a real worker before sending the task.','error'); return; }
    try{
      const {data}=await tasksAPI.create({
        title:task.name,name:task.name,description:task.description,details:task.description,status:'pending',priority:task.priority,
        workerId:workerId || undefined, assignedWorker:workerId || undefined, assignedWorkerId:workerId || undefined,
        assignedTo:workerId || workerEmail, assignedToEmail:workerEmail || undefined, workerEmail:workerEmail || undefined,
        sectorId:sector._id || sector.id, assignedSector:sector._id || sector.id, sector:sector.name, dueDate:task.dueDate || undefined
      });
      const payload=data?.task || data?.data?.task || data?.data || data;
      const created=normalizeTaskRecord(payload,0);
      if(!backendEntityId(created)){ addNotification(langFromDocument()==='ar'?'الباك إند لم يرجع ID صالح للمهمة.':'Backend did not return a valid task ID.','error'); return; }
      updateSector({...sector, tasks:[created,...(sector.tasks||[])]});
      setGlobalTasks([created,...getGlobalTasks().filter(x=>String(backendEntityId(x))!==String(backendEntityId(created)))]);
      addNotification(`${t('taskSent')} ${task.name} - ${t('workerMobileNotification')}`,'success');
      setTask({name:'',description:'',priority:'Medium',assignedToEmail:'',dueDate:''});
    }catch(err){ addNotification(serviceFriendlyError(err,langFromDocument()),'error'); }
  };
  const addNote=()=>{
    if(!note.trim()) return;
    updateSector({...sector, notes:[note,...(sector.notes||[])]});
    setNote('');
  };
  const rows=sector.rows||['Healthy','Healthy','Moderate Stress','Healthy'];
  const nextIrrigation=isDanger(sector.status)?t('irrigationUrgent'):t('irrigationNormal');
  const latestReading={cropType:firstNonEmpty(sectorSensorState.history[0]?.cropType, sectorSensorState.history[0]?.plantType, sectorSensorState.history[0]?.crop, ''),...(sector.sensors||{}),...(sectorSensorState.latest||{})};
  const latestHistory=sectorSensorState.history[0] || {};
  const latestFinalStatus=latestHistory.final_status || latestHistory.status || sector.status || sector.diagnosis?.condition || 'Healthy';
  const latestDisease=latestHistory.disease_name || latestHistory.diseaseName || latestHistory.disease || sector.diagnosis?.disease || t('noDiseaseDetected');
  const latestConfidence=Number(latestHistory.confidence ?? sector.diagnosis?.confidence ?? .8) || 0;
  const latestDiagnosisDate=latestHistory.date || latestHistory.createdAt || sector.lastUpdated || new Date().toISOString();
  const returnedSectorPlant = firstNonEmpty(
    sectorSensorState.imageHistory?.[0]?.cropType, sectorSensorState.imageHistory?.[0]?.crop_type, sectorSensorState.imageHistory?.[0]?.plantType, sectorSensorState.imageHistory?.[0]?.plant_type, sectorSensorState.imageHistory?.[0]?.crop,
    latestHistory.cropType, latestHistory.crop_type, latestHistory.plantType, latestHistory.plant_type, latestHistory.crop,
    sectorSensorState.latest?.cropType, sectorSensorState.latest?.crop_type, sectorSensorState.latest?.plantType, sectorSensorState.latest?.plant_type, sectorSensorState.latest?.crop,
    ''
  );
  const sectorPlantLabel = returnedSectorPlant ? localizeValue(returnedSectorPlant,t) : '';
  const sensorCards=[
    {key:'temperature',label:t('temperature'),value:`${latestReading.temperature ?? '-'}°C`,icon:ThermometerSun},
    {key:'humidity',label:t('humidity'),value:`${latestReading.humidity ?? '-'}%`,icon:Activity},
    {key:'soilMoisture',label:t('soilMoisture'),value:`${latestReading.soilMoisture ?? '-'}%`,icon:Sprout},
    {key:'soilTemp',label:`${t('soilTemp')}${latestReading.soilTempEstimated ? (langFromDocument()==='ar'?' تقديرية':' estimated') : ''}`,value:`${latestReading.soilTemp ?? '-'}°C`,icon:ThermometerSun},
    {key:'light',label:t('light'),value:latestReading.light ?? '-',icon:Sun},
  ];
  const rawActions=asArray(latestHistory.actions || latestHistory.recommendedActions || sector.actions, ['actions','recommendedActions']);
  const actionItems=(rawActions.length?rawActions:(sector.tasks||[]).slice(0,3)).map((item,i)=>({id:item.id||item._id||i,title:item.title||item.name||item.action||item.label||String(item),details:item.details||item.description||item.priority||''})).slice(0,4);
  const historyPreview=sectorSensorState.history.slice(0,5);
  const primaryWorker=sectorWorkersTable[0];
  const sectorCameraEventKey = `ecosense-sector-camera-${sector._id || sector.id || sector.sectorId || 'current'}`;
  const refreshSectorData=()=>{
    setSectorRefreshKey(v=>v+1);
    window.dispatchEvent(new CustomEvent(`${sectorCameraEventKey}:refresh`));
  };
  const analyzeSectorCamera=()=>window.dispatchEvent(new CustomEvent(`${sectorCameraEventKey}:analyze`));
  return <>
    <PageHead
      title={`${t('sectorDetails')} - ${localizeValue(sector.name,t)}`}
      sub={langFromDocument()==='ar'?'صفحة تشغيل مختصرة للقطاع: الكاميرا، الحساسات، التشخيص، الإجراءات، والسجل.':'Compact operating view for camera, sensors, diagnosis, actions, and sector history.'}
      action={<NavLink className="secondary-btn" to="/sectors">{t('backToSectors')}</NavLink>}
    />

    <main className="sector-ops-page">
      <section className={`sector-ops-header ${tone(latestFinalStatus)}`}>
        <div className="sector-ops-title">
          <span><Sprout size={16}/>{t('sectorOverview')}</span>
          <h1>{localizeValue(sector.name,t)}</h1>
          <p>{[localizeValue(sector.location,t), sectorPlantLabel].filter(Boolean).join(' • ')}</p>
        </div>
        <div className="sector-ops-meta">
          {sectorPlantLabel && <Box label={t('cropType')} value={sectorPlantLabel}/>}
          <Box label={t('finalStatus')} value={labelStatus(latestFinalStatus,t)}/>
          <Box label={t('lastUpdated')} value={formatDateTime(latestDiagnosisDate)}/>
          <Box label={t('assignedTo')} value={primaryWorker?.name || primaryWorker?.email || '-'}/>
        </div>
        <div className="sector-ops-actions">
          <button className="secondary-btn" onClick={refreshSectorData}><Zap size={16}/>{langFromDocument()==='ar'?'تحديث البيانات':'Refresh data'}</button>
          <button className="primary-btn" onClick={analyzeSectorCamera}><ScanSearch size={16}/>{langFromDocument()==='ar'?'تحليل الصورة':'Analyze image'}</button>
          <button className="secondary-btn" onClick={()=>document.querySelector('.sector-history-card')?.scrollIntoView({behavior:'smooth',block:'start'})}><FileText size={16}/>{langFromDocument()==='ar'?'فتح تقرير القطاع':'Open sector report'}</button>
        </div>
      </section>

      <section className="sector-ops-grid">
        <div className="sector-ops-main">
          <Panel title={langFromDocument()==='ar'?'كاميرا القطاع':'Sector Camera'}>
            <LiveCamera sector={sector} onAnalysisSaved={()=>setSectorRefreshKey(v=>v+1)} />
          </Panel>

          <Panel title={langFromDocument()==='ar'?'ملخص الحساسات':'Sensor Summary'}>
            {sectorSensorState.error && <p className="form-helper warning-text">{sectorSensorState.error}</p>}
            <div className="sensor-summary-mini-grid">
              {sensorCards.map(({key,label,value,icon:Icon})=><article className="sensor-mini-card" key={key}>
                <span><Icon size={17}/></span>
                <div><small>{label}</small><b>{localizeValue(value,t)}</b></div>
              </article>)}
            </div>
          </Panel>

          <Panel title={t('fieldMap')}>
            <div className="field-map compact-land-map">
              {rows.map((r,i)=><div key={i} className={`plant-row ${tone(r)}`}>
                <span>{i+1}</span>
                {[1,2,3,4,5,6].map(n=><i key={n}/>) }
                <b>{labelStatus(r,t)}</b>
              </div>)}
            </div>
            <div className="land-map-legend"><span className="healthy">Healthy</span><span className="moderate">Warning</span><span className="danger">Danger</span></div>
          </Panel>
        </div>

        <aside className="sector-ops-side">
          <Panel title={t('latestDiagnosis')}>
            <div className={`latest-diagnosis-mini ${tone(latestFinalStatus)}`}>
              <span className={`tag ${tone(latestFinalStatus)}`}>{labelStatus(latestFinalStatus,t)}</span>
              <h3>{translateModelText(latestDisease,t)}</h3>
              <div className="diagnosis-mini-meta"><b>{Math.round(latestConfidence*100)}%</b><span>{formatDateTime(latestDiagnosisDate)}</span></div>
              <button className="secondary-btn wide" onClick={()=>setSelectedDiagnosis(latestHistory)}><Eye size={15}/>{t('viewDetails')}</button>
            </div>
          </Panel>

          <Panel title={langFromDocument()==='ar'?'الإجراءات المطلوبة':'Recommended Actions'}>
            <div className="recommended-actions-mini">
              {actionItems.length ? actionItems.map(item=><article key={item.id}>
                <CheckCircle2 size={16}/><div><b>{translateModelText(item.title,t)}</b>{item.details?<small>{translateModelText(item.details,t)}</small>:null}</div>
              </article>) : <p className="empty-state-soft">{langFromDocument()==='ar'?'لا توجد إجراءات مطلوبة حاليًا.':'No actions required right now.'}</p>}
            </div>
          </Panel>

          <Panel title={t('assignedWorkers')}>
            <div className="sector-worker-mini-list">
              {sectorWorkersTable.slice(0,3).map((w,i)=><article key={w.email || w.username || w.id || w.name || i}>
                <span>{(w.name || w.email || 'W').slice(0,1).toUpperCase()}</span>
                <div><b>{w.name || w.email || w.username || '-'}</b><small>{localizeValue(w.role || w.jobTitle || 'Worker',t)} • {localizeValue(w.status || 'Online',t)}</small></div>
              </article>)}
              {!sectorWorkersTable.length && <p className="empty-state-soft">{langFromDocument()==='ar'?'لا يوجد عامل مسؤول حاليًا.':'No assigned worker yet.'}</p>}
            </div>
          </Panel>

          <Panel title={t('sectorSensors')}>
            <div className="device-list compact-device-list">{sectorDeviceList.slice(0,4).map(d=><div className="device-row" key={d.id || d._id || d.name}><Cpu size={17}/><div><b>{localizeValue(d.name || d.deviceName,t)}</b><span>{localizeValue(d.type || d.deviceType || 'Device',t)} • {localizeValue(d.value ?? d.lastReading?.value ?? '-',t)}{d.unit || d.lastReading?.unit || ''}</span></div><em>{localizeValue(d.status || 'Online',t)}</em></div>)}</div>
            {!sectorDeviceList.length && <p className="empty-state-soft">{langFromDocument()==='ar'?'لا توجد أجهزة مسجلة لهذا القطاع.':'No devices registered for this sector.'}</p>}
          </Panel>
        </aside>
      </section>

      <section className="sector-history-card">
        <div className="compact-section-head"><div><h3>{langFromDocument()==='ar'?'آخر تشخيصات القطاع':'Sector History'}</h3><p>{langFromDocument()==='ar'?'آخر 5 سجلات فقط لتقليل الزحمة.':'Showing the latest 5 records only.'}</p></div><NavLink className="secondary-btn" to={`/my-diagnoses?sectorId=${sector._id || sector.id}`}>{langFromDocument()==='ar'?'عرض كل التشخيصات':'View all diagnoses'}</NavLink></div>
        <div className="sector-history-mini-list">
          {historyPreview.length ? historyPreview.map((row,i)=><article key={row.id || row._id || i}>
            <span className={`tag ${tone(row.final_status || row.status)}`}>{labelStatus(row.final_status || row.status,t)}</span>
            <div><b>{translateModelText(row.disease_name || row.disease || row.diagnosis || t('diagnosisText'),t)}</b><small>{formatDateTime(row.date || row.createdAt)} • {Math.round((row.confidence || 0)*100)}%</small></div>
            <button className="ghost-btn" onClick={()=>setSelectedDiagnosis(row)}>{t('viewDetails')}</button>
          </article>) : <div className="farm-empty-focus small"><p>{langFromDocument()==='ar'?'لا توجد بيانات متاحة حاليًا':'No data is available right now'}</p></div>}
        </div>
      </section>

      <details className="calm-details sector-advanced-details">
        <summary><Layers3 size={18}/>{langFromDocument()==='ar'?'إدارة تشغيل إضافية':'Additional operations'}</summary>
        <div className="sector-detail-grid-final sector-detail-grid-small">
          <div className="sector-detail-column sector-assets-column">
            <Panel title={t('equipment')}>
              <div className="panel-action-head"><span>{(sectorDeviceList.length || (sector.equipment||[]).length)} {t('items')}</span><button className="primary-btn small-btn" onClick={()=>setEquipmentOpen(!equipmentOpen)}><Plus size={15}/>{langFromDocument()==='ar'?'إضافة معدات':'Add Equipment'}</button></div>
              <div className="equipment-grid">{[...sectorDeviceList,...(sector.equipment||[]).map((name,i)=>({id:`eq-${i}`,name,type:'Equipment',status:'Online'}))].map((e,i)=><div key={e.id || i} className="equipment-card"><Cpu size={20}/><b>{localizeValue(e.name||e,t)}</b><span>{localizeValue(e.type||t('equipment'),t)} • {localizeValue(e.status||'Online',t)}</span></div>)}</div>
              {equipmentOpen && <div className="form-grid compact equipment-add-form"><Field label={langFromDocument()==='ar'?'اسم المعدة':'Equipment name'}><input value={device.name} onChange={e=>setDevice({...device,name:e.target.value})}/></Field><Field label={langFromDocument()==='ar'?'نوع المعدة':'Equipment type'}><input value={device.type} onChange={e=>setDevice({...device,type:e.target.value})}/></Field><Field label={t('status')}><select value={device.status} onChange={e=>setDevice({...device,status:e.target.value})}><option>Online</option><option>Offline</option></select></Field><Field label={langFromDocument()==='ar'?'آخر قراءة / ملاحظة':'Last reading / note'}><input value={device.value} onChange={e=>setDevice({...device,value:e.target.value})}/></Field><button className="primary-btn" onClick={addDevice}><Plus size={18}/>{langFromDocument()==='ar'?'حفظ المعدة':'Save Equipment'}</button></div>}
            </Panel>
          </div>
          <div className="sector-detail-column sector-work-column">
            <Panel title={t('careTasks')}>
              <div className="task-list">{(sector.tasks||[]).map(x=><WorkerTaskCard key={x.id} task={x} sector={sector} updateSector={updateSector} t={t}/>)}</div>
              <div className="form-grid compact task-create-grid"><Field label={t('taskName')}><input value={task.name} onChange={e=>setTask({...task,name:e.target.value})}/></Field><Field label={t('assignedTo')}><select value={task.assignedToEmail} onChange={e=>setTask({...task,assignedToEmail:e.target.value})}><option value="">{langFromDocument()==='ar'?'اختر العامل':'Select worker'}</option>{availableWorkers.map(w=><option key={w.email || w.username || w.id || w.name} value={w.email || w.username || w.id || w.name}>{w.name || w.email || w.username}</option>)}</select></Field><Field label={t('priority')}><select value={task.priority} onChange={e=>setTask({...task,priority:e.target.value})}><option>High</option><option>Medium</option><option>Low</option></select></Field><Field label={langFromDocument()==='ar'?'موعد التنفيذ':'Due date'}><input type="datetime-local" value={task.dueDate} onChange={e=>setTask({...task,dueDate:e.target.value})}/></Field><Field label={langFromDocument()==='ar'?'وصف المهمة':'Task description'}><input value={task.description} onChange={e=>setTask({...task,description:e.target.value})}/></Field></div>
              <button className="primary-btn" onClick={addTask}><Plus size={18}/>{t('addTask')}</button>
            </Panel>
            <Panel title={t('notes')}><div className="notes-list">{(sector.notes||[]).map((n,i)=><p key={i}>{translateModelText(n,t)}</p>)}</div><Field label={t('noteText')}><input value={note} onChange={e=>setNote(e.target.value)}/></Field><button className="primary-btn wide" onClick={addNote}><Plus size={18}/>{t('addNote')}</button></Panel>
          </div>
        </div>
      </details>

      <section className="sector-assistant-bottom"><button className="primary-btn wide" onClick={()=>setSectorAssistantOpen(!sectorAssistantOpen)}><Sparkles size={18}/>{langFromDocument()==='ar'?'اسأل المساعد':'Ask Assistant'}</button>{sectorAssistantOpen && <div className="sector-assistant-box"><MiniChat context={sector}/></div>}</section>
      {selectedDiagnosis && <DiagnosisDetailsModal record={selectedDiagnosis} onClose={()=>setSelectedDiagnosis(null)} labels={{report:t('downloadPdf')}}/>}
    </main>
  </>
}



function getServiceOriginFromApiBase(){
  const base=String(API_BASE || '').replace(/\/$/, '');
  if(!base) return '';
  return base.endsWith('/api') ? base.slice(0,-4) : base;
}

function resolveServiceMediaUrl(value){
  if(!value) return '';
  if(typeof value === 'object'){
    return resolveServiceMediaUrl(
      value.imageUrl || value.url || value.secure_url || value.secureUrl || value.snapshotUrl || value.photoUrl || value.path || value.src
    );
  }
  const raw=String(value).trim();
  if(!raw) return '';
  if(/^data:|^blob:|^https?:\/\//i.test(raw)) return raw;
  if(raw.startsWith('//')) return `https:${raw}`;
  if(raw.startsWith('/')) return `${getServiceOriginFromApiBase()}${raw}`;
  return raw;
}

function firstArrayItem(value){
  if(Array.isArray(value)) return value[0];
  return value;
}

function firstImageRecord(payload){
  const rows = imageHistoryRecordsFromPayload(payload);
  if (rows.length) return rows[0];
  const body = payload?.data ?? payload;
  const candidates = [
    body?.latestImage,
    body?.latest,
    body?.image,
    body?.snapshot,
    firstArrayItem(body?.images),
    firstArrayItem(body?.items),
    firstArrayItem(body?.results),
    firstArrayItem(body?.data?.images),
    firstArrayItem(body?.data?.items),
    firstArrayItem(body?.data?.results),
    firstArrayItem(body?.data),
    firstArrayItem(body),
  ].filter(Boolean);
  return candidates.find(Boolean) || null;
}

function imageInfoFromRecord(record){
  if(!record) return {url:'', capturedAt:'', status:'', deviceId:''};
  const nested = record.camera || record.device || record.latestImage || record.image;
  const url = resolveServiceMediaUrl(
    record.imageUrl || record.url || record.secure_url || record.secureUrl || record.snapshotUrl || record.snapshotURL ||
    record.latestImageUrl || record.lastImageUrl || record.photoUrl || record.fileUrl || record.publicUrl ||
    record.image || record.photo || record.file || record.path || record.src ||
    nested?.imageUrl || nested?.url || nested?.snapshotUrl || nested?.photoUrl
  );
  return {
    url,
    capturedAt: record.capturedAt || record.createdAt || record.updatedAt || record.timestamp || record.time || nested?.capturedAt || nested?.updatedAt || '',
    status: record.cameraStatus || record.status || nested?.status || '',
    deviceId: record.deviceId || record.cameraId || nested?._id || nested?.id || nested?.deviceId || '',
  };
}

function sectorImageInfo(sector){
  return imageInfoFromRecord({
    imageUrl: sector?.lastCameraImage || sector?.lastCapturedImage || sector?.latestImage || sector?.imageUrl || sector?.image,
    capturedAt: sector?.lastImageAt || sector?.lastCapturedAt || sector?.latestImageAt || sector?.camera?.capturedAt,
    cameraStatus: sector?.cameraStatus || sector?.camera?.status,
    deviceId: sector?.cameraId || sector?.camera?._id || sector?.camera?.id,
    camera: sector?.camera,
  });
}

function LiveCamera({sector, onAnalysisSaved}){
  const {t,lang}=useLang();
  const initialInfo = sectorImageInfo(sector);
  const [online,setOnline]=useState(Boolean(sector?.cameraOnline ?? initialInfo.url));
  const [snapshotInfo,setSnapshotInfo]=useState(initialInfo);
  const [loading,setLoading]=useState(false);
  const [analyzing,setAnalyzing]=useState(false);
  const [error,setError]=useState('');
  const [analysisResult,setAnalysisResult]=useState(null);
  const [showAnalysisDetails,setShowAnalysisDetails]=useState(false);
  const sectorId = sector?._id || sector?.id || sector?.sectorId;
  const cameraEventKey = `ecosense-sector-camera-${sectorId || 'current'}`;

  const cameraLabels = lang === 'ar'
    ? { title:'آخر صورة من كاميرا القطاع', empty:'لا توجد صورة ملتقطة بعد من كاميرا هذا القطاع.', refresh:'تحديث الصورة', analyze:'تحليل الصورة', noImage:'لا توجد صورة لتحليلها. من فضلك حدّث صورة الكاميرا أولًا.', refreshed:'تم تحديث صورة الكاميرا.', noNew:'لا توجد صورة ملتقطة بعد. اضغط تحديث لاحقًا عند توفر صورة.', failed:'تعذر تحديث الصورة حاليًا. حاول مرة أخرى.', result:'نتيجة تحليل الصورة' }
    : { title:'Latest sector camera image', empty:'No camera image captured for this sector yet.', refresh:'Refresh Image', analyze:'Analyze Image', noImage:'No image is available to analyze. Please refresh the camera image first.', refreshed:'Camera image refreshed.', noNew:'No captured image is available yet. Refresh again when a capture is available.', failed:'Could not refresh the image right now. Try again.', result:'Image Analysis Result' };

  const loadLatestImage = async({silent=false}={})=>{
    if(!sectorId){
      const fallback = sectorImageInfo(sector);
      setSnapshotInfo(fallback);
      setOnline(Boolean(fallback.url || sector?.cameraOnline));
      return Boolean(fallback.url);
    }
    if(!silent) setLoading(true);
    setError('');
    try{
      const attempts = [
        () => imagesAPI.getLatestForSector(sectorId),
        () => imagesAPI.getHistoryWithFallback({sectorId, limit:1}),
        () => imagesAPI.getHistoryWithFallback({sector:sectorId, limit:1}),
        () => imagesAPI.getHistoryWithFallback({sectorId, page:1, limit:1}),
      ];
      let info = null;
      for(const request of attempts){
        const res = await request();
        const record = firstImageRecord(res.data);
        const next = imageInfoFromRecord(record);
        if(next.url){ info = next; break; }
      }
      if(info?.url){
        setSnapshotInfo(info);
        setOnline(String(info.status || '').toLowerCase() !== 'offline');
        return true;
      }
      const fallback = sectorImageInfo(sector);
      setSnapshotInfo(fallback);
      setOnline(Boolean(fallback.url || sector?.cameraOnline));
      return Boolean(fallback.url);
    }catch(err){
      const fallback = sectorImageInfo(sector);
      setSnapshotInfo(fallback);
      setOnline(Boolean(fallback.url || sector?.cameraOnline));
      setError(cameraLabels.failed);
      if(!silent) addNotification(userFriendlyServiceError(err,lang),'warning');
      return Boolean(fallback.url);
    }finally{
      if(!silent) setLoading(false);
    }
  };

  useEffect(()=>{
    loadLatestImage({silent:true});
    const timer = setInterval(()=>loadLatestImage({silent:true}), 60000);
    return ()=>clearInterval(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectorId]);

  const refreshImage=async()=>{
    const found=await loadLatestImage();
    addNotification(found ? cameraLabels.refreshed : cameraLabels.noNew, found?'success':'warning');
  };

  const analyzeSnapshot=async()=>{
    const snapshot=snapshotInfo.url;
    if(!snapshot){ addNotification(cameraLabels.noImage,'warning'); return; }
    setAnalyzing(true);
    setAnalysisResult(null);
    try{
      const sensorInput=sector?.sensors || {};
      const formData=new FormData();
      const snapshotFile = await fileFromImageUrl(snapshot);
      formData.append('image', snapshotFile);
      const cameraDeviceSerial = firstNonEmpty(sector?.deviceSerial, sector?.device_serial, '');
      if (cameraDeviceSerial) formData.append('deviceSerial', String(cameraDeviceSerial));
      formData.append('sectorId', String(sectorId || ''));
      const response=await imagesAPI.upload(formData);
      const data=response.data || response;
      const normalized=normalize(data, { ...sensorInput, sectorId, imageUrl:snapshot }, true);
      normalized.image_preview=snapshot;
      normalized.diagnosis_mode='image';
      setAnalysisResult(normalized);
      if (typeof onAnalysisSaved === 'function') onAnalysisSaved(normalized);
      // Persisted by POST /images/upload; no local fake history write here.
      triggerModelAlerts(normalized,t);
      addNotification(t('diagnosisSuccess'),'success');
    }catch(err){
      addNotification(userFriendlyServiceError(err,lang),'warning');
    }finally{ setAnalyzing(false); }
  };

  useEffect(()=>{
    const onRefresh = () => refreshImage();
    const onAnalyze = () => analyzeSnapshot();
    window.addEventListener(`${cameraEventKey}:refresh`, onRefresh);
    window.addEventListener(`${cameraEventKey}:analyze`, onAnalyze);
    return ()=>{
      window.removeEventListener(`${cameraEventKey}:refresh`, onRefresh);
      window.removeEventListener(`${cameraEventKey}:analyze`, onAnalyze);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraEventKey, snapshotInfo.url, analyzing]);

  const snapshot = snapshotInfo.url;
  const capturedDate = snapshotInfo.capturedAt ? new Date(snapshotInfo.capturedAt) : null;
  const capturedAt = capturedDate && !Number.isNaN(capturedDate.getTime()) ? capturedDate.toLocaleString(lang==='ar'?'ar-EG':'en-US') : '';
  const cameraState = online ? t('cameraOnline') : t('cameraOffline');

  return <div className={`live-camera pro-camera ${online?'online':'offline'}`}>
    <div className="camera-topline">
      <span><Camera size={18}/>{cameraLabels.title}</span>
      <div className="camera-top-actions camera-status-only">
        <b className={`tag ${online?'green':'amber'}`}>{cameraState}</b>
        <button className="secondary-btn small-btn" type="button" onClick={refreshImage} disabled={loading}>{loading?<Activity className="spin" size={14}/>:<ImagePlus size={14}/>} {cameraLabels.refresh}</button>
      </div>
    </div>
    <div className="camera-frame refined-camera-frame">
      <div className="snapshot-placeholder latest-sector-image">
        {snapshot ? <>
          <img src={snapshot} alt={cameraLabels.title} />
          <div className="camera-image-meta">
            <b>{t('lastCapturedImage')}</b>
            <span>{capturedAt || t('lastUpdated')}</span>
            {snapshotInfo.deviceId ? <small>{snapshotInfo.deviceId}</small> : null}
          </div>
        </> : <><ImagePlus size={46}/><b>{cameraLabels.title}</b><p>{loading ? t('loadingData') : (error || cameraLabels.empty)}</p><small>{lang==='ar'?'اضغط تحديث لجلب آخر صورة عند توفرها.':'Press refresh to load the latest image when available.'}</small></>}
      </div>
    </div>
    {analysisResult && <div className="camera-analysis-result camera-analysis-compact">
      <div className="camera-analysis-head"><b>{cameraLabels.result}</b><span className={`tag ${tone(analysisResult.final_status)}`}>{labelStatus(analysisResult.final_status,t)}</span></div>
      <p>{displayText(analysisResult.diagnosis || analysisResult.disease_name || analysisResult.final_status,t)}</p>
      <button className="secondary-btn small-btn" onClick={()=>setShowAnalysisDetails(v=>!v)}><Eye size={15}/>{lang==='ar'?'عرض التفاصيل':'View details'}</button>
      {showAnalysisDetails && <div className="camera-full-result"><Result result={analysisResult}/></div>}
    </div>}
  </div>
}

function getDiseaseGuideItems(t){
  return [
    { key:'leafSpot', aliases:['leaf spot','fungal','fungus','بقع','تبقع','فطر','فطري'], name:t('leafSpotFungal'), severity:t('high'), symptoms:t('diseaseLeafSpotSymptoms'), causes:t('diseaseLeafSpotCauses'), recommendations:t('diseaseLeafSpotSolution'), actions:t('diseaseLeafSpotPrevention'), task:t('createWorkerTaskRecommended') },
    { key:'chlorosis', aliases:['yellow','chlorosis','اصفرار','اصفر','صفراء'], name:t('generalVisualStress'), severity:t('moderate'), symptoms:t('diseaseVisualStressSymptoms'), causes:t('diseaseVisualStressCauses'), recommendations:t('diseaseVisualStressSolution'), actions:t('diseaseVisualStressPrevention'), task:t('createWorkerTaskNotRequired') },
    { key:'severeStress', aliases:['severe','high stress','critical','danger','ذبول','حرارة','جفاف','خطر','شديد'], name:t('severeEnvironmentalStress'), severity:t('high'), symptoms:t('diseaseSevereStressSymptoms'), causes:t('diseaseSevereStressCauses'), recommendations:t('diseaseSevereStressSolution'), actions:t('diseaseSevereStressPrevention'), task:t('createWorkerTaskRecommended') }
  ];
}

function formatDiseaseGuideAnswer(item,t){
  return `${t('assistantDiseaseCardTitle')}: ${item.name}\n\n${t('diseaseSymptoms')}: ${item.symptoms}\n${t('diseaseCauses')}: ${item.causes}\n${t('diseaseSeverity')}: ${item.severity}\n${t('recommendations')}: ${item.recommendations}\n${t('actions')}: ${item.actions}\n${t('suggestedTask')}: ${item.task}`;
}

function findDiseaseGuideItem(query,t,context=null){
  const s=String(query||'').toLowerCase();
  const guide=getDiseaseGuideItems(t);
  const contextDisease=String(context?.diagnosis?.disease || context?.disease || context?.latestDisease || '').toLowerCase();
  return guide.find(item=>item.aliases.some(a=>s.includes(String(a).toLowerCase()))) || guide.find(item=>item.aliases.some(a=>contextDisease.includes(String(a).toLowerCase()))) || (s.includes('recommend') || s.includes('توص') || s.includes('مرض') || s.includes('disease') ? guide[0] : null);
}

function agricultureReply(q, t, context=null){
  const s=String(q||'').toLowerCase();
  const isAr = t('name') === 'العربية';
  const disease = findDiseaseGuideItem(q,t,context);
  if(disease) return formatDiseaseGuideAnswer(disease,t);
  if(s.includes('ري')||s.includes('water')||s.includes('irrigation')) return context && isDanger(context.status) ? t('irrigationUrgent') : `${t('irrigationNormal')}. ${t('soilMoisture')}: ${context?.sensors?.soilMoisture ?? 45}%`;
  if(s.includes('اصفر')||s.includes('yellow')) return formatDiseaseGuideAnswer(getDiseaseGuideItems(t)[1],t);
  if(s.includes('فطر')||s.includes('fung')||s.includes('بقع')) return formatDiseaseGuideAnswer(getDiseaseGuideItems(t)[0],t);
  if(s.includes('حساس')||s.includes('sensor')) return isAr ? 'أهم الحساسات: رطوبة التربة، حرارة الجو، الرطوبة، حرارة التربة، والإضاءة. ويمكن إضافة pH وEC لاحقًا.' : 'Key sensors include soil moisture, air temperature, humidity, soil temperature, and light. pH and EC can be added later.';
  if(s.includes('تشخيص')||s.includes('diagnosis')) return context ? (isAr ? `آخر تشخيص للقطاع: ${localizeValue(context.status,t)} - ${localizeValue(context.diagnosis?.disease,t)}.\n${t('assistantCanHelpWithGuide')}` : `Latest sector diagnosis: ${localizeValue(context.status,t)} - ${localizeValue(context.diagnosis?.disease,t)}.\n${t('assistantCanHelpWithGuide')}`) : (isAr ? 'افتح صفحة التشخيص وارفع صورة النبات مع قراءات الحساسات للحصول على نتيجة أوضح. أقدر أيضًا أشرح الأمراض والتوصيات من داخل المساعد.' : 'Open the diagnosis page and upload a plant image with sensor readings for a clearer result. I can also explain diseases and recommendations inside the Assistant.');
  return isAr ? 'أقدر أساعدك في الري، الأمراض، الحساسات، التشخيص، التسميد، أو متابعة القطاعات. اكتب اسم المرض أو الأعراض وسأعرض الأعراض والأسباب والتوصيات وهل تحتاج مهمة للعامل.' : 'I can help with irrigation, diseases, sensors, diagnosis, fertilization, or sector follow-up. Type the disease name or symptoms and I will show symptoms, causes, recommendations, and whether a worker task is needed.';
}

function MiniChat({context=null}){
  const {t}=useLang();
  const [messages,setMessages]=useState([{from:'bot',text:t('assistantWelcome')}]);
  const [input,setInput]=useState('');
  const send=async(q=input)=>{
    if(!String(q).trim()) return;
    const userMsg={from:'user',text:q};
    setMessages(m=>[...m,userMsg,{from:'bot',text:'...'}]);
    setInput('');
    try{
      const res=await fetch(`${API_BASE}/chat/agriculture`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q,context})});
      if(!res.ok) throw new Error('chat');
      const data=await res.json();
      const reply=data.reply || data.answer || data.message || agricultureReply(q,t,context);
      setMessages(m=>[...m.slice(0,-1),{from:'bot',text:reply}]);
    }catch{
      const reply=agricultureReply(q,t,context);
      setMessages(m=>[...m.slice(0,-1),{from:'bot',text:reply}]);
    }
  };
  const quick=[t('leafSpotQuestion'),t('chlorosisQuestion'),t('showDiseaseRecommendations'),t('qWater'),t('qSensors'),t('explainDiagnosisPrompt')];
  return <div className="chat-widget">
    <div className="quick-questions">{quick.map(q=><button key={q} onClick={()=>send(q)}>{q}</button>)}</div>
    <div className="chat-messages">{messages.map((m,i)=><div key={i} className={`chat-msg ${m.from} ${String(m.text).includes('\n')?'structured':''}`}>{m.text}</div>)}</div>
    <div className="chat-input"><input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==='Enter')send()}} placeholder={t('chatPlaceholder')}/><button className="primary-btn" onClick={()=>send()}>{t('send')}</button></div>
  </div>
}


function PlantProfiles(){
  const {t}=useLang();
  const sectors=getUserStore('sph_sectors', getStore('sph_sectors',[]));
  const [plants,setPlants]=useState(getUserStore('sph_plants',[]));
  const [form,setForm]=useState({name:'',crop:'',sectorId:sectors[0]?.id||'',age:'',plantedAt:''});
  const save=()=>{
    if(!form.name.trim()) return;
    const p={...form,id:Date.now(),lastDiagnosis:'Healthy',disease:'No Clear Disease Detected',timeline:[]};
    const all=[p,...plants]; setPlants(all); setUserStore('sph_plants',all); setForm({name:'',crop:'',sectorId:sectors[0]?.id||'',age:'',plantedAt:''});
  };
  return <>
    <PageHead title={t('plantProfiles')} sub={t('userDataOnly')} action={null}/>
    <Panel title={t('addPlant')}>
      <div className="form-grid">
        <Field label={t('plantName')}><input value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></Field>
        <Field label={t('cropType')}><CropTypeSelect lang={langFromDocument()} value={form.crop || ''} onChange={value=>setForm({...form,crop:value})} /></Field>
        <Field label={t('linkedSector')}><select value={form.sectorId} onChange={e=>setForm({...form,sectorId:Number(e.target.value)})}>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}</option>)}</select></Field>
        <Field label={t('plantAge')}><input value={form.age} onChange={e=>setForm({...form,age:e.target.value})}/></Field>
        <Field label={t('plantedAt')}><input type="date" value={form.plantedAt} onChange={e=>setForm({...form,plantedAt:e.target.value})}/></Field>
      </div>
      <button className="primary-btn" onClick={save}><Plus size={18}/>{t('save')}</button>
    </Panel>
    <div className="plant-grid">{plants.length?plants.map(p=><article className={`plant-profile-card ${tone(p.lastDiagnosis)}`} key={p.id}><Sprout/><h3>{p.name}</h3><p>{localizeValue(p.crop,t)} • {localizeValue(sectors.find(s=>s.id==p.sectorId)?.name||'',t)}</p><strong>{labelStatus(p.lastDiagnosis,t)}</strong><span>{translateModelText(p.disease,t)}</span><NavLink className="secondary-btn" to={`/plants/${p.id}`}>{t('openProfile')}</NavLink></article>):<Panel title={t('noPlants')}/>}</div>
  </>
}

function PlantProfileDetails(){
  const {t}=useLang();
  const id=Number(useLocation().pathname.split('/').pop());
  const sectors=getUserStore('sph_sectors', getStore('sph_sectors',[]));
  const plants=getUserStore('sph_plants',[]);
  const p=plants.find(x=>Number(x.id)===id)||plants[0];
  const sector=sectors.find(s=>s.id==p.sectorId)||sectors[0];
  return <>
    <PageHead title={`${t('plantProfile')} - ${p.name}`} sub={t('userDataOnly')} action={<NavLink to="/plants" className="secondary-btn">{t('plantProfiles')}</NavLink>}/>
    <div className="grid-2">
      <Panel title={t('plantProfile')}>
        <div className={`diagnosis-hero ${tone(p.lastDiagnosis)}`}><div><span>{t('condition')}</span><h2>{labelStatus(p.lastDiagnosis,t)}</h2><p>{translateModelText(p.timeline?.[0]?.summary,t)}</p></div><div className="disease-card"><span>{t('diseaseProblem')}</span><b>{translateModelText(p.disease,t)}</b></div></div>
        <div className="data-grid"><Box label={t('cropType')} value={p.crop}/><Box label={t('linkedSector')} value={sector?.name}/><Box label={t('plantAge')} value={p.age}/><Box label={t('plantedAt')} value={p.plantedAt}/></div>
      </Panel>
      <Panel title={t('linkedSensors')}><ReadingGrid r={{cropType:p.crop,...(sector?.sensors||{})}}/></Panel>
    </div>
    <Panel title={t('diagnosisTimeline')}><div className="timeline">{(p.timeline||[]).map(x=><div key={x.id} className="timeline-item"><b>{labelStatus(x.status,t)}</b><span>{x.time}</span><p>{translateModelText(x.summary,t)}</p></div>)}</div></Panel>
  </>
}

function SettingsPage({theme,setTheme}){
  const {t,lang,setLang}=useLang();
  const role = currentRole();
  const [liveUser,setLiveUser]=useState(currentUser());
  useEffect(()=>{
    const token=localStorage.getItem('ecosense_token');
    if(!token || ['local-demo-token','local-worker-token'].includes(token)) return undefined;
    let mounted=true;
    authAPI.getMe().then((res)=>{
      if(!mounted) return;
      const user=res?.data?.user || res?.data?.data?.user || res?.data?.data || res?.data || {};
      if(user && (user.email || user.username || user.id || user._id)){
        setLiveUser(user);
        setLoggedInUser(user, token);
      }
    }).catch(()=>{});
    return()=>{mounted=false};
  },[]);
  const storedUser = liveUser || getStore('ecosense_user', {}) || {};
  const userEmail = String(storedUser.email || storedUser.username || currentUserEmail()).toLowerCase();
  const globalFarmSettings = getStore('sph_settings', {});
  const defaultProfile = {
    soundAlerts:true,
    tempUnit:'C',
    profileName: storedUser.name || storedUser.fullName || storedUser.displayName || 'Smart Plant User',
    phone: storedUser.phone || storedUser.phoneNumber || '',
    avatar: storedUser.picture || storedUser.avatar || storedUser.photoURL || storedUser.profileImage || '',
    farmName: globalFarmSettings.farmName || 'Ecosense Smart Farm',
    defaultCrop: globalFarmSettings.defaultCrop || '',
    timezone: globalFarmSettings.timezone || 'Africa/Cairo',
    twoFactor:false,
    sessionAlerts:true,
    reportAccess:'owner',
    themeMode: theme || 'light'
  };
  const [settings,setSettings]=useState(()=>({...defaultProfile,...getUserStore('sph_account_settings', defaultProfile, userEmail)}));
  const [uploadState,setUploadState]=useState('');
  useEffect(()=>{
    const nextDefaults={...defaultProfile,profileName:storedUser.name || storedUser.fullName || storedUser.displayName || defaultProfile.profileName,phone:storedUser.phone || storedUser.phoneNumber || defaultProfile.phone,avatar:storedUser.picture || storedUser.avatar || storedUser.photoURL || storedUser.profileImage || defaultProfile.avatar};
    setSettings({...nextDefaults,...getUserStore('sph_account_settings', nextDefaults, userEmail)});
  },[userEmail, storedUser.id, storedUser._id]);
  const profileName = settings.profileName || storedUser.name || storedUser.fullName || 'Smart Plant User';
  const roleTitle = role === 'owner' ? t('owner') : role === 'farm_manager' ? t('farmManager') : t('worker');
  const roleDesc = role === 'owner'
    ? (lang === 'ar' ? 'صلاحيات كاملة لإدارة المزرعة والعمال والأجهزة والتقارير وإنشاء حسابات العمال.' : 'Full access to farm, workers, devices, reports, and worker account creation.')
    : role === 'farm_manager'
      ? (lang === 'ar' ? 'صلاحيات تشغيل ومتابعة: تشخيص، تقارير، حساسات، أجهزة، تنبيهات، ومهام للعمال بدون حذف صلاحيات المالك.' : 'Operational management: diagnosis, reports, sensors, devices, alerts, and worker tasks without owner-sensitive deletion.')
      : (lang === 'ar' ? 'إعدادات عامل فقط: بيانات الحساب، اللغة، المظهر، التنبيهات، المهام، والقطاع المسؤول عنه.' : 'Worker-only settings: account details, language, appearance, personal alerts, tasks, and assigned sector.');
  const initials = profileName.split(' ').filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase() || 'SP';
  const assignedSector = storedUser.sector || storedUser.sectorName || getGlobalTasks().find(taskAssignedToCurrent)?.sector || '-';
  const L={
    profile: t('profile'), account: lang==='ar'?'إعدادات الحساب':'Account Settings', farm: t('farmPreferences'), notifications: lang==='ar'?'التنبيهات':'Notifications', appearance: t('appearance'), security: lang==='ar'?'الأمان':'Security',
    changePhoto: lang==='ar'?'تغيير الصورة':'Change Photo', accountOwner: lang==='ar'?'اسم صاحب الحساب':'Account owner name', contactInfo: lang==='ar'?'بيانات التواصل':'Contact information', saveChanges: lang==='ar'?'حفظ التغييرات':'Save Changes',
    rolePermissions: lang==='ar'?'الدور والصلاحيات':'Role & Permissions', serviceReady: lang==='ar'?'جاهز للتخزين':'Storage ready', languageTheme: lang==='ar'?'اللغة والمظهر':'Language & theme', defaultCrop: lang==='ar'?'المحصول الافتراضي':'Default crop', farmLocation: lang==='ar'?'موقع المزرعة':'Farm location', timezone: lang==='ar'?'المنطقة الزمنية':'Timezone',
    soundAlerts: t('soundAlerts'), sessionAlerts: lang==='ar'?'تنبيهات تسجيل الدخول':'Login alerts', twoFactor: lang==='ar'?'تفعيل حماية إضافية':'Extra security protection', reportAccess: lang==='ar'?'صلاحية التقارير':'Report access', ownerOnly: lang==='ar'?'المالك فقط':'Owner only', lightMode: lang==='ar'?'وضع نهاري':'Light mode', darkMode: lang==='ar'?'وضع ليلي':'Dark mode'
  };
  const save=()=>{
    setUserStore('sph_account_settings',settings,userEmail);
    if(!isWorker()){
      const farmPrefs=getStore('sph_settings',{});
      setStore('sph_settings',{...farmPrefs,farmName:settings.farmName,defaultCrop:settings.defaultCrop,timezone:settings.timezone,tempUnit:settings.tempUnit,reportAccess:settings.reportAccess});
    }
    const mergedUser = {...storedUser, name:settings.profileName, fullName:settings.profileName, phone:settings.phone, avatar:settings.avatar, picture:settings.avatar, email:userEmail};
    setStore('ecosense_user', mergedUser);
    setLiveUser(mergedUser);
    setUploadState(lang==='ar'?'تم حفظ إعدادات الحساب الحالي بنجاح.':'Current account settings saved successfully.');
    addNotification(lang==='ar'?'تم حفظ إعدادات الحساب الحالي.':'Current account settings saved.','success');
  };
  const uploadProfilePhotoToService=async(file,preview)=>{
    const token=localStorage.getItem('ecosense_token');
    if(!token) return null;
    try{
      const fd=new FormData(); fd.append('avatar',file); fd.append('profileImage',file);
      const res=await fetch(`${API_BASE}/users/profile/photo`,{method:'POST',headers:{Authorization:`Bearer ${token}`},body:fd});
      if(!res.ok) throw new Error('Upload failed');
      const data=await res.json();
      return data.url || data.avatar || data.photoURL || data.profileImage || data.imageUrl || data.data?.url || preview;
    }catch(err){ return null; }
  };
  const handleAvatar=e=>{
    const file=e.target.files?.[0];
    if(!file) return;
    if(!file.type.startsWith('image/')){ setUploadState(lang==='ar'?'اختر صورة فقط.':'Please choose an image file.'); return; }
    if(file.size>3*1024*1024){ setUploadState(lang==='ar'?'حجم الصورة يجب ألا يزيد عن 3MB.':'Image must be less than 3MB.'); return; }
    const reader=new FileReader();
    reader.onload=async()=>{
      const preview=reader.result;
      setSettings(prev=>({...prev,avatar:preview}));
      setUploadState(lang==='ar'?'تمت معاينة الصورة. سيتم حفظها للحساب الحالي.':'Image preview added for the current account.');
      const serviceUrl=await uploadProfilePhotoToService(file,preview);
      if(serviceUrl){ setSettings(prev=>({...prev,avatar:serviceUrl})); setUploadState(lang==='ar'?'تم رفع الصورة وتخزينها بنجاح.':'Profile image uploaded successfully.'); }
    };
    reader.readAsDataURL(file);
  };
  const setAppearance=(next)=>{
    if(next==='system'){
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      setTheme(prefersDark?'dark':'light');
      setSettings({...settings,themeMode:'system'});
    } else { setTheme(next); setSettings({...settings,themeMode:next}); }
  };
  const workerCards = isWorker();
  return <>
    <PageHead title={t('settings')} sub={workerCards?t('workerSettingsOnly'):(lang==='ar'?'إعدادات منظمة حسب صلاحيات الحساب الحالي.':'Settings organized by the current account permissions.')} action={!isWorker()?<NavLink className="secondary-btn" to="/onboarding"><Sparkles size={18}/>{t('reopenOnboarding')}</NavLink>:<AccountRoleBadge/>}/>

    <section className={`settings-page-v2 role-settings ${role}`}>
      <article className="settings-profile-card-v2 compact-profile-card">
        <div className="profile-avatar-v2 small-avatar">
          {settings.avatar ? <img src={settings.avatar} alt="Profile"/> : <span>{initials}</span>}
          <label title={L.changePhoto}><Camera size={15}/><input type="file" accept="image/*" onChange={handleAvatar}/></label>
        </div>
        <div className="profile-copy-v2">
          <span className={`profile-role-badge ${role}`}>{role === 'owner' ? <ShieldCheck size={15}/> : role === 'farm_manager' ? <Layers3 size={15}/> : <UserPlus size={15}/>} {roleTitle}</span>
          <h2>{profileName}</h2>
          <p>{userEmail}{settings.phone ? ` • ${settings.phone}` : ''}</p>
        </div>
        <button className="primary-btn" onClick={save}><ShieldCheck size={18}/>{L.saveChanges}</button>
      </article>
      {uploadState && <div className="profile-upload-note compact"><UploadCloud size={18}/>{uploadState}</div>}

      <div className="settings-card-grid-v2 worker-safe-grid">
        <section className="settings-card-v2 profile-section">
          <header><span><UserPlus size={20}/></span><div><h3>{L.profile}</h3><p>{L.contactInfo}</p></div></header>
          <Field label={t('profileName')}><input value={settings.profileName || ''} onChange={e=>setSettings({...settings,profileName:e.target.value})} placeholder={L.accountOwner}/></Field>
          <Field label={t('email')}><input value={userEmail} disabled/></Field>
          <Field label={t('phone')}><input value={settings.phone || ''} onChange={e=>setSettings({...settings,phone:e.target.value})} placeholder="01000000000"/></Field>
        </section>


        {!isWorker() && <section className="settings-card-v2 farm-section">
          <header><span><Layers3 size={20}/></span><div><h3>{L.farm}</h3><p>{lang==='ar'?'تفضيلات المزرعة الأساسية':'Main farm preferences'}</p></div></header>
          <Field label={t('farmName')}><input value={settings.farmName || ''} onChange={e=>setSettings({...settings,farmName:e.target.value})} placeholder={t('farmName')} disabled={!isOwner()}/></Field>
          <Field label={L.defaultCrop}><CropTypeSelect lang={lang} value={settings.defaultCrop || ''} onChange={value=>setSettings({...settings,defaultCrop:value})} allowEmpty={true}/></Field>
          <Field label={L.timezone}><input value={settings.timezone || 'Africa/Cairo'} onChange={e=>setSettings({...settings,timezone:e.target.value})}/></Field>
        </section>}

        <section className="settings-card-v2 notifications-section">
          <header><span><Bell size={20}/></span><div><h3>{L.notifications}</h3><p>{isWorker()? (lang==='ar'?'تنبيهات المهام والقطاع فقط':'Task and assigned-sector alerts only') : (lang==='ar'?'تحكم في أصوات وتنبيهات النظام':'Control alerts and notification behavior')}</p></div></header>
          <label className="setting-toggle v2"><span>{L.soundAlerts}</span><input type="checkbox" checked={settings.soundAlerts!==false} onChange={e=>setSettings({...settings,soundAlerts:e.target.checked})}/><b>{settings.soundAlerts!==false?t('soundOn'):t('soundOff')}</b></label>
          <label className="setting-toggle v2"><span>{L.sessionAlerts}</span><input type="checkbox" checked={settings.sessionAlerts!==false} onChange={e=>setSettings({...settings,sessionAlerts:e.target.checked})}/><b>{settings.sessionAlerts!==false?t('active'):t('offline')}</b></label>
          {!isWorker() && <Field label={t('temperatureUnit')}><select value={settings.tempUnit || 'C'} onChange={e=>setSettings({...settings,tempUnit:e.target.value})}><option value="C">{t('celsius')}</option><option value="F">{t('fahrenheit')}</option></select></Field>}
        </section>

        <section className="settings-card-v2 appearance-section simple-appearance-card">
          <header><span><Globe2 size={20}/></span><div><h3>{L.appearance}</h3><p>{L.languageTheme}</p></div></header>
          <label className="setting-toggle v2 appearance-mode-toggle">
            <span className="appearance-mode-label">{theme==='dark'?<Moon size={18}/>:<Sun size={18}/>} {theme==='dark'?L.darkMode:L.lightMode}</span>
            <input type="checkbox" checked={theme==='dark'} onChange={e=>setAppearance(e.target.checked?'dark':'light')}/>
            <b>{theme==='dark'?L.darkMode:L.lightMode}</b>
          </label>
          <Field label={t('language')}><select value={lang} onChange={e=>setLang(e.target.value)}>{Object.entries(dictionaries).map(([k,v])=><option key={k} value={k}>{v.name}</option>)}</select></Field>
        </section>

        {!isWorker() && <section className="settings-card-v2 security-section">
          <header><span><Lock size={20}/></span><div><h3>{L.security}</h3><p>{lang==='ar'?'حماية الحساب والصلاحيات الحساسة':'Account protection and sensitive permissions'}</p></div></header>
          <label className="setting-toggle v2"><span>{L.twoFactor}</span><input type="checkbox" checked={!!settings.twoFactor} onChange={e=>setSettings({...settings,twoFactor:e.target.checked})}/><b>{settings.twoFactor?t('active'):t('offline')}</b></label>
          <Field label={L.reportAccess}><select value={settings.reportAccess || 'owner'} onChange={e=>setSettings({...settings,reportAccess:e.target.value})} disabled={!isOwner()}><option value="owner">{L.ownerOnly}</option><option value="manager">Farm Manager</option></select></Field>
          <p className="muted-copy compact-note">{isFarmManager()?t('farmManagerPermissions'):t('permissionsOwnerOnly')}</p>
        </section>}

        {isWorker() && <section className="settings-card-v2 worker-task-summary-card">
          <header><span><CheckCircle2 size={20}/></span><div><h3>{t('myTasks')}</h3><p>{lang==='ar'?'مختصر المهام الخاصة بالحساب الحالي':'Current account task summary'}</p></div></header>
          <div className="settings-mini-list"><span>{t('pending')}</span><b>{getGlobalTasks().filter(taskAssignedToCurrent).filter(x=>normalizeTaskStatus(x.status)==='pending').length}</b></div>
          <div className="settings-mini-list"><span>{t('inProgress')}</span><b>{getGlobalTasks().filter(taskAssignedToCurrent).filter(x=>normalizeTaskStatus(x.status)==='inProgress').length}</b></div>
          <NavLink className="primary-btn wide" to="/tasks"><CheckCircle2 size={16}/>{t('myTasks')}</NavLink>
        </section>}
      </div>

      <button className="primary-btn wide settings-bottom-save" onClick={save}><ShieldCheck size={18}/>{L.saveChanges}</button>
    </section>
  </>
}

function FloatingAssistant(){
  const {t}=useLang();
  const [open,setOpen]=useState(false);
  useEffect(()=>{ const handler=()=>setOpen(true); window.addEventListener('ecosense-open-assistant',handler); return()=>window.removeEventListener('ecosense-open-assistant',handler); },[]);
  return <div className={`floating-assistant ${open?'open':''}`}>
    {open && <div className="floating-panel">
      <div className="floating-head"><div><b>{t('floatingAssistant')}</b><small>{t('diseaseGuideInAssistant')}</small></div><button onClick={()=>setOpen(false)}><X size={16}/></button></div>
      <p className="assistant-guide-note">{t('diseaseAssistantHint')}</p>
      <MiniChat/>
    </div>}
    <button className="floating-btn" onClick={()=>setOpen(!open)} title={open?t('closeAssistant'):t('openAssistant')}><Sparkles/></button>
  </div>
}

function AgricultureChat(){
  const {t}=useLang();
  return <>
    <PageHead title={t('chat')} sub={t('chatSubtitle')}/>
    <Panel title={t('sectorChat')}>
      <MiniChat/>
    </Panel>
  </>
}

function generateWorkerPassword(){
  return `WRK-${Math.random().toString(36).slice(2,6).toUpperCase()}-${Math.floor(100+Math.random()*900)}`;
}

function WorkerAccountsPage(){
  const {t,lang}=useLang();
  const [sectors,setSectors]=useState([]);
  const [workers,setWorkers]=useState([]);
  const [lastAccount,setLastAccount]=useState(null);
  const [serviceState,setServiceState]=useState({loading:false,error:'',live:false});
  const [form,setForm]=useState({firstName:'',lastName:'',email:'',password:'',phone:'',address:'',jobTitle:'Plant Care Worker',sectorId:'', permissions:{tasks:true,alerts:true,upload:true,finishTask:true}});
  const [assigningWorkerId,setAssigningWorkerId]=useState(null);
  const [assignSectorId,setAssignSectorId]=useState('');
  const [showWorkerModal,setShowWorkerModal]=useState(false);
  useEffect(()=>{
    let mounted=true;
    setServiceState(s=>({...s,loading:true,error:''}));
    Promise.allSettled([sectorsAPI.getAll(), usersAPI.getWorkers()]).then(([sectorsRes, workersRes])=>{
      if(!mounted) return;

      if(sectorsRes.status==='fulfilled'){
        const sectorRows=asArray(sectorsRes.value.data,['sectors','items','rows','records']);
        const mappedSectors=sectorRows.map((sec,idx)=>({
          ...sec,
          id:sec._id || sec.id || sec.sectorId || idx+1,
          name:sec.name || sec.title || `Sector ${idx+1}`,
          crop:sec.crop || sec.cropType || sec.plantType || '',
          cropType:sec.cropType || sec.crop || sec.plantType || '',
          workers:sec.workers || sec.assignedWorkers || [],
          tasks:sec.tasks || [],
        }));
        setSectors(mappedSectors);
        setStore('sph_sectors',mappedSectors);
        if(!form.sectorId && mappedSectors[0]?.id){
          setForm(prev=>({...prev, sectorId:String(prev.sectorId || mappedSectors[0].id)}));
        }
      }

      if(workersRes.status==='rejected'){
        setWorkers([]);
        setStore('sph_worker_accounts',[]);
        setServiceState({loading:false,error:serviceFriendlyError(workersRes.reason,lang),live:false});
        return;
      }

      const rawList=asArray(workersRes.value.data,['workers','users','items','rows','records']);
      const list=filterWorkersForCurrentOwner(rawList);
      if(!list.length){ setWorkers([]); setStore('sph_worker_accounts',[]); setServiceState({loading:false,error:'',live:true}); return; }
      const mapped=list.map((w,idx)=>({
        id:w._id || w.id || idx+1,
        name:w.name || w.fullName || [w.firstName,w.lastName].filter(Boolean).join(' ') || w.email || `Worker ${idx+1}`,
        email:w.email || w.username || '', username:w.username || w.email || '', phone:w.phone || w.phoneNumber || '',
        address:w.address || w.location || '',
        role:normalizeRole(w.role || 'worker',w.email), jobTitle:w.jobTitle || w.roleLabel || 'Plant Care Worker',
        status:w.status || (w.isActive===false?'Disabled':'Active'), sectorId:w.sectorId || w.assignedSectorId || w.assignedSector?._id || w.sector?._id || w.assignedSector,
        sector:w.sectorName || w.assignedSectorName || w.assignedSector?.name || w.sector?.name || w.sector || '-', lastTask:w.lastTask || w.currentTask || '-',
        password:w.password || w.generatedPassword || w.credentials?.password || '', service:true
      }));
      setWorkers(mapped); setStore('sph_worker_accounts',mapped); setServiceState({loading:false,error:'',live:true});
    });
    return()=>{mounted=false};
  },[lang]);
  const existingWorkers=sectors.flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name,sectorId:s.id,jobTitle:w.jobTitle || w.role || 'Plant Care Worker'})));
  const selectedSector=sectors.find(s=>String(s.id)===String(form.sectorId)) || sectors[0];
  const allWorkers=[...workers,...existingWorkers.filter(w=>!workers.some(a=>String(a.id)===String(w.id) || (a.email && a.email===w.email)))];
  const canManageWorkers=isOwner();
  const L={
    title: t('workers'),
    subtitle: lang==='ar'?'إدارة العاملين والمسؤولين بشكل منظم ومناسب للموبايل.':'Manage workers and supervisors in a clean mobile-first layout.',
    create: t('createNewWorker'), existing:t('addExistingWorker'), credentials:t('generatedCredentials'), addAssign: lang==='ar'?'إضافة / تعيين عامل':'Add / Assign Worker', workersList: lang==='ar'?'قائمة العاملين':'Workers List',
    edit: lang==='ar'?'تعديل':'Edit', assign: lang==='ar'?'تعيين':'Assign', disable: lang==='ar'?'تعطيل':'Disable', disabled: lang==='ar'?'معطل':'Disabled', active:t('active'), status:t('workerStatus'), lastTask:t('lastTask'),
    mobileHint: lang==='ar'?'الكروت مصممة لتظهر بدون تداخل على الهاتف.':'Cards are optimized to stay clean on mobile.', role:t('workerRole'), sector:t('sector'), phone:t('phone')
  };
  const selectExistingWorker=(id)=>{
    const found=existingWorkers.find(w=>String(w.id)===String(id));
    if(found){ const parts=String(found.name||'').trim().split(/\s+/); setForm(prev=>({...prev,firstName:parts[0]||'',lastName:parts.slice(1).join(' ')||'',email:found.email||'',password:'',phone:found.phone||'',address:found.address||prev.address||'',jobTitle:found.jobTitle||found.role||'Plant Care Worker',sectorId:String(found.sectorId||prev.sectorId||'')})); }
  };
  const createWorker=async()=>{
    if(!canManageWorkers){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    const firstName=String(form.firstName||'').trim();
    const lastName=String(form.lastName||'').trim();
    const cleanEmail=String(form.email||'').trim().toLowerCase();
    const submittedPassword=String(form.password||'').trim();
    if(!firstName || !lastName || !cleanEmail || !submittedPassword){
      addNotification(lang==='ar'?'اكتب الاسم الأول والاسم الأخير والإيميل وكلمة المرور أولًا.':'Enter first name, last name, email, and password first.','warning');
      return;
    }
    if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)){
      addNotification(lang==='ar'?'اكتب بريد إلكتروني صحيح للعامل.':'Enter a valid worker email.','warning');
      return;
    }
    if(submittedPassword.length < 6){
      addNotification(lang==='ar'?'كلمة مرور العامل يجب ألا تقل عن 6 أحرف.':'Worker password must be at least 6 characters.','warning');
      return;
    }
    const token=localStorage.getItem('ecosense_token') || localStorage.getItem('sph_token');
    if(!token){
      addNotification(lang==='ar'?'سجل الدخول بحساب المالك أولًا لإنشاء حساب عامل حقيقي.':'Sign in as owner first to create a real worker account.','warning');
      return;
    }
    const selected=form.sectorId ? sectors.find(s=>String(s.id)===String(form.sectorId)) : null;
    const selectedSectorId=selected?.id || form.sectorId || '';
    try{
      const workerBody={
        firstName,
        lastName,
        email:cleanEmail,
        password:submittedPassword,
        phoneNumber:String(form.phone||'').trim(),
        address:String(form.address||'').trim(),
        ...(selectedSectorId ? {
          assignedSector:selectedSectorId,
          sectorId:selectedSectorId,
          assignedSectorId:selectedSectorId,
        } : {}),
        role:'worker',
        accountType:'worker',
        jobTitle:'Plant Care Worker',
      };
      const {data}=await usersAPI.addWorker(workerBody);
      const root=data || {};
      const payload=root.worker || root.user || root.data?.worker || root.data?.user || root.data?.workerAccount || root.data?.account || root.data || root;
      const credentials=extractWorkerCredentials(root,payload,{email:cleanEmail,username:cleanEmail,password:submittedPassword});
      const account={
        id:payload._id || payload.id || root.workerId || Date.now(),
        name:payload.name || payload.fullName || [payload.firstName || firstName,payload.lastName || lastName].filter(Boolean).join(' '),
        email:credentials.email || payload.email || cleanEmail,
        username:credentials.email || credentials.username || payload.email || cleanEmail,
        password:credentials.password || submittedPassword,
        phone:payload.phone || payload.phoneNumber || String(form.phone||'').trim(),
        role:normalizeRole(payload.role || 'worker', payload.email || cleanEmail),
        jobTitle:payload.jobTitle || 'Plant Care Worker',
        status:payload.status || (payload.isActive===false?'Disabled':'Active'),
        permissions:payload.permissions || form.permissions || {},
        sectorId:payload.sectorId || payload.assignedSectorId || payload.assignedSector?._id || payload.sector?._id || selectedSectorId,
        sector:payload.sectorName || payload.assignedSectorName || payload.assignedSector?.name || payload.sector?.name || selected?.name || (lang==='ar'?'غير معين':'Not assigned'),
        lastTask:payload.lastTask || payload.currentTask || (selected?.tasks||[])[0]?.name || '-',
        createdAt:new Date().toLocaleString(),
        service:true,
      };
      if(!account.email){
        addNotification(lang==='ar'?'تم إنشاء العامل، لكن استجابة الباك إند لا تحتوي على بريد الدخول.':'Worker was created, but the backend response did not include a login email.','warning');
        return;
      }
      const nextWorkers=[account,...workers.filter(w=>String(w.email||w.username).toLowerCase()!==String(account.email||account.username).toLowerCase())];
      setWorkers(nextWorkers); setStore('sph_worker_accounts',nextWorkers); setLastAccount(account);
      const nextSectors=sectors.map(s=>String(s.id)===String(account.sectorId)?{...s,workers:[...(s.workers||[]).filter(w=>String(w.id)!==String(account.id)),{id:account.id,name:account.name,role:account.jobTitle,phone:account.phone,email:account.email,status:account.status}]}:s);
      setSectors(nextSectors); setStore('sph_sectors',nextSectors);
      setForm({firstName:'',lastName:'',email:'',password:'',phone:'',address:'',jobTitle:'Plant Care Worker',sectorId:'',permissions:{tasks:true,alerts:true,upload:true,finishTask:true}});
      setShowWorkerModal(false);
      addNotification(lang==='ar'?'تم إنشاء حساب العامل بنجاح. استخدم نفس الإيميل وكلمة المرور لتسجيل الدخول.':'Worker account created successfully. Use the same email and password to sign in.','success');
    }catch(err){
      addNotification(userFriendlyServiceError(err,lang),'warning');
    }
  };
  const removeWorker=async(id)=>{
    if(!canManageWorkers){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    try{ await usersAPI.deleteWorker(id); }catch(err){ addNotification(serviceFriendlyError(err,lang),'warning'); }
    const next=workers.filter(w=>String(w.id)!==String(id)); setWorkers(next); setStore('sph_worker_accounts',next);
    const nextSectors=sectors.map(s=>({...s,workers:(s.workers||[]).filter(w=>String(w.id)!==String(id))})); setSectors(nextSectors); setStore('sph_sectors',nextSectors);
    if(String(lastAccount?.id)===String(id)) setLastAccount(null);
  };
  const toggleStatus=(worker)=>{
    const nextStatus = worker.status === 'Disabled' ? 'Active' : 'Disabled';
    const next=workers.map(w=>String(w.id)===String(worker.id)?{...w,status:nextStatus}:w);
    setWorkers(next); setStore('sph_worker_accounts',next);
    const nextSectors=sectors.map(s=>({...s,workers:(s.workers||[]).map(w=>String(w.id)===String(worker.id)?{...w,status:nextStatus}:w)}));
    setSectors(nextSectors); setStore('sph_sectors',nextSectors);
  };
  const assignWorker=async(worker, nextSectorId)=>{
    const target=sectors.find(sec=>String(sec.id)===String(nextSectorId));
    if(!target){ addNotification(lang==='ar'?'اختر قطاعًا صحيحًا من القائمة.':'Choose a valid sector from the list.','warning'); return; }
    try{
      if(worker.service && worker.id){
        await usersAPI.assignWorker(worker.id, { sectorId:target.id, assignedSectorId:target.id });
      }
    }catch(err){
      addNotification(userFriendlyServiceError(err,lang),'warning');
    }
    const nextWorkers=workers.map(w=>String(w.id)===String(worker.id)?{...w,sectorId:target.id,sector:target.name}:w);
    setWorkers(nextWorkers); setStore('sph_worker_accounts',nextWorkers);
    const nextSectors=sectors.map(s=>{
      const without=(s.workers||[]).filter(w=>String(w.id)!==String(worker.id));
      return String(s.id)===String(target.id) ? {...s,workers:[...without,{id:worker.id,name:worker.name,role:worker.jobTitle||worker.role,phone:worker.phone,email:worker.email,status:worker.status||'Active'}]} : {...s,workers:without};
    });
    setSectors(nextSectors); setStore('sph_sectors',nextSectors);
    setAssigningWorkerId(null); setAssignSectorId('');
    addNotification(lang==='ar'?'تم تعيين العامل للقطاع.':'Worker assigned to sector.','success');
  };
  const editWorker=(worker)=>{
    const parts=String(worker.name||'').trim().split(/\s+/);
    setForm({firstName:parts[0]||'',lastName:parts.slice(1).join(' ')||'',email:worker.email||'',password:'',phone:worker.phone||'',address:worker.address||'',jobTitle:worker.jobTitle||worker.role||'Plant Care Worker',sectorId:String(worker.sectorId||''),permissions:worker.permissions||{tasks:true,alerts:true,upload:true,finishTask:true}});
    setShowWorkerModal(true);
  };
  const copyText=(text)=>{ try{ navigator.clipboard.writeText(text); addNotification(lang==='ar'?'تم نسخ بيانات الحساب.':'Account data copied.','success'); }catch{} };
  const copyAllAccount=(acc)=>copyText(`${t('workerName')}: ${acc.name}\n${t('role')}: ${localizeValue(acc.jobTitle || acc.role,t)}\n${t('sector')}: ${localizeValue(acc.sector,t)}\n${t('accountStatus')}: ${localizeValue(acc.status,t)}\n${t('username')}: ${acc.username || acc.email}\n${t('password')}: ${acc.password}`);
  return <>
    <PageHead title={L.title} sub={L.subtitle} action={canManageWorkers?<button className="primary-btn worker-add-trigger" onClick={()=>{setForm({firstName:'',lastName:'',email:'',password:'',phone:'',address:'',jobTitle:'Plant Care Worker',sectorId:'',permissions:{tasks:true,alerts:true,upload:true,finishTask:true}}); setShowWorkerModal(true);}}><Plus size={18}/>{lang==='ar'?'إضافة عامل':'Add Worker'}</button>:null}/>
    {serviceState.error && <p className="form-helper warning-text">{serviceState.error}</p>}
    {serviceState.live && <p className="form-helper success-text">{lang==='ar'?'تم تحميل العاملين بنجاح.':'Workers loaded successfully.'}</p>}
    <section className="workers-page-v2">
      <details className="calm-details worker-summary-details">
        <summary><UserPlus size={18}/>{t('viewDetails')}</summary>
        <div className="worker-kpis-v2">
          <Kpi icon={<UserPlus/>} label={t('workers')} value={allWorkers.length}/>
          <Kpi icon={<ShieldCheck/>} label={lang==='ar'?'مديرين':'Managers'} value={allWorkers.filter(w=>String(w.jobTitle||w.role).includes('Manager')).length}/>
          <Kpi icon={<Layers3/>} label={t('sectors')} value={sectors.length}/>
          <Kpi icon={<CheckCircle2/>} label={t('tasks')} value={allWorkers.filter(w=>String(w.status||'Active')!=='Disabled').length}/>
        </div>
      </details>

      {canManageWorkers && showWorkerModal && <div className="modal-backdrop worker-add-modal-backdrop"><section className="modal-card worker-add-modal-card worker-create-clean-modal">
        <button className="modal-close" onClick={()=>setShowWorkerModal(false)}><X size={18}/></button>
        <header className="worker-create-modal-head">
          <h3>{lang==='ar'?'إضافة عامل جديد':'Add New Worker'}</h3>
          <p>{lang==='ar'?'أنشئ حساب عامل حقيقي مرتبط بالباك إند.':'Create a real backend worker account.'}</p>
        </header>
        <div className="worker-create-form-clean">
          <Field label={lang==='ar'?'الاسم الأول *':'First Name *'}><input value={form.firstName} onChange={e=>setForm({...form,firstName:e.target.value})} placeholder={lang==='ar'?'أحمد':'Ahmed'} autoComplete="given-name"/></Field>
          <Field label={lang==='ar'?'الاسم الأخير *':'Last Name *'}><input value={form.lastName} onChange={e=>setForm({...form,lastName:e.target.value})} placeholder={lang==='ar'?'حسن':'Hassan'} autoComplete="family-name"/></Field>
          <Field label={lang==='ar'?'البريد الإلكتروني *':'Email Address *'}><input type="email" value={form.email} onChange={e=>setForm({...form,email:e.target.value})} placeholder="worker@farm.com" autoComplete="off"/></Field>
          <Field label={lang==='ar'?'كلمة المرور *':'Password *'}><input type="password" value={form.password} onChange={e=>setForm({...form,password:e.target.value})} placeholder={lang==='ar'?'كلمة مرور العامل':'Secure password'} autoComplete="new-password"/></Field>
          <Field label={lang==='ar'?'رقم الهاتف':'Phone'}><input value={form.phone} onChange={e=>setForm({...form,phone:e.target.value})} placeholder="+20..." autoComplete="tel"/></Field>
          <Field label={lang==='ar'?'العنوان':'Address'}><input value={form.address} onChange={e=>setForm({...form,address:e.target.value})} placeholder={lang==='ar'?'المدينة...':'City...'} autoComplete="street-address"/></Field>
          <Field label={lang==='ar'?'تعيين إلى قطاع':'Assign to Sector'}><select value={form.sectorId} onChange={e=>setForm({...form,sectorId:e.target.value})}><option value="">{lang==='ar'?'— بدون تعيين الآن —':'— No assignment yet —'}</option>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}</option>)}</select></Field>
          <button className="primary-btn worker-create-submit clean-worker-submit" onClick={createWorker}><UserPlus size={18}/>{lang==='ar'?'إضافة العامل':'Add Worker'}</button>
        </div>
      </section></div>}

      {lastAccount && <section className="worker-credentials-card-v2 account-result-card clean-generated-account-card standalone-credentials"><div className="credentials-icon"><ShieldCheck size={30}/></div><h3>{t('workerAccountDataTitle')}</h3><div className="account-result-content"><div className="generated-account-summary"><div className="worker-avatar-v2 small"><span>{(lastAccount.name||'W').slice(0,1).toUpperCase()}</span></div><div className="generated-account-fields"><div><span>{t('workerName')}</span><b>{lastAccount.name}</b></div><div><span>{t('role')}</span><b>{localizeValue(lastAccount.jobTitle || lastAccount.role,t)}</b></div><div><span>{t('sector')}</span><b>{localizeValue(lastAccount.sector,t)}</b></div><div><span>{t('accountStatus')}</span><b>{localizeValue(lastAccount.status,t)}</b></div></div></div><div className="account-divider" /><div className="credentials-box modern clean-credentials-box"><div className="credential-line"><div><span>{t('username')}</span><code>{lastAccount.username || lastAccount.email}</code></div><button onClick={()=>copyText(lastAccount.username || lastAccount.email)}>{t('copyUsername')}</button></div><div className="credential-line"><div><span>{t('password')}</span><code>{lastAccount.password}</code></div><button onClick={()=>copyText(lastAccount.password)}>{t('copyPassword')}</button></div></div><button className="primary-btn wide" onClick={()=>copyAllAccount(lastAccount)}><FileText size={16}/>{t('copyAll')}</button></div></section>}

      <section className="workers-list-card-v2">
        <div className="compact-section-head"><div><h3>{L.workersList}</h3><p>{L.mobileHint}</p></div><span className="compact-head-icon"><UserPlus/></span></div>
        <div className="worker-cards-grid-v2">
          {allWorkers.length ? allWorkers.map(w=>{
            const isStored=workers.some(a=>String(a.id)===String(w.id));
            const initials=(w.name||'W').split(' ').filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase() || 'W';
            const status=String(w.status || 'Active');
            return <article className={`worker-card-v2 ${status==='Disabled'?'disabled':''}`} key={w.id || w.email}>
              <div className="worker-avatar-v2">{w.avatar?<img src={w.avatar} alt={w.name}/>:<span>{initials}</span>}</div>
              <div className="worker-info-v2">
                <div className="worker-title-row"><h3>{w.name}</h3><span className={`tag ${status==='Disabled'?'red':'green'}`}>{status==='Disabled'?L.disabled:L.active}</span></div>
                <div className="worker-meta-grid-v2">
                  <span><b>{t('email')}</b>{w.email || w.username || '-'}</span>
                  <span><b>{L.role}</b>{localizeValue(w.jobTitle || w.role,t)}</span>
                  <span><b>{L.phone}</b>{w.phone || '-'}</span>
                  <span><b>{L.sector}</b>{localizeValue(w.sector || selectedSector?.name,t)}</span>
                  <span><b>{L.lastTask}</b>{localizeValue(w.lastTask || 'Inspect leaves',t)}</span>
                </div>
              </div>
              <div className="worker-actions-v2">
                {canManageWorkers ? <><button className="secondary-btn" onClick={()=>editWorker(w)}><FileText size={15}/>{L.edit}</button>
                <button className="secondary-btn" onClick={()=>{setAssigningWorkerId(w.id); setAssignSectorId(String(w.sectorId || sectors[0]?.id || ''));}}><Layers3 size={15}/>{L.assign}</button>
                <button className="secondary-btn" onClick={()=>toggleStatus(w)}><X size={15}/>{status==='Disabled'?L.active:L.disable}</button>
                <button className="danger-soft-btn" onClick={()=>removeWorker(w.id)}><X size={15}/>{t('delete')}</button></> : <span className="manager-view-only"><Eye size={15}/>{t('workerLimitedView')}</span>}
              </div>
              {canManageWorkers && String(assigningWorkerId)===String(w.id) && <div className="inline-sector-assign">
                <Field label={t('selectSector')}><select value={assignSectorId || String(w.sectorId || sectors[0]?.id || '')} onChange={e=>setAssignSectorId(e.target.value)}>
                  {sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}{s.crop?` - ${localizeValue(s.crop,t)}`:''}{s.status?` (${labelStatus(s.status,t)})`:''}</option>)}
                </select></Field>
                <div className="inline-assign-actions"><button className="primary-btn" onClick={()=>assignWorker(w, assignSectorId || w.sectorId || sectors[0]?.id)}><Layers3 size={15}/>{L.assign}</button><button className="secondary-btn" onClick={()=>setAssigningWorkerId(null)}>{t('close')}</button></div>
              </div>}
            </article>}) : <EmptyActionState icon={Users} title={lang==='ar'?'لا يوجد عمال حتى الآن':'No workers yet'} description={lang==='ar'?'ابدأ بإضافة أول عامل واربطه بقطاع من المزرعة.':'Add your first worker and assign them to a farm sector.'} action={canManageWorkers ? <button className="primary-btn" onClick={()=>setShowWorkerModal(true)}><UserPlus size={16}/>{t('createNewWorker')}</button> : null}/>}
        </div>
      </section>
    </section>
  </>
}


function FarmSensorsPanel({sectors=[]}){
  const {t}=useLang();
  const [historyOpen,setHistoryOpen]=useState(false);
  const readings=getStore('sph_readings',[]).map((r,i)=>({...r,sector:r.sector || sectors[i%Math.max(sectors.length,1)]?.name || 'Greenhouse A'}));
  const sensorRows=sectors.map((sector,i)=>{
    const s=sector.sensors || {};
    return {
      id:sector.id,
      sector:sector.name,
      crop:sector.crop,
      status:sector.status,
      source: sector.source || (sector.serviceLive ? 'Live' : 'Simulation'),
      lastUpdated: sector.lastUpdated || readings[i]?.dateTime || readings[i]?.time || new Date().toLocaleString(),
      readings:{cropType:sector.crop, temperature:s.temperature, humidity:s.humidity, soilMoisture:s.soilMoisture, soilTemp:s.soilTemp, light:s.light}
    };
  });
  return <section className="compact-section farm-sensors-section">
    <div className="compact-section-head"><div><h3>{t('farmSensors')}</h3><p>{t('farmSensorsSub')}</p></div><span className="compact-head-icon"><ThermometerSun/></span></div>
    <div className="sensor-merge-notice"><CheckCircle2 size={18}/><span>{t('sensorsMergedNotice')}</span></div>
    <div className="sector-sensor-grid">
      {sensorRows.map(row=><article className={`sector-sensor-card ${tone(row.status)}`} key={row.id}>
        <header><div><b>{localizeValue(row.sector,t)}</b><small>{localizeValue(row.crop,t)} • {t('lastUpdated')}: {row.lastUpdated}</small></div><span className={`tag ${row.source==='Live'?'green':'amber'}`}>{row.source==='Live'?t('liveSource'):t('simulationSource')}</span></header>
        <ReadingGrid r={row.readings}/>
        <button className="secondary-btn wide" onClick={()=>setHistoryOpen(true)}><BarChart3 size={16}/>{t('viewSensorHistory')}</button>
      </article>)}
    </div>
    <p className="muted-copy compact-note">{t('sensorHistoryCompact')}</p>
    {historyOpen && <SensorHistoryModal readings={readings} onClose={()=>setHistoryOpen(false)} />}
  </section>
}

function MoreMenu(){
  const {t}=useLang();
  const tools=[
    {to:'/workers', icon:UserPlus, title:t('workers'), desc:t('workerDetailsModalNote'), show:hasPermission('workers') || hasPermission('workersView')},
    {to:'/farm-management', icon:FileText, title:t('farmReports'), desc:t('reportsMovedNotice'), show:hasPermission('reports')},
    {to:'/settings', icon:ShieldCheck, title:t('settings'), desc:t('settingsSubtitle') || t('profileInfo'), show:true},
    {to:'/chat', icon:Sparkles, title:t('askAssistant'), desc:t('assistantLearningNotice'), show:true},
    {to:'/farm-management', icon:ThermometerSun, title:t('farmSensors'), desc:t('sensorsMergedNotice'), show:hasPermission('sensors') || hasPermission('sectors')}
  ].filter(x=>x.show);
  return <>
    <PageHead title={t('moreMenuTitle')} sub={t('moreMenuSub')} action={<AccountRoleBadge/>}/>
    <div className="more-menu-grid">
      {tools.map(({to,icon:Icon,title,desc})=><NavLink className="more-tool-card" to={to} key={title}><span><Icon size={24}/></span><div><b>{title}</b><small>{desc}</small></div><ChevronRight size={18}/></NavLink>)}
    </div>
    <Panel title={t('sectionOrganizationSummary')}>
      <div className="organization-summary-grid">
        <Box label={t('removedFromMainMenu')} value={`${t('connectedSensors')} / ${t('diagnosisReportsMerged')}`}/>
        <Box label={t('mergedInsideFarmManagement')} value={`${t('devices')}, ${t('farmSensors')}, ${t('farmReports')}`}/>
        <Box label={t('movedInsideAssistant')} value={`${t('diseaseGuideInAssistant')}, ${t('recommendations')}`}/>
        <Box label={t('convertedToModal')} value={`${t('sensorHistory')}, ${t('deviceDetails')}, ${t('viewFullReport')}`}/>
        <Box label={t('mustStayVisible')} value={`${t('dashboard')}, ${t('diagnosis')}, ${t('farmManagementTitle')}, ${t('alerts')}`}/>
      </div>
    </Panel>
  </>;
}

function FarmManagement(){
  const {t, lang}=useLang();
  const nav=useNavigate();
  const [view,setView]=useState(isWorker() ? 'tasks' : 'overview');
  const [showAdd,setShowAdd]=useState(false);
  const [sectors,setSectors]=useState(getStore('sph_sectors',[]));
  const [serviceLive,setServiceLive]=useState(false);
  const [form,setForm]=useState({name:'',location:'',crop:''});
  const [workerForm,setWorkerForm]=useState({name:'',role:'Plant Care Worker',phone:'',sectorId:''});
  const workers=sectors.flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name})));
  const visibleSectors=isWorker()?sectors.filter(s=>String(s.id)===String(currentUser()?.sectorId) || String(s.name)===String(currentUser()?.sector) || (s.workers||[]).some(w=>String(w.email||'').toLowerCase()===currentUserEmail())):sectors;
  const devices=visibleSectors.flatMap(s=>(s.devices||[]).map(d=>({...d,sector:s.name,sectorId:s.id})));
  const tasks=getGlobalTasks().filter(task=>!isWorker() || taskAssignedToCurrent(task));
  const savedFarmSettings=getStore('sph_settings',{});
  const [adminFarm,setAdminFarm]=useState({farmName:savedFarmSettings.farmName || '', crop:savedFarmSettings.defaultCrop || '', location:savedFarmSettings.farmLocation || '', manager:savedFarmSettings.farmManager || workers[0]?.name || ''});
  useEffect(()=>{
    let mounted=true;
    const token=localStorage.getItem('ecosense_token');
    if(!token) return;
    sectorsAPI.getAll().then(({data})=>{
      const list = Array.isArray(data) ? data : (data.sectors || data.data || []);
      if(mounted && Array.isArray(list) && list.length){
        const normalized=list.map((item,idx)=>({
          id:item._id || item.id || idx+1,
          name:item.name || item.sectorName || `Sector ${idx+1}`,
          location:item.location || item.area || '-',
          crop:item.crop || item.cropType || item.plantType || '',
          status:item.status || item.final_status || item.healthStatus || '',
          sensors:item.sensors || {},
          diagnosis:item.diagnosis || {condition:item.status || '', disease:item.disease || '', confidence:item.confidence || 0, summary:item.summary || ''},
          workers:item.workers || [], tasks:item.tasks || [], notes:item.notes || [], rows:item.rows || [], equipment:item.equipment || [], devices:item.devices || []
        }));
        setSectors(normalized); setStore('sph_sectors',normalized); setServiceLive(true);
      }
    }).catch(()=>{});
    devicesAPI.getAll().then(({data})=>{
      const list = Array.isArray(data) ? data : (data.devices || data.data || []);
      if(mounted && list.length){
        const current=getStore('sph_sectors', []);
        const next=current.map(sec=>({...sec, devices:[...(sec.devices||[]), ...list.filter(d=>String(d.sectorId || d.sector || '')===String(sec.id) || String(d.sectorName || '')===String(sec.name)).map((d,i)=>({id:d._id||d.id||`${sec.id}-${i}`, name:d.name||d.deviceName||d.deviceSerial||'Device', type:d.type||d.deviceType||'Device', status:d.status||'-', value:d.lastReading?.value ?? d.value ?? '-', unit:d.lastReading?.unit ?? d.unit ?? '', lastContact:d.lastContact||d.updatedAt||'', battery:d.battery}))]}));
        setSectors(next); setStore('sph_sectors',next); setServiceLive(true);
      }
    }).catch(()=>{});
    return()=>{mounted=false};
  },[]);
  const ownerLabel = lang==='ar' ? 'مالك' : t('owner');
  const saveSector=async()=>{
    if(!form.name.trim()) return;
    try{
      const {data}=await sectorsAPI.create({name:form.name,location:form.location,...(form.crop ? {crop:form.crop,cropType:form.crop} : {})});
      const item=data.sector || data.data || data;
      const created={...item,id:item._id || item.id || Date.now(),name:item.name || form.name,location:item.location || form.location,crop:item.crop || item.cropType || form.crop,workers:item.workers || [],tasks:item.tasks || [],notes:item.notes || [],rows:item.rows || [],equipment:item.equipment || [],devices:item.devices || []};
      const next=[created,...sectors];
      setSectors(next); setStore('sph_sectors',next); setForm({name:'',location:'',crop:''}); setShowAdd(false); setView('sectors'); setServiceLive(true);
      addNotification(lang==='ar'?'تم حفظ القطاع في الباك إند.':'Sector saved in backend.','success');
    }catch(err){ addNotification(userFriendlyServiceError(err,lang),'danger'); }
  };
  const saveAdminFarm=()=>{
    const current=getStore('sph_settings',{});
    setStore('sph_settings',{...current,farmName:adminFarm.farmName,defaultCrop:adminFarm.crop,farmLocation:adminFarm.location,farmManager:adminFarm.manager});
    addNotification(lang==='ar'?'تم تحديث بيانات المزرعة من لوحة الإدارة.':'Farm data updated from admin dashboard.','success');
  };
  const deleteSectorAdmin=async(id)=>{
    if(!hasPermission('deleteSector')){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    try{ await sectorsAPI.delete(id); setServiceLive(true); }
    catch(err){ addNotification(userFriendlyServiceError(err,lang),'danger'); return; }
    const next=sectors.filter(s=>String(s.id)!==String(id));
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم حذف القطاع من الباك إند.':'Sector deleted from backend.','success');
  };
  const editSectorAdmin=async(sector)=>{
    const name=prompt(lang==='ar'?'اسم القطاع الجديد':'New sector name', sector.name);
    if(!name) return;
    const crop=prompt(lang==='ar'?'نوع المحصول':'Crop type', sector.crop) || sector.crop;
    const location=prompt(lang==='ar'?'بيانات أو موقع القطاع':'Sector details or location', sector.location) || sector.location;
    try{ await sectorsAPI.update(sector.id || sector._id, {name,location,...(crop ? {crop,cropType:crop} : {})}); setServiceLive(true); }
    catch(err){ addNotification(userFriendlyServiceError(err,lang),'danger'); return; }
    const next=sectors.map(s=>String(s.id)===String(sector.id)?{...s,name,crop,location}:s);
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم تعديل بيانات القطاع في الباك إند.':'Sector updated in backend.','success');
  };
  const assignAdminWorkerToSector=(sector)=>{
    const workerName=prompt(lang==='ar'?'اكتب اسم العامل أو المدير للقطاع':'Enter worker or manager name for this sector', workers[0]?.name || adminFarm.manager || '');
    if(!workerName) return;
    const newWorker={id:Date.now(),name:workerName,role:'Farm Manager',phone:'-',status:'Active'};
    const next=sectors.map(s=>String(s.id)===String(sector.id)?{...s,workers:[newWorker,...(s.workers||[])]}:s);
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم تعيين مسؤول للقطاع.':'Sector manager assigned.','success');
  };
  const addWorkerToSector=async()=>{
    if(!isOwner() || !workerForm.name.trim()) return;
    const targetId = Number(workerForm.sectorId || sectors[0]?.id);
    let newWorker={...workerForm,id:Date.now(),sectorId:targetId};
    try{
      const {data}=await usersAPI.addWorker({name:workerForm.name,role:workerForm.role,phone:workerForm.phone,sectorId:targetId});
      const item=data.worker || data.user || data.data || data;
      newWorker={...newWorker,...item,id:item._id || item.id || newWorker.id};
      setServiceLive(true);
    }catch(err){ addNotification(userFriendlyServiceError(err,lang),'danger'); return; }
    const next = sectors.map(s=>Number(s.id)===targetId ? {...s, workers:[...(s.workers||[]), newWorker]} : s);
    setSectors(next); setStore('sph_sectors',next); setWorkerForm({name:'',role:'Plant Care Worker',phone:'',sectorId:targetId});
  };
  const deleteWorker=async(workerId)=>{
    if(!isOwner()) return;
    const token=localStorage.getItem('ecosense_token');
    if(token){ try{ await usersAPI.deleteWorker(workerId); setServiceLive(true); } catch(err){ /* local delete fallback */ } }
    const next = sectors.map(s=>({...s, workers:(s.workers||[]).filter(w=>w.id!==workerId)}));
    setSectors(next); setStore('sph_sectors',next);
  };
  const farmLabels = lang === 'ar'
    ? { sections:'أقسام إدارة المزرعة', sectionsSub:'نظرة منظمة على القطاعات، الأجهزة، الحساسات، الكاميرا، العمال، التنبيهات، والمهام.', overview:'Overview', map:'Sectors Map', details:'Sector Details', sensors:'Sensors', devices:'Devices', camera:'Camera', workers:'Workers', alerts:'Alerts', tasks:'Tasks' }
    : { sections:'Farm Management Sections', sectionsSub:'A structured view for sectors, devices, sensors, camera, workers, alerts, and tasks.', overview:'Overview', map:'Sectors Map', details:'Sector Details', sensors:'Sensors', devices:'Devices', camera:'Camera', workers:'Workers', alerts:'Alerts', tasks:'Tasks' };
  const cards=[
    {key:'overview', icon:Layers3, label:t('farmOverview'), count:serviceLive?t('apiConnected'):t('apiFallback'), hint:t('sectorsDevicesStatus')},
    {key:'sectors', icon:Sprout, label:t('sectors'), count:visibleSectors.length, hint:t('sectorHealth')},
    {key:'devices', icon:Cpu, label:t('devices'), count:devices.length, hint:t('devicesAndSensors')},
    {key:'sensors', icon:ThermometerSun, label:t('farmSensors'), count:visibleSectors.length, hint:t('farmSensorsSub')},
    {key:'tasks', icon:CheckCircle2, label:t('farmTasks'), count:tasks.length, hint:t('careTasks')},
    {key:'reports', icon:FileText, label:t('farmReports'), count:getStore('sph_readings',[]).length, hint:t('reportsManagement')}
  ].filter(card => isOwner() || (isFarmManager() && ['overview','sectors','devices','sensors','tasks','reports'].includes(card.key)) || (isWorker() && ['sectors','tasks'].includes(card.key)));
  return <div className="farm-slim-page">
    <PageHead title={t('farmManagementTitle')} sub={t('farmManagementSub')} action={<div className="farm-head-actions">{isOwner()?<button className="farm-add-fab" onClick={()=>setShowAdd(!showAdd)}><Plus size={18}/>{t('addSector')}</button>:<AccountRoleBadge/>}<button className="assistant-mini-btn" type="button" onClick={()=>window.dispatchEvent(new Event('ecosense-open-assistant'))} title={t('openAssistant')}><Sparkles size={18}/></button></div>}/>

    <section className="farm-management-overview-note">
      <div><span><Layers3 size={16}/>{farmLabels.sections}</span><p>{farmLabels.sectionsSub}</p></div>
      <div className="farm-section-pills"><b>{farmLabels.overview}</b><b>{farmLabels.map}</b><b>{farmLabels.sensors}</b><b>{farmLabels.devices}</b><b>{farmLabels.camera}</b><b>{farmLabels.workers}</b><b>{farmLabels.reports || t('farmReports')}</b><b>{farmLabels.tasks}</b></div>
    </section>

    <section className="farm-service-row farm-service-row-polished">
      {cards.map(({key,icon:Icon,label,count,hint})=><button key={key} className={`farm-service-card ${key} ${view===key?'active':''}`} onClick={()=>setView(key)}>
        <span className="farm-service-icon"><Icon size={28}/></span>
        <span className="farm-service-text"><b>{label}</b><small>{hint}</small></span>
        <strong>{count}</strong>
        <i className="farm-card-arrow"><ChevronRight size={18}/></i>
      </button>)}
    </section>

    {isOwner() && showAdd && <section className="farm-add-panel">
      <div className="farm-add-head"><b>{t('addSector')}</b><button onClick={()=>setShowAdd(false)}><X size={16}/></button></div>
      <div className="farm-add-grid">
        <Field label={t('sectorName')}><input value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></Field>
        <Field label={t('location')}><input value={form.location} onChange={e=>setForm({...form,location:e.target.value})}/></Field>
        <Field label={t('crop')}><CropTypeSelect lang={langFromDocument()} value={form.crop || ''} onChange={value=>setForm({...form,crop:value})} /></Field>
        <button className="primary-btn" onClick={saveSector}><Plus size={18}/>{t('save')}</button>
      </div>
    </section>}

    {view==='overview' && <section className="farm-production-overview">
      <div className="farm-overview-hero"><Layers3 size={42}/><div><h3>{t('sectorsDevicesStatus')}</h3><p>{t('farmManagementSub')}</p></div></div>
      <div className="farm-overview-metrics">
        <Box label={t('totalSectors')} value={visibleSectors.length}/>
        <Box label={t('devices')} value={devices.length}/>
        <Box label={t('latestDiagnosis')} value={labelStatus(sectors[0]?.status,t)}/>
        <Box label={t('apiStatus')} value={serviceLive?t('apiConnected'):t('apiFallback')}/>
      </div>
      <div className="farm-management-section-grid">
        <div><b>{farmLabels.map}</b><span>{visibleSectors.length} {t('sectors')}</span></div>
        <div><b>{farmLabels.camera}</b><span>{lang==='ar'?'آخر صورة لكل قطاع عند توفرها':'Latest image per sector when available'}</span></div>
        <div><b>{farmLabels.reports || t('farmReports')}</b><span>{getStore('sph_readings',[]).length} {t('reports')}</span></div>
        <div><b>{farmLabels.tasks}</b><span>{tasks.length} {t('tasks')}</span></div>
      </div>
    </section>}

    {view==='sectors' && <section className="compact-section">
      <div className="compact-section-head"><div><h3>{t('sectors')}</h3><p>{lang==='ar'?'اضغط على أي قطاع للدخول لتفاصيله.':'Open any sector to view details.'}</p></div>{isOwner() && <button className="icon-add-btn" onClick={()=>setShowAdd(true)}><Plus size={20}/></button>}</div>
      <div className="compact-sector-grid">
        {visibleSectors.map(s=><button key={s.id} className={`compact-sector-icon ${tone(s.status)}`} onClick={()=>nav(`/sectors/${s.id}`)}>
          <span><Sprout size={28}/></span>
          <b>{localizeValue(s.name,t)}</b>
          <small>{localizeValue(s.crop,t)} • {labelStatus(s.status,t)}</small>
          <em>{Math.round((s.diagnosis?.confidence ?? .8)*100)}%</em>
        </button>)}
      </div>
    </section>}

    {view==='devices' && <DeviceInventory devices={devices} sectors={visibleSectors}/>}
    {view==='sensors' && <FarmSensorsPanel sectors={visibleSectors}/>}
    {view==='tasks' && (isWorker()?<MyTasks embedded/>:<CareTasksPanel tasks={tasks} workers={workers} sectors={visibleSectors}/>)}
    {view==='reports' && <FarmReportsPanel sectors={visibleSectors}/>}
  </div>
}

function CompactList({title,icon,items,empty,render}){
  const {t}=useLang();
  return <section className="compact-section">
    <div className="compact-section-head"><div><h3>{title}</h3><p>{items.length} {t('items')}</p></div><span className="compact-head-icon">{icon}</span></div>
    <div className="compact-list-grid">
      {items.length ? items.map((item,i)=><article className="compact-list-card" key={item.id||i}>{render(item)}</article>) : <div className="farm-empty-focus small"><p>{empty}</p></div>}
    </div>
  </section>
}
function WorkersPage({embedded=false}){
  const {t, lang}=useLang();
  const sectors=getStore('sph_sectors',[]);
  const accountWorkers=workerAccounts().map(w=>({
    ...w,
    sector:w.assignedSectorName || w.sector || w.assignedSector || '-',
    role:w.role || w.jobTitle || 'Worker'
  }));
  const sectorWorkers=sectors.flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name,sectorId:s.id,role:w.role || w.jobTitle || 'Worker'})));
  const workers=[...accountWorkers,...sectorWorkers.filter(w=>!accountWorkers.some(a=>String(a.email||a.username||a.id||a.name)===String(w.email||w.username||w.id||w.name)))];
  const yes=<span className="permission-dot yes">✓</span>;
  const no=<span className="permission-dot no">×</span>;
  const permissionCell=(worker,key)=>{
    const roleText=String(worker.role || worker.jobTitle || '').toLowerCase();
    const perms=worker.permissions || {};
    const manager=roleText.includes('manager') || roleText.includes('مدير');
    const owner=roleText.includes('owner') || roleText.includes('مالك');
    if(key==='view') return perms.view===false?no:yes;
    if(owner) return yes;
    if(manager && ['add','edit'].includes(key)) return yes;
    return perms[key] ? yes : no;
  };
  return <>
    {!embedded && <PageHead title={t('workers')} sub={lang==='ar'?'إدارة العاملين وصلاحياتهم والقطاعات المرتبطين بها.':'Manage workers, permissions, and assigned sectors.'}/>}
    <Panel title={t('workers')}>
      <div className="table-wrap compact-admin-table permissions-owner-table workers-page-permissions-table">
        <table>
          <thead><tr><th>{t('workerName')}</th><th>{t('workerRole')}</th><th>{t('sector')}</th><th>{lang==='ar'?'عرض':'View'}</th><th>{lang==='ar'?'إضافة':'Add'}</th><th>{lang==='ar'?'تعديل':'Edit'}</th><th>{lang==='ar'?'حذف':'Delete'}</th></tr></thead>
          <tbody>{workers.length?workers.map(w=><tr key={w.email || w.username || w.id || w.name}>
            <td><b>{w.name || w.fullName || w.email || w.username}</b></td>
            <td>{translateModelText(w.role || w.jobTitle || 'Worker',t)}</td>
            <td>{translateModelText(w.sector || w.assignedSectorName || w.assignedSector || '-',t)}</td>
            <td>{permissionCell(w,'view')}</td>
            <td>{permissionCell(w,'add')}</td>
            <td>{permissionCell(w,'edit')}</td>
            <td>{permissionCell(w,'delete')}</td>
          </tr>):<tr><td colSpan="7" className="muted-cell">{lang==='ar'?'لا يوجد عاملون مسجلون حاليًا.':'No workers registered yet.'}</td></tr>}</tbody>
        </table>
      </div>
    </Panel>
  </>
}

function DevicesPage({embedded=false}){
  const {t,lang}=useLang();
  const sectors=getStore('sph_sectors',[]);
  const localDevices=sectors.flatMap(s=>(s.devices||[]).map(d=>({...d,sector:s.name,sectorId:s.id})));
  const [devices,setDevices]=useState(localDevices);
  const [msg,setMsg]=useState('');
  useEffect(()=>{
    let mounted=true;
    devicesAPI.getAll().then(({data})=>{
      const list=asArray(data,['devices','items']);
      if(!mounted || !list.length) return;
      const mapped=list.map((d,i)=>({
        id:d._id || d.id || i+1,
        name:d.name || d.deviceName || `Device ${i+1}`,
        type:d.type || d.deviceType || 'Sensor Device',
        value:d.value ?? d.lastValue ?? d.lastReading ?? '-', unit:d.unit || '',
        status:d.status || (d.online===false?'Offline':'Online'),
        sector:d.sectorName || d.sector || d.sectorId || '-', sectorId:d.sectorId || d.sector?._id
      }));
      setDevices(mapped); setMsg(lang==='ar'?'تم تحميل الأجهزة من الخدمة.':'Devices loaded from service.');
    }).catch(err=>setMsg(serviceFriendlyError(err,lang)));
    return()=>{mounted=false};
  },[lang]);
  const remove=async(id)=>{
    if(!hasPermission('deleteDevice')) return addNotification(t('ownerSensitiveBlocked'),'warning');
    try{ await devicesAPI.delete(id); }catch(err){ addNotification(serviceFriendlyError(err,lang),'warning'); }
    setDevices(devices.filter(d=>String(d.id)!==String(id)));
  };
  return <>{!embedded && <PageHead title={t('devices')} sub={t('sectorSensors')}/>} {msg && <p className={`form-helper ${msg.includes('تعذر')||msg.includes('Could not')?'warning-text':'success-text'}`}>{msg}</p>}<Panel title={t('devices')}><div className="table-wrap"><table><thead><tr><th>{t('sensorName')}</th><th>{t('sensorType')}</th><th>{t('sensorValue')}</th><th>{t('status')}</th><th>{t('sector')}</th><th>{t('actions')}</th></tr></thead><tbody>{devices.map(d=><tr key={d.id}><td>{localizeValue(d.name,t)}</td><td>{localizeValue(d.type,t)}</td><td>{localizeValue(d.value,t)}{d.unit}</td><td>{localizeValue(d.status,t)}</td><td>{localizeValue(d.sector,t)}</td><td>{hasPermission('deleteDevice')?<button className="danger-soft-btn" onClick={()=>remove(d.id)}>{t('delete')}</button>:<span>{t('workerLimitedView')}</span>}</td></tr>)}</tbody></table></div></Panel></>
}


function PlantLibrary(){
  const {t}=useLang();
  const diseases=[
    {name:t('leafSpotFungal'), symptoms:t('diseaseLeafSpotSymptoms'), causes:t('diseaseLeafSpotCauses'), solution:t('diseaseLeafSpotSolution'), prevention:t('diseaseLeafSpotPrevention')},
    {name:t('generalVisualStress'), symptoms:t('diseaseVisualStressSymptoms'), causes:t('diseaseVisualStressCauses'), solution:t('diseaseVisualStressSolution'), prevention:t('diseaseVisualStressPrevention')},
    {name:t('severeEnvironmentalStress'), symptoms:t('diseaseSevereStressSymptoms'), causes:t('diseaseSevereStressCauses'), solution:t('diseaseSevereStressSolution'), prevention:t('diseaseSevereStressPrevention')}
  ];
  return <>
    <PageHead title={t('diseaseGuide')} sub={t('libraryPurpose')}/>
    <div className="library-warning clean"><Leaf size={20}/>{t('modelDisclaimer')}</div>
    <div className="library-grid clean-library">
      {diseases.map((d,i)=><article className="library-card" key={i}>
        <div className="library-icon"><Leaf size={28}/><span>{i+1}</span></div>
        <h3>{d.name}</h3>
        <p><b>{t('diseaseSymptoms')}:</b> {d.symptoms}</p>
        <p><b>{t('diseaseCauses')}:</b> {d.causes}</p>
        <p><b>{t('diseaseInitialSolution')}:</b> {d.solution}</p>
        <p><b>{t('diseasePrevention')}:</b> {d.prevention}</p>
      </article>)}
    </div>
  </>
}

function WorkerTaskCard({task,sector,updateSector,t}){
  const [note,setNote]=useState(task.note||'');
  const [proof,setProof]=useState(task.proof||'');
  const choose=e=>{const f=e.target.files?.[0]; if(!f) return; const reader=new FileReader(); reader.onload=()=>setProof(reader.result); reader.readAsDataURL(f);};
  const setStatus=status=>{const next={...sector,tasks:(sector.tasks||[]).map(y=>y.id===task.id?{...y,status,note,proof}:y)}; updateSector(next); if(status==='done') addNotification(t('ownerNotifiedDone'),'good');};
  return <div className={`task-item ${String(task.priority).toLowerCase()} ${task.status||'todo'}`}>
    <CheckCircle2 size={18}/>
    <b>{localizeValue(task.name||task.title,t)}</b>
    <small>{task.assignedTo ? `${t('assignedTo')}: ${task.assignedTo}` : t('workerOnlyTasks')}</small>
    <span>{task.status==='done'?t('taskCompleted'):task.status==='inProgress'?t('taskInProgress'):t('taskNotStarted')}</span>
    {proof && <img src={proof} className="proof-thumb"/>}
    <label className="proof-upload"><Camera size={16}/>{t('proofUpload')}<input hidden type="file" accept="image/*" onChange={choose}/></label>
    <input value={note} onChange={e=>setNote(e.target.value)} placeholder={t('proofNote')}/>
    <button onClick={()=>setStatus('inProgress')}>{t('startTask')}</button>
    <button onClick={()=>setStatus('done')}>{t('completeAndNotifyOwner')}</button>
  </div>
}

function Reports(){ 
  const {t,lang}=useLang(); 
  const localReadings=getStore('sph_readings',[]); 
  const sectors=getStore('sph_sectors',[]);
  const [status,setStatus]=useState('all');
  const [sector,setSector]=useState('all');
  const [date,setDate]=useState('');
  const [serviceRows,setServiceRows]=useState([]);
  const [serviceStats,setServiceStats]=useState(null);
  const [serviceError,setServiceError]=useState('');
  const [loading,setLoading]=useState(false);

  const normalizeReportRow=(r,i)=>({
    ...r,
    id:r._id || r.id || i,
    sector: r.sectorName || r.sector || r.sectorId || sectors[i % Math.max(sectors.length,1)]?.name || '-',
    date: r.date || String(r.createdAt || r.timestamp || new Date().toISOString()).slice(0,10),
    time: r.time || new Date(r.createdAt || r.timestamp || Date.now()).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),
    cropType:r.cropType || r.crop || r.plantType || '-',
    final_status: r.final_status || r.finalStatus || r.status || '-',
    diagnosis: r.diagnosis?.explanation || r.diagnosis || r.summary || '-',
    recommendations: r.recommendations || r.actions || (r.solution ? [r.solution] : []),
    actions: r.actions || [],
    image: r.image || r.imageUrl || r.photoUrl || '',
    workerTask: r.workerTask || r.task || '-' 
  });

  const loadStats=async()=>{
    setLoading(true); setServiceError('');
    try{
      const sectorId=sector==='all'?'':sector;
      const {data}=await reportsAPI.getStats(sectorId,7);
      const payload=data?.data || data?.stats || data;
      setServiceStats(payload);
      const rows=asArray(payload,['rows','reports','items','history','diagnoses']).map(normalizeReportRow);
      if(rows.length) setServiceRows(rows);
    }catch(err){ setServiceError(serviceFriendlyError(err,lang)); }
    finally{ setLoading(false); }
  };
  useEffect(()=>{ loadStats(); },[lang]);

  const sourceRows=(serviceRows.length?serviceRows:localReadings.map(normalizeReportRow));
  const reportRows=sourceRows.filter(r => (status==='all' || r.final_status===status) && (sector==='all' || String(r.sector)===String(sector) || String(r.sectorId)===String(sector)) && (!date || r.date===date));
  const total=Number(serviceStats?.total ?? serviceStats?.totalReports ?? reportRows.length);
  const healthy=Number(serviceStats?.healthy ?? reportRows.filter(r=>r.final_status==='Healthy').length);
  const warning=Number(serviceStats?.moderate ?? serviceStats?.warning ?? reportRows.filter(r=>String(r.final_status).includes('Moderate')).length);
  const danger=Number(serviceStats?.high ?? serviceStats?.critical ?? reportRows.filter(r=>isDanger(r.final_status)).length);
  const preview=reportRows[0] || sourceRows[0];
  const csv=async()=>{
    try{
      const sectorId=sector==='all'?'':sector;
      const res=await reportsAPI.exportCSV(sectorId);
      const blob=res.data instanceof Blob ? res.data : new Blob([JSON.stringify(res.data,null,2)],{type:'application/json'});
      const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=blob.type.includes('json')?'ecosense-report.json':'ecosense-report.csv'; a.click();
      addNotification(lang==='ar'?'تم تحميل التقرير بنجاح.':'Report exported successfully.','success');
    }catch(err){
      const header=['time','sector','cropType','final_status','temperature','humidity','soilMoisture','soilTemp','light','diagnosis','recommendations','actions','workerTask'];
      const lines=[header.join(','),...reportRows.map(r=>header.map(k=>`"${String(Array.isArray(r[k])?r[k].join(' | '):(r[k]??'')).replaceAll('"','""')}"`).join(','))];
      download('ecosense-diagnosis-report.csv', lines.join('\n'), 'text/csv');
      addNotification(serviceFriendlyError(err,lang),'warning');
    }
  };
  const pdf=()=>{ openProfessionalReport({ t, lang, rows: reportRows, titleKey:'farmReports' }); };
  return <>
    <PageHead title={t('farmReports')} sub={t('reportsCombinedNotice')} action={<div className="report-actions"><button className="secondary-btn" onClick={loadStats} disabled={loading}><Activity size={18}/>{t('refreshReadings')}</button><button className="secondary-btn" onClick={pdf}><FileText size={18}/>{t('exportPdf')}</button><button className="primary-btn" onClick={csv}><BarChart3 size={18}/>{t('exportCsv')}</button></div>}/>
    {serviceError && <p className="form-helper warning-text">{serviceError}</p>}
    {serviceRows.length>0 && <p className="form-helper success-text">{lang==='ar'?'تم تحميل التقارير بنجاح.':'Reports loaded successfully.'}</p>}
    <div className="report-filter-panel panel">
      <Field label={t('filterByDate')}><input type="date" value={date} onChange={e=>setDate(e.target.value)}/></Field>
      <Field label={t('filterByStatus')}><select value={status} onChange={e=>setStatus(e.target.value)}><option value="all">{t('filterByStatusAll')}</option><option>Healthy</option><option>Moderate Stress</option><option>High Stress</option></select></Field>
      <Field label={t('filterBySector')}><select value={sector} onChange={e=>setSector(e.target.value)}><option value="all">{t('allSectors')}</option>{sectors.map(s=><option key={s.id} value={s.id || s.name}>{localizeValue(s.name,t)}</option>)}</select></Field>
    </div>
    <div className="kpis report-kpis">
      <Kpi icon={<FileText/>} label={t('diagnosisReport')} value={total}/>
      <Kpi icon={<Leaf/>} label={t('healthy')} value={healthy} tone="green"/>
      <Kpi icon={<AlertTriangle/>} label={t('monitor')} value={warning} tone="amber"/>
      <Kpi icon={<Bell/>} label={t('critical')} value={danger} tone="red"/>
    </div>
    <div className="grid-2 reports-production-grid">
      <Panel title={t('reportPreview')}>
        {preview ? <article className={`report-preview-card compact ${tone(preview.final_status)}`}>
          <div className="report-plant-image compact">{preview.image ? <img src={preview.image} alt={t('plantPhoto')}/> : <Leaf size={42}/>}</div>
          <div className="report-preview-body compact">
            <div className="report-preview-topline"><span className={`tag ${tone(preview.final_status)}`}>{labelStatus(preview.final_status,t)}</span><small>{t('diagnosisTime')}: {preview.date} {preview.time}</small></div>
            <h3>{localizeValue(preview.sector || sectors[0]?.name,t)}</h3>
            <p className="report-diagnosis-short"><b>{t('diagnosisSummary')}:</b> {translateModelText(String(preview.diagnosis || '').slice(0,120),t)}{String(preview.diagnosis||'').length>120?'...':''}</p>
            <div className="report-mini-readings">
              <span>{t('temperature')}: <b>{formatInputValue('temperature',preview.temperature)}</b></span>
              <span>{t('humidity')}: <b>{formatInputValue('humidity',preview.humidity)}</b></span>
              <span>{t('soilMoisture')}: <b>{formatInputValue('soilMoisture',preview.soilMoisture)}</b></span>
            </div>
            <div className="report-preview-actions"><button className="secondary-btn" onClick={()=>alert(`${t('finalStatus')}: ${labelStatus(preview.final_status,t)}\n${t('diagnosisSummary')}: ${translateModelText(preview.diagnosis,t)}`)}><Eye size={16}/>{lang==='ar'?'عرض التقرير كامل':'View Full Report'}</button><button className="primary-btn" onClick={pdf}><FileText size={16}/>{t('exportPdf')}</button></div>
          </div>
        </article> : <p className="muted-copy">{t('emptyState')}</p>}
      </Panel>
      <details className="calm-details report-chart-details">
        <summary><BarChart3 size={18}/>{t('statistics')}</summary>
        <Panel title={t('statistics')}>
          <ResponsiveContainer width="100%" height={240}><BarChart data={reportRows.slice().reverse()}><CartesianGrid vertical={false} stroke="var(--line)"/><XAxis dataKey="time"/><YAxis/><Tooltip/><Bar dataKey="soilMoisture" fill="var(--primary)" radius={[12,12,0,0]}/><Bar dataKey="temperature" fill="var(--amber)" radius={[12,12,0,0]}/></BarChart></ResponsiveContainer>
        </Panel>
      </details>
    </div>
    <div className="mobile-report-cards">
      {reportRows.map(r=><article className="report-row-card" key={r.id}><span className={`tag ${tone(r.final_status)}`}>{labelStatus(r.final_status,t)}</span><b>{localizeValue(r.sector,t)}</b><small>{r.date} {r.time}</small><p>{translateModelText(r.diagnosis,t)}</p></article>)}
    </div>
  </> 
}
function Alerts(){ 
  const {t,lang}=useLang(); 
  const nav=useNavigate();
  const [notes,setNotes]=useState(notificationStore().map(n=>({...n,read:n.read||false, local:true})));
  const [loading,setLoading]=useState(false);
  const [serviceError,setServiceError]=useState('');
  const sectors=getStore('sph_sectors',[]);
  const risky=getStore('sph_readings',[]).filter(r=>r.final_status!=='Healthy');

  const mapServiceNotification=(n,i)=>({
    id:n._id || n.id || n.notificationId || `service-${i}`,
    service:true,
    type:n.category || n.type || (String(n.severity||'').toLowerCase()==='high'?'criticalAlerts':'warningAlerts'),
    reason:n.message || n.title || n.body || n.reason || 'Notification',
    sector:n.sectorName || n.sector || n.sectorId || sectors[i%Math.max(sectors.length,1)]?.name || '-',
    time:n.createdAt || n.updatedAt || n.time || new Date().toLocaleString(),
    severity:n.severity || (n.type==='danger'?'High':'Medium'),
    read:!!(n.read || n.isRead),
    raw:n
  });

  const loadNotifications=async()=>{
    setLoading(true); setServiceError('');
    try{
      const {data}=await notificationsAPI.getAll({ force: true });
      const list=asArray(data,['notifications','items']);
      const mapped=list.map(mapServiceNotification);
      if(mapped.length){
        setNotes(mapped);
        setNotificationStore(mapped.map(n=>({id:n.id,time:n.time,message:n.reason,type:n.severity==='High'?'danger':'warning',read:n.read,service:true})));
      }
    }catch(err){ setServiceError(serviceFriendlyError(err,lang)); }
    finally{ setLoading(false); }
  };

  useEffect(()=>{ loadNotifications(); },[lang]);

  const safeNotes = Array.isArray(notes) ? notes : [];
  const localAlerts=safeNotes.map((n,i)=>({
    id:n.id,type:n.type==='danger'?'criticalAlerts':(String(n.type).includes('Alerts')?n.type:'warningAlerts'),reason:n.message || n.reason,sector:n.sector || sectors[i%Math.max(sectors.length,1)]?.name,time:n.time,severity:n.severity || (n.type==='danger'?'High':'Medium'),read:n.read,service:n.service
  }));
  const typedAlerts=[
    ...localAlerts,
    ...risky.map((r,i)=>({id:`r-${r.id}`,type:isDanger(r.final_status)?'criticalAlerts':'warningAlerts',reason:labelStatus(r.final_status,t),sector:sectors[i%Math.max(sectors.length,1)]?.name,time:r.time,severity:isDanger(r.final_status)?'High':'Medium',reading:r,localGenerated:true})),
  ];
  const counts=['criticalAlerts','warningAlerts','deviceAlerts','diseaseAlerts','sensorOfflineAlerts'].map(k=>[k,typedAlerts.filter(a=>a.type===k).length]);
  const markOne=async(a)=>{
    if(a.localGenerated) return;
    try{ if(a.service) await notificationsAPI.markRead(a.id); }catch(err){ addNotification(serviceFriendlyError(err,lang),'warning'); }
    const all=safeNotes.map(n=>String(n.id)===String(a.id)?{...n,read:true}:n); setNotes(all); setNotificationStore(all);
  };
  const deleteOne=async(a)=>{
    if(!window.confirm(lang==='ar'?'حذف هذا الإشعار؟':'Delete this notification?')) return;
    try{ if(a.service) await notificationsAPI.delete(a.id); }catch(err){ addNotification(serviceFriendlyError(err,lang),'warning'); }
    const all=safeNotes.filter(n=>String(n.id)!==String(a.id)); setNotes(all); setNotificationStore(all);
  };
  const markAll=async()=>{
    await Promise.allSettled(safeNotes.filter(n=>n.service && !n.read).map(n=>notificationsAPI.markRead(n.id)));
    const all=safeNotes.map(n=>({...n,read:true})); setNotes(all); setNotificationStore(all);
  };
  const deleteAll=async()=>{
    if(!typedAlerts.length) return;
    if(!window.confirm(lang==='ar'?'حذف كل الإشعارات؟':'Delete all notifications?')) return;
    await Promise.allSettled(safeNotes.filter(n=>n.service).map(n=>notificationsAPI.delete(n.id)));
    setNotes([]); setNotificationStore([]);
    addNotification(lang==='ar'?'تم حذف كل الإشعارات.':'All notifications deleted.','warning');
  };
  const createTask=(a)=>{
    const workers=workerAccounts();
    const worker=workers[0] || {};
    const next={id:Date.now(),title:a.reason,details:a.reason,status:'pending',priority:a.severity,assignedTo:worker.name || 'Plant Care Worker',assignedToEmail:worker.email || worker.username || '',createdAt:new Date().toLocaleString(),dueDate:new Date(Date.now()+86400000).toISOString().slice(0,10),sector:a.sector};
    setGlobalTasks([next,...getGlobalTasks()]);
    if(next.assignedToEmail) addUserNotification(next.assignedToEmail, taskToastMessage('created',lang), 'success');
    addNotification(t('treatmentTaskCreated'),'success');
  };
  return <>
    <PageHead title={t('alertsTitle')} sub={t('alertsSubtitle')} action={<div className="report-actions"><button className="secondary-btn" onClick={loadNotifications} disabled={loading}><Activity size={16}/>{t('refreshReadings')}</button><button className="secondary-btn" onClick={markAll}>{t('markAllRead')}</button><button className="danger-soft-btn" onClick={deleteAll}>{lang==='ar'?'حذف كل الإشعارات':'Delete all notifications'}</button></div>}/>
    {serviceError && <p className="form-helper warning-text">{serviceError}</p>}
    <div className="notice-types pro-alert-types">
      {counts.map(([k,c])=><span key={k}>{t(k)} <b>{c}</b></span>)}
    </div>
    <div className="alert-card-list">
      {typedAlerts.map(a=><article className={`alert-pro-card ${String(a.severity).toLowerCase()} ${a.read?'read':''}`} key={a.id}>
        <div className="alert-pro-icon"><AlertTriangle/></div>
        <div className="alert-pro-body">
          <div className="alert-pro-head"><b>{t(a.type)}</b><span className={`tag ${a.severity==='High'?'red':'amber'}`}>{localizeValue(a.severity,t)}</span></div>
          <p><strong>{t('alertReason')}:</strong> {translateModelText(a.reason,t)}</p>
          <small>{t('sector')}: {localizeValue(a.sector,t)} • {t('time')}: {a.time} • {t('alertSeverity')}: {localizeValue(a.severity,t)}</small>
        </div>
        <div className="alert-pro-actions">
          <button className="secondary-btn" onClick={()=>nav('/diagnosis')}><Eye size={16}/>{t('viewDiagnosis')}</button>
          {!a.read && !a.localGenerated && <button className="secondary-btn" onClick={()=>markOne(a)}><CheckCircle2 size={16}/>{t('markAllRead')}</button>}
          {!a.localGenerated && <button className="danger-soft-btn" onClick={()=>deleteOne(a)}><X size={16}/>{t('delete')}</button>}
          {hasPermission('assignTasks')&&<button className="primary-btn" onClick={()=>createTask(a)}><CheckCircle2 size={16}/>{t('createTask')}</button>}
        </div>
      </article>)}
      {!typedAlerts.length && <section className="empty-state-card alerts-empty-state"><Bell size={34}/><h3>{lang==='ar'?'لا توجد إشعارات حاليًا':'No notifications right now'}</h3><p>{lang==='ar'?'سيظهر هنا أي تنبيه جديد من الباك إند أو من تحليل النبات.':'New backend or diagnosis notifications will appear here.'}</p></section>} 
    </div>
  </>
}
function statusLabel(status,t){
  const s=normalizeTaskStatus(status);
  if(s==='completed') return t('completed');
  if(s==='inProgress') return t('inProgress');
  return t('pending');
}

function taskToastMessage(type, lang){
  const map={
    created:{ar:'تم إرسال المهمة للعامل.', en:'Task sent to worker.'},
    updated:{ar:'تم تحديث حالة المهمة.', en:'Task status updated.'},
    done:{ar:'تم إنهاء المهمة بنجاح.', en:'Task completed successfully.'},
  };
  return lang==='ar'?map[type].ar:map[type].en;
}

function FarmAlertsPanel({sectors=[], devices=[]}){
  const {t,lang}=useLang();
  const nav=useNavigate();
  const readings=getStore('sph_readings',[]);
  const notes=notificationStore();
  const alerts=[
    ...notes.map((n,i)=>({id:n.id,type:n.type==='danger'?'criticalAlerts':'warningAlerts',reason:n.message,sector:sectors[i%Math.max(sectors.length,1)]?.name || '-',device:'-',time:n.time,severity:n.type==='danger'?'High':'Medium'})),
    ...devices.filter(d=>String(d.status).toLowerCase()==='offline').map(d=>({id:`dev-${d.id}`,type:'deviceAlerts',reason:`${d.name} ${t('offline')}`,sector:d.sector,device:d.name,time:d.lastContact || d.lastSync || '-',severity:'Medium'})),
    ...readings.filter(r=>r.final_status!=='Healthy').map((r,i)=>({id:`diag-${r.id}`,type:isDanger(r.final_status)?'criticalAlerts':'warningAlerts',reason:labelStatus(r.final_status,t),sector:r.sector || sectors[i%Math.max(sectors.length,1)]?.name || '-',device:'AI Diagnosis',time:r.time,severity:isDanger(r.final_status)?'High':'Medium'}))
  ];
  const createTask=(a)=>{
    const workers=workerAccounts(); const worker=workers[0] || {};
    const next={id:Date.now(),title:a.reason,details:`${a.reason} / ${a.device || ''}`,status:'pending',priority:a.severity,assignedTo:worker.name || '-',assignedToEmail:worker.email || worker.username || '',createdAt:new Date().toLocaleString(),sector:a.sector};
    setGlobalTasks([next,...getGlobalTasks()]);
    if(next.assignedToEmail) addUserNotification(next.assignedToEmail, taskToastMessage('created',lang), 'success');
    addNotification(lang==='ar'?'تم إنشاء مهمة من التنبيه.':'Task created from alert.','success');
  };
  return <section className="compact-section farm-alerts-operational">
    <div className="compact-section-head"><div><h3>{t('farmAlerts')}</h3><p>{t('linkedServiceFeature')}</p></div><span className="compact-head-icon"><Bell/></span></div>
    <div className="farm-alerts-grid">
      {alerts.length?alerts.map(a=><article className={`farm-alert-mini-card farm-alert-card-clean ${String(a.severity).toLowerCase()}`} key={a.id}>
        <div className="alert-pro-icon"><AlertTriangle/></div>
        <div className="farm-alert-content-clean">
          <div className="farm-alert-head-clean"><b>{t(a.type)}</b><span className={`tag ${a.severity==='High'?'red':'amber'}`}>{localizeValue(a.severity,t)}</span></div>
          <div className="farm-alert-grid-clean">
            <span><b>{t('alertReason')}</b>{translateModelText(a.reason,t)}</span>
            <span><b>{t('sector')}</b>{localizeValue(a.sector,t)}</span>
            <span><b>{t('time')}</b>{a.time}</span>
            <span><b>{t('alertSeverity')}</b>{localizeValue(a.severity,t)}</span>
          </div>
        </div>
        <div className="farm-alert-actions"><button className="secondary-btn" onClick={()=>nav('/diagnosis')}><Eye size={15}/>{t('viewDiagnosis')}</button>{hasPermission('assignTasks')&&<button className="primary-btn" onClick={()=>createTask(a)}><CheckCircle2 size={15}/>{t('createTask')}</button>}</div>
      </article>):<div className="farm-empty-focus small"><p>{t('emptyState')}</p></div>}
    </div>
  </section>
}

function FarmReportsPanel({sectors=[]}){
  const {t,lang}=useLang();
  const initialReports=getStore('sph_readings',[]).map((r,i)=>({...r,sector:r.sector || sectors[i%Math.max(sectors.length,1)]?.name || '-', diagnosis:r.diagnosis || (r.final_status==='Healthy'?'Stable plant condition with balanced readings.':'Water or heat stress indicators detected.'), date:r.date || new Date().toISOString().slice(0,10)}));
  const [readings,setReadings]=useState(initialReports);
  const [confirmReport,setConfirmReport]=useState(null);
  const [deletingReport,setDeletingReport]=useState(false);
  const bySector=sectors.map(s=>({name:s.name,count:readings.filter(r=>String(r.sector)===String(s.name)).length,status:s.status}));
  const deleteReport=(r)=>setConfirmReport(r);
  const confirmDeleteReport=async()=>{
    if(!confirmReport || deletingReport) return;
    if(!hasPermission('deleteReports')){
      addNotification(lang==='ar'?'ليست لديك صلاحية حذف التقارير.':'You do not have permission to delete reports.','error');
      return;
    }
    const reportId=backendEntityId(confirmReport);
    if(!reportId){
      addNotification(lang==='ar'?'لا يوجد ID صالح لهذا التقرير للحذف من الباك إند.':'This report has no valid backend ID to delete.','error');
      return;
    }
    setDeletingReport(true);
    try{
      await reportsAPI.delete(reportId);
      const next=readings.filter(x=>String(backendEntityId(x))!==String(reportId));
      setReadings(next); setStore('sph_readings',next); setConfirmReport(null);
      addNotification(lang==='ar'?'تم حذف التقرير من الباك إند وتحديث القائمة.':'Report deleted from backend and the list was updated.','success');
    }catch(err){
      addNotification(serviceFriendlyError(err,lang),'error');
    }finally{
      setDeletingReport(false);
    }
  };
  const exportLocal=()=>{
    const lines=['sector,time,final_status,diagnosis',...readings.map(r=>`"${r.sector}","${r.time}","${r.final_status}","${String(r.diagnosis).replaceAll('"','""')}"`)];
    download('ecosense-farm-management-reports.csv', lines.join('\n'), 'text/csv');
    addNotification(lang==='ar'?'تم تجهيز CSV لتقارير المزرعة.':'Farm report CSV exported.','success');
  };
  const exportFarmPdf=()=>{
    openProfessionalReport({ t, lang, rows: readings, titleKey:'farmReports' });
  };
  return <section className="compact-section farm-reports-operational">
    <div className="compact-section-head"><div><h3>{t('farmReports')}</h3><p>{t('finalStatusPrimary')}</p></div><span className="compact-head-icon"><FileText/></span></div>
    <div className="farm-report-action-row"><button className="secondary-btn" onClick={()=>window.location.hash='reports-preview'}><Eye size={16}/>{t('viewFullReport')}</button>{hasPermission('exportReports')&&<button className="primary-btn" onClick={exportLocal}><BarChart3 size={16}/>{t('exportCsv')}</button>}</div>
    <div className="farm-report-summary-grid">
      {bySector.map(s=><article key={s.name} className={`farm-report-sector-card ${tone(s.status)}`}><b>{localizeValue(s.name,t)}</b><span>{t('farmReports')}: {s.count}</span><em>{labelStatus(s.status,t)}</em></article>)}
    </div>
    <div className="mobile-report-cards farm-report-list-clean" id="reports-preview">
      {readings.slice(0,6).map(r=><article className="report-row-card farm-report-card-clean" key={r.id}>
        <div className="farm-report-card-head">
          <div><span>{t('reportName')}</span><b>{displayReportName('diagnosis', r.sector, t)}</b></div>
          <span className={`tag ${tone(r.final_status)}`}>{labelStatus(r.final_status,t)}</span>
        </div>
        <div className="farm-report-card-grid">
          <span><b>{t('sector')}</b>{localizeValue(r.sector,t)}</span>
          <span><b>{t('date')}</b>{r.date} {r.time}</span>
          <span><b>{t('finalStatus')}</b>{labelStatus(r.final_status,t)}</span>
        </div>
        <p><b>{t('diagnosisSummary')}:</b> {translateModelText(r.diagnosis,t)}</p>
        <div className="farm-report-actions-clean">
          <button className="secondary-btn" onClick={()=>{ openProfessionalReport({ t, lang, rows:[r], titleKey:'farmReports' }); addNotification(t('fullReportOpened'),'success'); }}><Eye size={15}/>{t('viewFullReport')}</button>
          {hasPermission('exportReports')&&<button className="primary-btn" onClick={exportFarmPdf}><FileText size={15}/>{t('exportPdf')}</button>}
          {hasPermission('deleteReports')&&<button className="danger-soft-btn" onClick={()=>deleteReport(r)}><X size={15}/>{t('delete')}</button>}
        </div>
      </article>)}
    </div>
    {confirmReport && <ConfirmModal title={lang==='ar'?'حذف التقرير':'Delete report'} message={lang==='ar'?'سيتم حذف التقرير من الباك إند وتحديث القائمة مباشرة. هل تريد المتابعة؟':'This report will be deleted from the backend and the list will update immediately. Continue?'} onCancel={()=>!deletingReport&&setConfirmReport(null)} onConfirm={confirmDeleteReport} loading={deletingReport} cancelLabel={lang==='ar'?'إلغاء':'Cancel'} confirmLabel={lang==='ar'?'حذف التقرير':'Delete report'}/>}
  </section>
}

function DeviceInventory({devices=[], sectors=[]}){
  const {t,lang}=useLang();
  const [localDevices,setLocalDevices]=useState(devices);
  const [selected,setSelected]=useState(null);
  const [showAdd,setShowAdd]=useState(false);
  const [form,setForm]=useState({name:'',type:'Sensor Device',sectorId:String(sectors[0]?.id || ''),status:'Online',value:''});
  useEffect(()=>setLocalDevices(devices),[devices]);
  const syncDevices=(next)=>{
    setLocalDevices(next);
    const allSectors=getStore('sph_sectors',[]).map(sec=>({...sec,devices:next.filter(d=>String(d.sectorId)===String(sec.id) || String(d.sector)===String(sec.name)).map(({sector,sectorId,...rest})=>rest)}));
    setStore('sph_sectors', allSectors);
  };
  const enriched=localDevices.map((d,i)=>({
    ...d,
    type:d.type || 'Sensor Device',
    status:d.status || (i%4===0?'Offline':'Online'),
    lastContact:d.lastContact || d.lastSync || (lang==='ar'?'منذ 5 دقائق':'5 min ago'),
    battery:d.battery ?? (i%3===0?74:92),
    lastReading:d.lastReading || (d.value!=null?`${d.value}${d.unit||''}`:'--')
  }));
  const reconnect=(device)=>{
    const next=enriched.map(d=>String(d.id || d.name)===String(device.id || device.name)?{...d,status:'Online',lastContact:new Date().toLocaleString()}:d);
    syncDevices(next);
    addNotification(t('deviceReconnectRequested'),'success');
  };

  const addEquipment=async()=>{
    if(!form.name.trim()) return;
    const target=sectors.find(s=>String(s.id)===String(form.sectorId)) || sectors[0] || {};
    let created={id:Date.now(),name:form.name,type:form.type,sectorId:target.id,sector:target.name,status:form.status,value:form.value,lastContact:new Date().toLocaleString(),lastReading:form.value || '--'};
    try{ const {data}=await devicesAPI.create({name:form.name,type:form.type,sectorId:target.id,status:form.status,value:form.value}); const payload=data?.device || data?.data || data; if(payload) created={...created,...payload,id:payload._id || payload.id || created.id,sector:target.name,sectorId:target.id}; }catch(err){ addNotification(serviceFriendlyError(err,lang),'warning'); }
    syncDevices([created,...enriched]);
    addNotification(lang==='ar'?'تمت إضافة المعدة.':'Equipment added.','success');
    setForm({name:'',type:'Sensor Device',sectorId:String(target.id || ''),status:'Online',value:''});
    setShowAdd(false);
  };

  const remove=async(device)=>{
    if(!hasPermission('deleteDevice')){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    try{ await devicesAPI.delete(device.id); }catch(err){ addNotification(serviceFriendlyError(err,lang),'warning'); }
    const next=enriched.filter(d=>String(d.id || d.name)!==String(device.id || device.name));
    syncDevices(next);
    addNotification(t('deviceRemoved'),'warning');
  };
  return <section className="compact-section device-inventory-section">
    <div className="compact-section-head"><div><h3>{t('devices')}</h3><p>{t('linkedServiceFeature')}</p></div><button className="primary-btn small-btn" onClick={()=>setShowAdd(!showAdd)}><Plus size={15}/>{lang==='ar'?'إضافة معدات':'Add Equipment'}</button></div>
    {showAdd && <div className="farm-add-panel compact-equipment-panel"><div className="farm-add-grid"><Field label={lang==='ar'?'اسم المعدة':'Equipment name'}><input value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></Field><Field label={lang==='ar'?'نوع المعدة':'Equipment type'}><input value={form.type} onChange={e=>setForm({...form,type:e.target.value})}/></Field><Field label={t('sector')}><select value={form.sectorId} onChange={e=>setForm({...form,sectorId:e.target.value})}>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)} - {localizeValue(s.crop,t)}</option>)}</select></Field><Field label={t('status')}><select value={form.status} onChange={e=>setForm({...form,status:e.target.value})}><option>Online</option><option>Offline</option></select></Field><Field label={lang==='ar'?'ملاحظة / آخر قراءة':'Note / last reading'}><input value={form.value} onChange={e=>setForm({...form,value:e.target.value})}/></Field><button className="primary-btn" onClick={addEquipment}><Plus size={18}/>{t('save')}</button></div></div>}
    <div className="device-card-grid-pro">
      {enriched.length?enriched.map(d=><article className={`device-card-pro ${String(d.status).toLowerCase()}`} key={d.id || d.name}>
        <header><span className="device-card-icon"><Cpu size={22}/></span><div><b>{localizeValue(d.name,t)}</b><small>{localizeValue(d.type,t)} • {localizeValue(d.sector,t)}</small></div><em className={`tag ${String(d.status).toLowerCase()==='online'?'green':'red'}`}>{localizeValue(d.status,t)}</em></header>
        <div className="device-meta-grid-pro">
          <span><b>{lang==='ar'?'آخر اتصال':'Last contact'}</b>{d.lastContact}</span>
          <span><b>{lang==='ar'?'البطارية':'Battery'}</b>{d.battery}%</span>
          <span><b>{lang==='ar'?'آخر قراءة':'Last reading'}</b>{d.lastReading}</span>
          <span><b>{t('sector')}</b>{localizeValue(d.sector || sectors[0]?.name,t)}</span>
        </div>
        <div className="device-actions-pro">
          <button className="secondary-btn" onClick={()=>setSelected(d)}><Eye size={15}/>{t('deviceDetails')}</button>
          <button className="secondary-btn" onClick={()=>reconnect(d)}><Zap size={15}/>{lang==='ar'?'إعادة الاتصال':'Reconnect'}</button>
          {hasPermission('deleteDevice') && <button className="danger-soft-btn" onClick={()=>remove(d)}><X size={15}/>{lang==='ar'?'حذف الجهاز':'Remove Device'}</button>}
        </div>
      </article>):<div className="farm-empty-focus small"><p>{t('emptyState')}</p><small>{t('serviceStructureReady')}</small></div>}
    </div>
    {selected && <div className="modal-backdrop"><section className="modal-card device-detail-modal"><button className="modal-close" onClick={()=>setSelected(null)}><X size={18}/></button><h3>{t('deviceDetails')}</h3><ReadingGrid r={{cropType:selected.type,temperature:selected.value||'-',humidity:selected.battery||'-',soilMoisture:selected.lastReading||'-',soilTemp:selected.lastContact||'-',light:selected.status||'-'}}/><p className="muted-copy">{t('linkedServiceFeature')}</p></section></div>}
  </section>
}

function CareTasksPanel({tasks=[], workers=[], sectors=[]}){
  const {t,lang}=useLang();
  const [all,setAll]=useState([]);
  const mergedWorkers=[...workers,...workerAccounts().filter(w=>!workers.some(a=>String(a.email||a.username||a.id||a._id)===String(w.email||w.username||w.id||w._id)))];
  const defaultWorkerKey = mergedWorkers[0]?._id || mergedWorkers[0]?.id || mergedWorkers[0]?.email || mergedWorkers[0]?.username || '';
  const defaultSectorId = sectors[0]?._id || sectors[0]?.id || '';
  const [form,setForm]=useState({title:'',details:'',workerKey:defaultWorkerKey,sectorId:defaultSectorId,priority:'Medium',dueDate:''});
  const [assigning,setAssigning]=useState(null);
  const loadTasks=async()=>{
    try{
      const {data}=await tasksAPI.getAll();
      const serviceTasks=asArray(data,['tasks','items','rows','records','results']).map(normalizeTaskRecord);
      setAll(serviceTasks);
      setGlobalTasks(serviceTasks);
      return serviceTasks;
    }catch(err){
      setAll([]);
      addNotification(serviceFriendlyError(err,lang),'warning');
      return [];
    }
  };
  useEffect(()=>{ let mounted=true; loadTasks().then(()=>mounted&&true); return()=>{mounted=false}; },[]);
  const createTask=async()=>{
    if(!form.title.trim()) return;
    const worker=mergedWorkers.find(w=>String(w._id||w.id||w.email||w.username||w.name)===String(form.workerKey)) || {};
    const sector=sectors.find(s=>String(s._id||s.id)===String(form.sectorId)) || {};
    if(!worker._id && !worker.id && !worker.email){
      addNotification(lang==='ar'?'اختار عامل حقيقي من القائمة قبل إرسال المهمة.':'Select a real worker before sending the task.','error');
      return;
    }
    const workerId=worker._id || worker.id || '';
    const workerEmail=worker.email || worker.username || '';
    const sectorId=sector._id || sector.id || '';
    const requestBody={
      title:form.title, name:form.title, description:form.details, details:form.details,
      status:'pending', priority:form.priority, dueDate:form.dueDate || undefined,
      workerId:workerId || undefined, assignedWorker:workerId || undefined, assignedWorkerId:workerId || undefined,
      assignedTo:workerId || workerEmail, assignedToEmail:workerEmail || undefined, workerEmail:workerEmail || undefined,
      sectorId:sectorId || undefined, assignedSector:sectorId || undefined, sector:sector.name || undefined
    };
    try{
      const {data}=await tasksAPI.create(requestBody);
      const payload=data?.task || data?.data?.task || data?.data || data;
      const task=normalizeTaskRecord(payload,0);
      if(!backendEntityId(task)){
        addNotification(lang==='ar'?'تم إنشاء المهمة لكن الباك إند لم يرجع ID صالح. اطلب من الباك إند إرجاع _id للمهمة.':'Task was created but backend did not return a valid _id. Ask backend to return the task _id.','error');
        await loadTasks();
        return;
      }
      const next=[task,...all.filter(x=>String(backendEntityId(x))!==String(backendEntityId(task)))];
      setAll(next); setGlobalTasks(next);
      addNotification(taskToastMessage('created',lang),'success');
      setForm({title:'',details:'',workerKey:defaultWorkerKey,sectorId:defaultSectorId,priority:'Medium',dueDate:''});
      await loadTasks();
    }catch(err){
      addNotification(serviceFriendlyError(err,lang),'error');
    }
  };
  const assign=async(task, workerKey)=>{
    const worker=mergedWorkers.find(w=>String(w._id || w.id || w.email || w.username || w.name)===String(workerKey)) || {};
    const taskId=backendEntityId(task);
    if(!taskId){ addNotification(lang==='ar'?'هذه المهمة غير محفوظة في الباك إند ولا يمكن إعادة تعيينها.':'This task is not saved in backend and cannot be reassigned.','error'); return; }
    const workerId=worker._id || worker.id || '';
    const workerEmail=worker.email || worker.username || '';
    try{
      await tasksAPI.update(taskId,{workerId:workerId || undefined, assignedWorker:workerId || undefined, assignedWorkerId:workerId || undefined, assignedTo:workerId || workerEmail, assignedToEmail:workerEmail || undefined, status:'pending'});
      await loadTasks();
      setAssigning(null);
      addNotification(lang==='ar'?'تم تعيين المهمة للعامل من الباك إند.':'Task assigned to the worker from backend.','success');
    }catch(err){ addNotification(serviceFriendlyError(err,lang),'error'); }
  };
  const [confirmTask,setConfirmTask]=useState(null);
  const [deletingTask,setDeletingTask]=useState(false);
  const markDone=async(task)=>{
    const taskId=backendEntityId(task);
    if(!taskId){ addNotification(lang==='ar'?'هذه المهمة غير محفوظة في الباك إند ولا يمكن تحديثها.':'This task is not saved in backend and cannot be updated.','error'); return; }
    try{ await tasksAPI.markDone(taskId,{status:'completed'}); await loadTasks(); addNotification(lang==='ar'?'تم تعليم المهمة كمكتملة.':'Task marked as done.','success'); }catch(err){ addNotification(serviceFriendlyError(err,lang),'error'); }
  };
  const confirmDeleteTask=async()=>{
    if(!confirmTask || deletingTask) return;
    if(!hasPermission('deleteTasks')){
      addNotification(lang==='ar'?'ليست لديك صلاحية حذف المهام.':'You do not have permission to delete tasks.','error');
      return;
    }
    const taskId=backendEntityId(confirmTask);
    if(!taskId){
      addNotification(lang==='ar'?'لا يوجد ID صالح لهذه المهمة للحذف من الباك إند.':'This task has no valid backend ID to delete.','error');
      return;
    }
    setDeletingTask(true);
    try{
      await tasksAPI.delete(taskId);
      const next=all.filter(x=>String(backendEntityId(x))!==String(taskId));
      setAll(next); setGlobalTasks(next); setConfirmTask(null);
      addNotification(lang==='ar'?'تم حذف المهمة من الباك إند وتحديث القائمة.':'Task deleted from backend and the list was updated.','success');
    }catch(err){
      addNotification(serviceFriendlyError(err,lang),'error');
    }finally{
      setDeletingTask(false);
    }
  };
  const list=(tasks.length?tasks:all).map(t=>({...t,status:normalizeTaskStatus(t.status)}));
  return <section className="compact-section care-tasks-section-pro">
    <div className="compact-section-head"><div><h3>{t('farmTasks')}</h3><p>{lang==='ar'?'مهام الرعاية منظمة بدون ازدحام وتظهر للعامل كإشعار.':'Care tasks are structured and sent to assigned workers.'}</p></div><span className="compact-head-icon"><CheckCircle2/></span></div>
    {hasPermission('assignTasks') && <div className="farm-add-panel compact-task-create"><div className="farm-add-grid"><Field label={t('taskName')}><input value={form.title} onChange={e=>setForm({...form,title:e.target.value})}/></Field><Field label={t('assignedTo')}><select value={form.workerKey} onChange={e=>setForm({...form,workerKey:e.target.value})}><option value="">{lang==='ar'?'اختر العامل':'Select worker'}</option>{mergedWorkers.map(w=><option key={w._id || w.id || w.email || w.username || w.name} value={w._id || w.id || w.email || w.username || w.name}>{w.name || w.fullName || w.email || w.username}</option>)}</select></Field><Field label={t('sector')}><select value={form.sectorId} onChange={e=>setForm({...form,sectorId:e.target.value})}><option value="">{lang==='ar'?'بدون قطاع':'No sector'}</option>{sectors.map(s=><option key={s._id || s.id || s.name} value={s._id || s.id || ''}>{localizeValue(s.name,t)}</option>)}</select></Field><Field label={t('priority')}><select value={form.priority} onChange={e=>setForm({...form,priority:e.target.value})}><option>High</option><option>Medium</option><option>Low</option></select></Field><Field label={lang==='ar'?'موعد التنفيذ':'Due date'}><input type="datetime-local" value={form.dueDate} onChange={e=>setForm({...form,dueDate:e.target.value})}/></Field><Field label={lang==='ar'?'وصف المهمة':'Description'}><input value={form.details} onChange={e=>setForm({...form,details:e.target.value})}/></Field><button className="primary-btn" onClick={createTask}><Plus size={18}/>{t('createTask')}</button></div></div>}
    <div className="care-task-grid-pro">
      {list.length?list.map(task=><article className={`care-task-card-pro care-task-card-clean ${normalizeTaskStatus(task.status)}`} key={task.id}>
        <div className="care-task-main"><span className="mini-row-icon task"><CheckCircle2 size={19}/></span><div><span>{t('taskName')}</span><b>{translateModelText(task.title || task.name,t)}</b></div></div>
        <div className="care-task-meta-pro">
          <span><b>{t('assignedTo')}</b>{task.assignedTo || '-'}</span>
          <span><b>{t('sector')}</b>{localizeValue(task.sector || sectors[0]?.name,t)}</span>
          <span><b>{t('priority')}</b>{localizeValue(task.priority || 'Medium',t)}</span>
          <span><b>{t('status')}</b>{statusLabel(task.status,t)}</span>
          <span><b>{t('createdAt')}</b>{task.createdAt || '-'}</span>
        </div>
        <div className="care-task-actions-pro">
          <button className="secondary-btn" disabled={!backendEntityId(task)} onClick={()=>setAssigning(assigning===task.id?null:task.id)}><UserPlus size={15}/>{task.assignedTo?t('reassign'):t('assign')}</button>
          <button className="secondary-btn" onClick={()=>alert(`${translateModelText(task.title || task.name,t)}\n${translateModelText(task.details || '',t)}`)}><Eye size={15}/>{t('viewDetails')}</button>
          <button className="primary-btn" onClick={()=>markDone(task)}><CheckCircle2 size={15}/>{lang==='ar'?'تمت':'Done'}</button>
          {hasPermission('deleteTasks')&&<button className="danger-soft-btn" disabled={!backendEntityId(task)} onClick={()=>setConfirmTask(task)}><X size={15}/>{t('delete')}</button>}
        </div>
        {assigning===task.id && <div className="inline-worker-select"><select defaultValue="" onChange={e=>assign(task,e.target.value)}><option value="">{lang==='ar'?'اختر العامل لإرسال المهمة':'Select worker to send task'}</option>{mergedWorkers.map(w=><option key={w._id || w.id || w.email || w.username || w.name} value={w._id || w.id || w.email || w.username || w.name}>{w.name || w.email || w.username}</option>)}</select></div>}
      </article>):<div className="farm-empty-focus small"><p>{t('emptyState')}</p></div>}
    </div>
    {confirmTask && <ConfirmModal title={lang==='ar'?'حذف المهمة':'Delete task'} message={lang==='ar'?'سيتم حذف المهمة من الباك إند وتختفي من القائمة مباشرة. هل تريد المتابعة؟':'This task will be deleted from the backend and removed from the list immediately. Continue?'} onCancel={()=>!deletingTask&&setConfirmTask(null)} onConfirm={confirmDeleteTask} loading={deletingTask} cancelLabel={lang==='ar'?'إلغاء':'Cancel'} confirmLabel={lang==='ar'?'حذف المهمة':'Delete task'}/>}
  </section>
}

function WorkerDashboard(){
  const {t,lang}=useLang();
  const tasks=getGlobalTasks().filter(taskAssignedToCurrent);
  const pending=tasks.filter(x=>normalizeTaskStatus(x.status)==='pending').length;
  const inProgress=tasks.filter(x=>normalizeTaskStatus(x.status)==='inProgress').length;
  const completed=tasks.filter(x=>normalizeTaskStatus(x.status)==='completed').length;
  const user=currentUser();
  return <>
    <PageHead title={t('workerDashboard')} sub={lang==='ar'?'لوحة العامل تعرض المهام والقطاع والتنبيهات المخصصة فقط.':'Worker dashboard shows only assigned tasks, sector, and alerts.'} action={<AccountRoleBadge/>}/>
    <section className="farm-greeting-banner worker-greeting-banner">
      <div className="farm-greeting-content">
        <span className="farm-greeting-eyebrow"><Leaf size={16}/>{t('appName')}</span>
        <h2>{t(greetingKeyByTime())}, {displayUserName()} 👋</h2>
        <p>{t('overviewSubtitle')}</p>
      </div>
      <div className="farm-greeting-status">
        <span className={`role-badge role-${currentRole()}`}>{roleLabel(t)}</span>
        <small><i />{t('online')}</small>
      </div>
    </section>
    <div className="kpis">
      <Kpi icon={<CheckCircle2/>} label={t('pending')} value={pending}/>
      <Kpi icon={<Activity/>} label={t('inProgress')} value={inProgress}/>
      <Kpi icon={<ShieldCheck/>} label={t('completed')} value={completed}/>
      <Kpi icon={<Layers3/>} label={t('assignedSector')} value={localizeValue(user.sector || '-',t)}/>
    </div>
    <MyTasks embedded/>
  </>
}

function MyTasks({embedded=false}){
  const {t,lang}=useLang();
  const [tasks,setTasks]=useState([]);
  const [note,setNote]=useState('');
  const [proof,setProof]=useState('');
  const choose=e=>{const f=e.target.files?.[0]; if(!f) return; const reader=new FileReader(); reader.onload=()=>setProof(reader.result); reader.readAsDataURL(f);};
  const refresh=async()=>{
    try{
      const {data}=await tasksAPI.getAll();
      const serviceTasks=asArray(data,['tasks','items','rows','records','results']).map(normalizeTaskRecord);
      if(serviceTasks.length){
        setGlobalTasks(serviceTasks);
        const assigned = serviceTasks.filter(taskAssignedToCurrent);
        setTasks(isWorker() && assigned.length === 0 ? serviceTasks : assigned.length ? assigned : serviceTasks);
        return;
      }
      setTasks([]);
    }catch(err){
      addNotification(serviceFriendlyError(err,lang),'warning');
      setTasks([]);
    }
  };
  useEffect(()=>{ refresh(); },[]);
  const updateTask=async(task,status)=>{
    const taskId=backendEntityId(task);
    if(!taskId){ addNotification(lang==='ar'?'هذه المهمة غير محفوظة في الباك إند ولا يمكن تحديثها.':'This task is not saved in backend and cannot be updated.','error'); return; }
    try{
      await tasksAPI.update(taskId,{status,note:note || task.note,proof:proof || task.proof,completedAt:status==='completed'?new Date().toISOString():undefined});
      addNotification(status==='completed'?taskToastMessage('done',lang):taskToastMessage('updated',lang),'success');
      setNote(''); setProof('');
      await refresh();
    }catch(err){ addNotification(serviceFriendlyError(err,lang),'error'); }
  };
  return <section className="my-tasks-page-pro">
    {!embedded && <PageHead title={t('myTasks')} sub={lang==='ar'?'المهام الخاصة بالحساب الحالي فقط.':'Tasks assigned to the current logged-in account only.'} action={<AccountRoleBadge/>}/>} 
    <div className="worker-notice-pro"><Bell size={20}/><div><b>{lang==='ar'?'إشعارات المهام':'Task notifications'}</b><span>{tasks.filter(t=>normalizeTaskStatus(t.status)==='pending').length} {t('pending')}</span></div><button className="secondary-btn" onClick={refresh}>{lang==='ar'?'تحديث':'Refresh'}</button></div>
    <div className="my-task-grid-pro">
      {tasks.length?tasks.map(task=><article className={`my-task-card-pro ${normalizeTaskStatus(task.status)}`} key={task.id}>
        <header><div><b>{translateModelText(task.title || task.name,t)}</b><small>{t('sector')}: {localizeValue(task.sector,t)} • {lang==='ar'?'الأولوية':'Priority'}: {localizeValue(task.priority || 'Medium',t)}</small></div><span className={`tag ${normalizeTaskStatus(task.status)==='completed'?'green':normalizeTaskStatus(task.status)==='inProgress'?'amber':'red'}`}>{statusLabel(task.status,t)}</span></header>
        <p>{translateModelText(task.details || task.description || 'Complete the requested plant care action.',t)}</p>
        <div className="task-time-grid"><span><b>{lang==='ar'?'وقت الإنشاء':'Created at'}</b>{task.createdAt || '-'}</span><span><b>{lang==='ar'?'وقت الانتهاء':'Completed at'}</b>{task.completedAt || '-'}</span></div>
        {(proof || task.proof) && <img src={proof || task.proof} className="proof-thumb" alt="proof"/>}
        <div className="proof-tools-pro"><label className="proof-upload"><Camera size={16}/>{t('proofUpload')}<input hidden type="file" accept="image/*" capture="environment" onChange={choose}/></label><input value={note} onChange={e=>setNote(e.target.value)} placeholder={t('proofNote')}/></div>
        <div className="my-task-actions-pro"><button className="secondary-btn" onClick={()=>updateTask(task,'inProgress')}><Activity size={15}/>{t('startTask')}</button><button className="primary-btn" onClick={()=>updateTask(task,'completed')}><CheckCircle2 size={15}/>{t('markDone')}</button></div>
      </article>):<EmptyActionState icon={CheckCircle2} title={lang==='ar'?'لا توجد مهام مخصصة لك':'No assigned tasks'} description={lang==='ar'?'عند تعيين مهمة لك من المالك ستظهر هنا من الباك إند.':'Tasks assigned by the owner will appear here from the backend.'}/>}
    </div>
  </section>
}

function TaskBoard({embedded=false}){
  const {t,lang}=useLang();
  const [tasks,setTasks]=useState([]);
  const [loading,setLoading]=useState(false);
  const [serviceError,setServiceError]=useState('');
  const sectors=getStore('sph_sectors',[]);
  const accountWorkers=workerAccounts();
  const sectorWorkers=sectors.flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name,sectorId:s.id})));
  const workers=[...accountWorkers,...sectorWorkers.filter(w=>!accountWorkers.some(a=>String(a.email||a.id)===String(w.email||w.id)) )];
  const [form,setForm]=useState({title:'',assignedToEmail:workers[0]?.email || '',sector:sectors[0]?.name || '',sectorId:sectors[0]?.id || '',priority:'Medium',details:''});
  const normalizeTaskRow=(task,idx=0)=>({
    ...task,
    id:task._id || task.id || task.taskId || idx+1,
    title:task.title || task.name || task.taskName || '-',
    details:task.details || task.description || task.note || '',
    status:normalizeTaskStatus(task.status),
    priority:task.priority || 'Medium',
    assignedTo:task.assignedTo?.name || task.worker?.name || task.assignedToName || task.assignedTo || '-',
    assignedToEmail:task.assignedTo?.email || task.worker?.email || task.assignedToEmail || '',
    workerId:task.workerId || task.worker?._id || task.assignedTo?._id || '',
    sector:task.sector?.name || task.sectorName || task.sector || '',
    sectorId:task.sector?._id || task.sectorId || task.assignedSector || '',
    createdAt:task.createdAt || task.created_at || '',
    completedAt:task.completedAt || task.completed_at || '',
    service:true,
  });
  const loadTasks=async()=>{
    setLoading(true); setServiceError('');
    try{
      const {data}=await tasksAPI.getAll();
      const serviceTasks=asArray(data,['tasks','items','rows','records']).map(normalizeTaskRow);
      setTasks(serviceTasks);
    }catch(err){
      setTasks([]);
      setServiceError(serviceFriendlyError(err,lang));
    }finally{ setLoading(false); }
  };
  useEffect(()=>{ loadTasks(); },[lang]);
  const save=async()=>{
    if(!form.title.trim()) return addNotification(lang==='ar'?'اكتب اسم المهمة أولًا.':'Enter the task name first.','warning');
    if(!form.assignedToEmail) return addNotification(lang==='ar'?'اختر العامل المسؤول عن المهمة.':'Select the worker assigned to the task.','warning');
    const worker=workers.find(w=>String(w.email || w.username || w.id || w.name)===String(form.assignedToEmail)) || {};
    const sector=sectors.find(s=>String(s.id)===String(form.sectorId) || String(s.name)===String(form.sector));
    try{
      const {data}=await tasksAPI.create({
        title:form.title,
        name:form.title,
        description:form.details,
        details:form.details,
        status:'pending',
        priority:form.priority,
        workerId:worker.id || worker._id || '',
        assignedTo:worker.id || worker._id || worker.email || worker.username || form.assignedToEmail,
        assignedToEmail:worker.email || worker.username || form.assignedToEmail,
        sectorId:sector?.id || form.sectorId || '',
        sector:sector?.name || form.sector || '',
      });
      const payload=data?.task || data?.data?.task || data?.data || data;
      const row=normalizeTaskRow(payload,0);
      setTasks(prev=>[row,...prev]);
      if(row.assignedToEmail) addUserNotification(row.assignedToEmail, taskToastMessage('created',lang), 'success');
      addNotification(taskToastMessage('created',lang),'success');
      setForm({title:'',assignedToEmail:workers[0]?.email || '',sector:sectors[0]?.name || '',sectorId:sectors[0]?.id || '',priority:'Medium',details:''});
    }catch(err){
      setServiceError(serviceFriendlyError(err,lang));
      addNotification(serviceFriendlyError(err,lang),'danger');
    }
  };
  const move=async(task,status)=>{
    const taskId=backendEntityId(task);
    if(!taskId) return addNotification(lang==='ar'?'هذه المهمة ليست مربوطة بالباك إند ولا يمكن تحديثها.':'This task is not linked to the backend and cannot be updated.','warning');
    try{
      await tasksAPI.update(taskId,{status:normalizeTaskStatus(status)});
      const all=tasks.map(x=>String(x.id)===String(task.id)?{...x,status:normalizeTaskStatus(status),completedAt:normalizeTaskStatus(status)==='completed'?new Date().toLocaleString():x.completedAt}:x);
      setTasks(all);
      addNotification(taskToastMessage(normalizeTaskStatus(status)==='completed'?'done':'updated',lang),'success');
    }catch(err){ addNotification(serviceFriendlyError(err,lang),'danger'); }
  };
  const cols=[['pending',t('pending')],['inProgress',t('inProgress')],['completed',t('completed')]];
  return <>
    {!embedded && <PageHead title={t('taskBoard')} sub={t('createTreatmentTask')} action={<button className="secondary-btn" onClick={loadTasks} disabled={loading}><Activity size={16}/>{lang==='ar'?'تحديث':'Refresh'}</button>}/>} 
    {serviceError && <p className="form-helper warning-text">{serviceError}</p>}
    <Panel title={t('createTask')}>
      <div className="form-grid">
        <Field label={t('taskName')}><input value={form.title} onChange={e=>setForm({...form,title:e.target.value})}/></Field>
        <Field label={t('assignedTo')}><select value={form.assignedToEmail} onChange={e=>setForm({...form,assignedToEmail:e.target.value})}><option value="">{lang==='ar'?'اختر العامل':'Select worker'}</option>{workers.map(w=><option key={w.email || w.id || w.name} value={w.email || w.username || w.id || w.name}>{w.name || w.email}</option>)}</select></Field>
        <Field label={t('sector')}><select value={form.sectorId} onChange={e=>{const sec=sectors.find(s=>String(s.id)===String(e.target.value)); setForm({...form,sectorId:e.target.value,sector:sec?.name || ''});}}><option value="">{lang==='ar'?'بدون قطاع':'No sector'}</option>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}</option>)}</select></Field>
        <Field label={t('priority')}><select value={form.priority} onChange={e=>setForm({...form,priority:e.target.value})}><option>High</option><option>Medium</option><option>Low</option></select></Field>
        <Field label={lang==='ar'?'المطلوب تنفيذه':'Task details'}><input value={form.details} onChange={e=>setForm({...form,details:e.target.value})}/></Field>
      </div>
      <button className="primary-btn" onClick={save} disabled={loading}><Plus size={18}/>{t('createTask')}</button>
    </Panel>
    {loading ? <SkeletonGrid cards={3}/> : tasks.length ? <div className="kanban">{cols.map(([key,label])=><section className="kanban-col" key={key}><h3>{label}</h3>{tasks.filter(x=>normalizeTaskStatus(x.status)===key).map(task=><article className="task-card" key={task.id}><b>{translateModelText(task.title,t)}</b><span>{t('assignedTo')}: {translateModelText(task.assignedTo,t)}</span><span>{t('sector')}: {translateModelText(task.sector,t)}</span><span>{lang==='ar'?'وقت الإنشاء':'Created'}: {formatDateTime(task.createdAt) || '-'}</span><em>{localizeValue(task.priority,t)}</em><div><button onClick={()=>move(task,'pending')}>{t('pending')}</button><button onClick={()=>move(task,'inProgress')}>{t('inProgress')}</button><button onClick={()=>move(task,'completed')}>{t('completed')}</button></div></article>)}</section>)}</div> : <EmptyActionState icon={CheckCircle2} title={lang==='ar'?'لا توجد مهام محفوظة':'No saved tasks'} description={lang==='ar'?'أنشئ مهمة حقيقية من خلال endpoint /tasks لتظهر للعامل.':'Create a real task through the /tasks endpoint so it appears for the worker.'}/>} 
  </>
}

function AdminPanel({embedded=false}){
  const {t, lang}=useLang();
  const nav=useNavigate();
  const role=currentRole();
  const [sectors,setSectors]=useState(getStore('sph_sectors',[]));
  const [adminDevices,setAdminDevices]=useState([]);
  const [adminWorkers,setAdminWorkers]=useState(workerAccounts());
  const [adminNotifications,setAdminNotifications]=useState([]);
  const [adminTasks,setAdminTasks]=useState([]);
  const [recentDiagnosis,setRecentDiagnosis]=useState(null);
  const [latestSensor,setLatestSensor]=useState(null);
  const [adminStatus,setAdminStatus]=useState({loading:false,live:false,error:''});
  const savedFarmSettings=getStore('sph_settings',{});
  const ownerLabel = lang==='ar' ? 'مالك' : t('owner');
  const emptyText = lang==='ar' ? 'لا توجد بيانات متاحة حاليًا' : 'No data is available right now';
  const [adminFarm,setAdminFarm]=useState({
    farmName:savedFarmSettings.farmName || 'Ecosense Smart Farm',
    crop:savedFarmSettings.defaultCrop || '',
    location:savedFarmSettings.farmLocation || '',
    manager:savedFarmSettings.farmManager || ''
  });

  useEffect(()=>{
    if(role!=='owner') return;
    const token=localStorage.getItem('ecosense_token') || localStorage.getItem('sph_token');
    if(!token) return;
    let mounted=true;
    setAdminStatus({loading:true,live:false,error:''});
    Promise.allSettled([
      sectorsAPI.getAll(),
      devicesAPI.getAll(),
      usersAPI.getWorkers(),
      notificationsAPI.getAll(),
      sensorsAPI.getLatestHardware(),
      sensorsAPI.getHardwareHistory({limit:1}),
      imagesAPI.getHardwareHistory({limit:1}),
      tasksAPI.getAll().catch(()=>({data:[]})),
    ]).then(results=>{
      if(!mounted) return;
      const [sectorsRes,devicesRes,workersRes,notificationsRes,latestRes,sensorHistoryRes,imageHistoryRes,tasksRes]=results;
      if(sectorsRes.status==='fulfilled'){
        const list=asArray(sectorsRes.value.data,['sectors','items','results']);
        const normalized=list.map((item,idx)=>({
          id:item._id || item.id || item.sectorId || idx+1,
          _id:item._id || item.id || item.sectorId,
          name:item.name || item.sectorName || `Sector ${idx+1}`,
          crop:item.crop || item.cropType || item.plantType || '-',
          location:item.location || item.area || '-',
          status:item.status || item.final_status || item.healthStatus || '',
          workers:item.workers || [],
          devices:item.devices || [],
          diagnosis:item.diagnosis || item.analysis || {},
        }));
        setSectors(normalized); setStore('sph_sectors',normalized);
      }
      if(devicesRes.status==='fulfilled') setAdminDevices(asArray(devicesRes.value.data,['devices','items','results']));
      if(workersRes.status==='fulfilled'){
        const mapped=asArray(workersRes.value.data,['workers','users','items']).map((w,idx)=>({
          id:w._id || w.id || idx+1,
          name:w.name || w.fullName || [w.firstName,w.lastName].filter(Boolean).join(' ') || w.email || `Worker ${idx+1}`,
          email:w.email || w.username || '', phone:w.phone || w.phoneNumber || '',
          role:w.role || w.jobTitle || 'worker', jobTitle:w.jobTitle || w.role || 'Worker',
          sectorId:w.sectorId || w.assignedSectorId || w.sector?._id,
          sector:w.sectorName || w.assignedSector || w.sector?.name || w.sector || '-',
          status:w.status || (w.isActive===false?'Offline':'Online'),
        }));
        setAdminWorkers(mapped); setStore('sph_worker_accounts',mapped);
      }
      if(notificationsRes.status==='fulfilled') setAdminNotifications(asArray(notificationsRes.value.data,['notifications','items','results']));
      if(latestRes.status==='fulfilled') setLatestSensor(normalizeServiceSensorPayload(latestRes.value.data || {}).reading || null);
      const sensorRecords=sensorHistoryRes.status==='fulfilled'?diagnosisRecordsFromPayload(sensorHistoryRes.value.data):[];
      const imageRecords=imageHistoryRes.status==='fulfilled'?diagnosisRecordsFromPayload(imageHistoryRes.value.data):[];
      setRecentDiagnosis(sensorRecords[0] || imageRecords[0] || null);
      if(tasksRes.status==='fulfilled') setAdminTasks(asArray(tasksRes.value.data,['tasks','items','results']));
      setAdminStatus({loading:false,live:results.some(r=>r.status==='fulfilled'),error:''});
    }).catch(err=>mounted && setAdminStatus({loading:false,live:false,error:serviceFriendlyError(err,lang)}));
    return()=>{mounted=false};
  },[role,lang]);

  const saveAdminFarm=()=>{
    const current=getStore('sph_settings',{});
    setStore('sph_settings',{...current,farmName:adminFarm.farmName,defaultCrop:adminFarm.crop,farmLocation:adminFarm.location,farmManager:adminFarm.manager});
    addNotification(lang==='ar'?'تم تحديث بيانات المزرعة من لوحة الإدارة.':'Farm data updated from admin dashboard.','success');
  };
  const deleteSectorAdmin=async(id)=>{
    if(!hasPermission('deleteSector')){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    try{ await sectorsAPI.delete(id); setServiceLive(true); }
    catch(err){ addNotification(userFriendlyServiceError(err,lang),'danger'); return; }
    const next=sectors.filter(s=>String(s.id)!==String(id));
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم حذف القطاع من الباك إند.':'Sector deleted from backend.','success');
  };
  const editSectorAdmin=async(sector)=>{
    const name=prompt(lang==='ar'?'اسم القطاع الجديد':'New sector name', sector.name);
    if(!name) return;
    const crop=prompt(lang==='ar'?'نوع المحصول':'Crop type', sector.crop) || sector.crop;
    const location=prompt(lang==='ar'?'بيانات أو موقع القطاع':'Sector details or location', sector.location) || sector.location;
    try{ await sectorsAPI.update(sector.id || sector._id, {name,location,...(crop ? {crop,cropType:crop} : {})}); setServiceLive(true); }
    catch(err){ addNotification(userFriendlyServiceError(err,lang),'danger'); return; }
    const next=sectors.map(s=>String(s.id)===String(sector.id)?{...s,name,crop,location}:s);
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم تعديل بيانات القطاع في الباك إند.':'Sector updated in backend.','success');
  };
  const assignAdminWorkerToSector=(sector)=>{
    const workerName=prompt(lang==='ar'?'اكتب اسم العامل أو المدير للقطاع':'Enter worker or manager name for this sector', adminWorkers[0]?.name || adminFarm.manager || '');
    if(!workerName) return;
    const newWorker={id:Date.now(),name:workerName,role:'Farm Manager',phone:'-',status:'Active'};
    const next=sectors.map(s=>String(s.id)===String(sector.id)?{...s,workers:[newWorker,...(s.workers||[])]}:s);
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم تعيين مسؤول للقطاع.':'Sector manager assigned.','success');
  };

  const diagnosisTotal=(recentDiagnosis?1:0);
  const overviewCards=[
    {icon:Layers3,label:t('sectors'),value:sectors.length},
    {icon:Cpu,label:t('devices'),value:adminDevices.length},
    {icon:Users,label:t('workers'),value:adminWorkers.length},
    {icon:ScanSearch,label:t('myDiagnosis'),value:diagnosisTotal},
    {icon:Bell,label:t('alerts'),value:adminNotifications.length,tone:adminNotifications.length?'amber':'green'},
  ];
  const recentActivity=[
    recentDiagnosis && {icon:ScanSearch,title:t('latestDiagnosis'),desc:labelStatus(recentDiagnosis.final_status || recentDiagnosis.status || recentDiagnosis.analysis?.status,t),time:formatDateTime(recentDiagnosis.createdAt || recentDiagnosis.date)},
    adminNotifications[0] && {icon:Bell,title:t('alerts'),desc:adminNotifications[0].title || adminNotifications[0].message || t('alerts'),time:formatDateTime(adminNotifications[0].createdAt || adminNotifications[0].date)},
    latestSensor && {icon:Activity,title:t('latestReading'),desc:`${t('soilMoisture')}: ${latestSensor.soilMoisture ?? '-'}%`,time:formatDateTime(latestSensor.createdAt || latestSensor.date || latestSensor.time)},
    adminTasks[0] && {icon:CheckCircle2,title:t('latestTask'),desc:adminTasks[0].title || adminTasks[0].name || t('tasks'),time:formatDateTime(adminTasks[0].createdAt || adminTasks[0].date)},
  ].filter(Boolean);

  return <>
    {!embedded && <PageHead title={t('adminPanel')} sub={lang==='ar'?'لوحة إدارة مختصرة تعرض بيانات حسابك من الخدمات المتصلة.':'Compact admin dashboard for your connected account data.'} action={<AccountRoleBadge/>}/>}    
    {role!=='owner'?<Unauthorized/>:<section className="admin-dashboard-v3 admin-page-standalone">
      <section className="admin-ops-header">
        <div><span><ShieldCheck size={16}/>{t('adminPanel')}</span><h2>{lang==='ar'?'لوحة تشغيل المزرعة':'Farm Operations Dashboard'}</h2><p>{adminStatus.loading?t('loadingData'):(adminStatus.live? (lang==='ar'?'البيانات معروضة من الخدمات المتاحة.':'Data is shown from available services.') : emptyText)}</p></div>
        <strong>{ownerLabel}</strong>
      </section>

      <section className="admin-overview-grid">
        {overviewCards.map(({icon:Icon,label,value,tone:tn})=><article className={`admin-overview-card ${tn||''}`} key={label}><span><Icon size={18}/></span><div><small>{label}</small><b>{value}</b></div></article>)}
      </section>

      <section className="admin-quick-actions">
        <button onClick={()=>nav('/farm-management')}><Plus size={17}/>{lang==='ar'?'إضافة قطاع':'Add Sector'}</button>
        <button onClick={()=>nav('/workers')}><UserPlus size={17}/>{lang==='ar'?'إضافة عامل':'Add Worker'}</button>
        <button onClick={()=>nav('/farm-management')}><Cpu size={17}/>{lang==='ar'?'إضافة جهاز':'Add Device'}</button>
        <button onClick={()=>nav('/diagnosis')}><ScanSearch size={17}/>{t('diagnosisCenter')}</button>
        <button onClick={()=>nav('/farm-management')}><FileText size={17}/>{t('farmReports')}</button>
      </section>

      <section className="admin-activity-card">
        <div className="compact-section-head"><div><h3>{lang==='ar'?'آخر النشاط':'Recent Activity'}</h3><p>{lang==='ar'?'أحدث بيانات فعلية من الباك إند عند توفرها.':'Latest real backend activity when available.'}</p></div></div>
        <div className="admin-activity-list">
          {recentActivity.length ? recentActivity.map(({icon:Icon,title,desc,time},i)=><article key={i}><span><Icon size={17}/></span><div><b>{title}</b><small>{desc}</small></div><em>{time || '-'}</em></article>) : <div className="farm-empty-focus small"><p>{emptyText}</p></div>}
        </div>
      </section>

      <div className="admin-sections-grid-v3">
        <section className="admin-section-card-v2 farm-data compact-admin-card">
          <header><Layers3 size={20}/><div><h4>{lang==='ar'?'بيانات المزرعة':'Farm Data'}</h4><p>{lang==='ar'?'تعديل البيانات الأساسية فقط.':'Edit only the core farm details.'}</p></div></header>
          <div className="admin-form-grid-v2">
            <Field label={t('farmName')}><input value={adminFarm.farmName} onChange={e=>setAdminFarm({...adminFarm,farmName:e.target.value})}/></Field>
            <Field label={t('cropType')}><input value={adminFarm.crop || ''} onChange={e=>setAdminFarm({...adminFarm,crop:e.target.value})} placeholder={lang==='ar'?'من بيانات الباك إند أو اكتب عند تعديل المزرعة':'Backend value or enter crop when editing'}/></Field>
            <Field label={t('location')}><input value={adminFarm.location} onChange={e=>setAdminFarm({...adminFarm,location:e.target.value})}/></Field>
            <Field label={lang==='ar'?'مدير المزرعة':'Farm Manager'}><select value={adminFarm.manager} onChange={e=>setAdminFarm({...adminFarm,manager:e.target.value})}><option value="">{lang==='ar'?'اختر مدير':'Select manager'}</option>{adminWorkers.map(w=><option key={w.id || w.email || w.name} value={w.name}>{w.name}</option>)}</select></Field>
          </div>
          <button className="primary-btn" onClick={saveAdminFarm}><ShieldCheck size={18}/>{t('save')}</button>
        </section>

        <section className="admin-section-card-v2 sectors-control compact-admin-card">
          <header><Sprout size={20}/><div><h4>{lang==='ar'?'القطاعات':'Sectors'}</h4><p>{lang==='ar'?'قائمة مختصرة للقطاعات الفعلية.':'Compact list of real sectors.'}</p></div></header>
          <div className="admin-sector-list-v2 compact-admin-sector-list">
            {sectors.length ? sectors.map(s=><article key={s.id || s._id}>
              <div><b>{localizeValue(s.name,t)}</b><span>{localizeValue(s.crop,t)} • {localizeValue(s.location,t)}</span></div>
              <em className={`tag ${tone(s.status)}`}>{labelStatus(s.status,t)}</em>
              <div className="admin-row-actions-v2"><button onClick={()=>editSectorAdmin(s)}><FileText size={15}/>{lang==='ar'?'تعديل':'Edit'}</button><button onClick={()=>assignAdminWorkerToSector(s)}><UserPlus size={15}/>{lang==='ar'?'تعيين':'Assign'}</button><button className="danger" onClick={()=>deleteSectorAdmin(s.id)}><X size={15}/>{t('delete')}</button></div>
            </article>) : <div className="farm-empty-focus small"><p>{emptyText}</p></div>}
          </div>
        </section>
      </div>
    </section>}
  </>
}

function Unauthorized(){ const {t}=useLang(); return <section className="unauthorized"><Lock size={46}/><h2>{t('unauthorized')}</h2><p>{t('unauthorizedText')}</p></section> }

function Onboarding(){
  const {t}=useLang();
  const nav=useNavigate();
  const [form,setForm]=useState({farmName:'',firstSector:'',firstDevice:'',enableAlerts:true});
  const finish=()=>{
    setStore('sph_onboarding',form); localStorage.setItem('sph_onboarded','true'); alert(t('setupSaved')); nav('/diagnosis');
  };
  return <main className="auth-page onboarding-page">
    <div className="auth-left"><LogoBlock/><h1>{t('onboarding')}</h1><p>{t('welcomeSetup')}</p><div className="demo-flow"><span>{t('onboardingStepAddSector')}</span><span>{t('onboardingStepLinkDevice')}</span><span>{t('onboardingStepFirstDiagnosis')}</span><span>{t('onboardingStepAddWorker')}</span><span>{t('onboardingStepEnableAlerts')}</span></div></div>
    <section className="auth-card">
      <h2>{t('onboarding')}</h2>
      <Field label={t('farmName')}><input value={form.farmName} onChange={e=>setForm({...form,farmName:e.target.value})}/></Field>
      <Field label={t('firstSector')}><input value={form.firstSector} onChange={e=>setForm({...form,firstSector:e.target.value})}/></Field>
      <Field label={t('firstDevice')}><input value={form.firstDevice} onChange={e=>setForm({...form,firstDevice:e.target.value})}/></Field>
      <label className="setting-toggle"><span>{t('enableAlerts')}</span><input type="checkbox" checked={form.enableAlerts} onChange={e=>setForm({...form,enableAlerts:e.target.checked})}/></label>
      <button className="primary-btn wide" onClick={finish}>{t('finishSetup')}</button>
      <button className="secondary-btn wide" onClick={()=>{localStorage.setItem('sph_onboarded','true');nav('/diagnosis')}}>{t('skipNow')}</button>
    </section>
  </main>
}


function resetCopy(lang, key){
  const ar={
    forgotTitle:'استعادة كلمة المرور',
    forgotSub:'اكتب بريد حسابك وسنرسل لك كود تحقق مكون من 6 أرقام لإعادة تعيين كلمة السر.',
    resetTitle:'تعيين كلمة سر جديدة',
    resetSub:'اكتب كود التحقق الذي وصلك على الإيميل، ثم أدخل كلمة سر جديدة وقوية.',
    email:'البريد الإلكتروني', otp:'كود التحقق', newPassword:'كلمة السر الجديدة', confirmPassword:'تأكيد كلمة السر الجديدة',
    sendCode:'إرسال كود الاستعادة', changePassword:'تغيير كلمة السر',
    codeSent:'تم إرسال كود التحقق إلى بريدك الإلكتروني. اكتب الكود وكلمة السر الجديدة.',
    passwordChanged:'تم تغيير كلمة المرور بنجاح. يمكنك تسجيل الدخول الآن بكلمة السر الجديدة.',
    backLogin:'الرجوع إلى تسجيل الدخول', requestNew:'طلب كود جديد', needEmail:'اكتب البريد الإلكتروني أولًا.', validEmail:'اكتب بريد إلكتروني صحيح.',
    needOtp:'اكتب كود التحقق.', invalidOtp:'كود التحقق يجب أن يكون 6 أرقام.',
    weakPassword:'كلمة السر يجب ألا تقل عن 8 أحرف وتحتوي على حروف وأرقام.', mismatch:'تأكيد كلمة السر غير مطابق.',
    expired:'انتهت جلسة إعادة التعيين، من فضلك اطلب كود جديد.',
    missingService:'لو ظهر خطأ 404 فهذا يعني أن الخدمة يحتاج تفعيل service path استعادة كلمة المرور.',
    step1:'إرسال الكود', step2:'تعيين كلمة السر', strength:'قوة كلمة السر', weak:'ضعيفة', medium:'متوسطة', strong:'قوية',
    apiReady:'Auth flow حقيقي متصل بالخدمة', tokenSaved:'تم حفظ جلسة الاستعادة مؤقتًا وآمنًا في sessionStorage.'
  };
  const en={
    forgotTitle:'Forgot Password',
    forgotSub:'Enter your account email and we will send a 6-digit reset code to your inbox.',
    resetTitle:'Reset Password',
    resetSub:'Enter the OTP sent to your email, then create a strong new password.',
    email:'Email Address', otp:'Reset Code', newPassword:'New Password', confirmPassword:'Confirm New Password',
    sendCode:'Send Reset Code', changePassword:'Change Password',
    codeSent:'Reset code has been sent to your email. Enter the code and your new password.',
    passwordChanged:'Password changed successfully. You can now sign in with the new password.',
    backLogin:'Back to Login', requestNew:'Request New Code', needEmail:'Please enter your email first.', validEmail:'Please enter a valid email address.',
    needOtp:'Please enter the reset code.', invalidOtp:'Reset code must be 6 digits.',
    weakPassword:'Password must be at least 8 characters and include letters and numbers.', mismatch:'Confirm password does not match.',
    expired:'Reset session expired. Please request a new code.',
    missingService:'If you receive 404, the service still needs to enable the reset-password service path.',
    step1:'Send Code', step2:'Reset Password', strength:'Password strength', weak:'Weak', medium:'Medium', strong:'Strong',
    apiReady:'Real service auth flow', tokenSaved:'Reset session is stored temporarily in sessionStorage.'
  };
  return (lang==='ar'?ar:en)[key] || key;
}
function getResetTokenFromResponse(data={}){
  return data?.resetToken || data?.token || data?.data?.resetToken || data?.data?.token || data?.reset_token || data?.data?.reset_token || data?.data?.resetToken?.token || '';
}
function passwordScore(value=''){
  const p=String(value||'');
  let score=0;
  if(p.length>=8) score++;
  if(/[A-Z]/.test(p) && /[a-z]/.test(p)) score++;
  if(/\d/.test(p)) score++;
  if(/[^A-Za-z0-9]/.test(p)) score++;
  return Math.min(score,3);
}
function ResetAuthFlow({mode='forgot', theme, setTheme}){
  const {t, lang}=useLang();
  const nav=useNavigate();
  const location=useLocation();
  const queryEmail=new URLSearchParams(location.search).get('email') || '';
  const readSession=key=>sessionStorage.getItem(key) || '';
  const [email,setEmail]=useState(readSession('sph_reset_email') || queryEmail);
  const [otp,setOtp]=useState('');
  const [resetToken,setResetToken]=useState(readSession('sph_reset_token'));
  const [newPassword,setNewPassword]=useState('');
  const [confirmPassword,setConfirmPassword]=useState('');
  const [loading,setLoading]=useState(false);
  const [msg,setMsg]=useState({type:'', text:''});
  const isForgot=mode==='forgot';
  const score=passwordScore(newPassword);
  const status=score>=3?'strong':score===2?'medium':'weak';
  const title=isForgot?resetCopy(lang,'forgotTitle'):resetCopy(lang,'resetTitle');
  const subtitle=isForgot?resetCopy(lang,'forgotSub'):resetCopy(lang,'resetSub');
  const apiPath=isForgot?'/auth/forgot-password':'/auth/reset-password';
  const cleanEmail=()=>String(email||'').trim();

  useEffect(()=>{
    if(mode==='reset' && !readSession('sph_reset_token')){
      setMsg({type:'error', text:resetCopy(lang,'expired')});
      const timer=setTimeout(()=>nav('/forgot-password',{replace:true,state:{error:resetCopy(lang,'expired')}}),1200);
      return ()=>clearTimeout(timer);
    }
  },[mode, lang, nav]);

  const handleError=(err)=>{
    const statusCode=err?.response?.status;
    const service=err?.response?.data?.message || err?.response?.data?.error || err?.response?.data?.details;
    let friendly=service || serviceFriendlyError(err,lang);
    if(statusCode===404) friendly = `${friendly} — ${resetCopy(lang,'missingService')}`;
    setMsg({type:'error', text:friendly});
  };

  const sendCode=async()=>{
    const e=cleanEmail();
    if(!e){ setMsg({type:'error',text:resetCopy(lang,'needEmail')}); return; }
    if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e)){ setMsg({type:'error',text:resetCopy(lang,'validEmail')}); return; }
    setLoading(true); setMsg({type:'',text:''});
    try{
      const res=await authAPI.forgotPassword(e);
      const token=getResetTokenFromResponse(res.data || {});
      if(!token){
        setMsg({type:'error', text:lang==='ar'?'الخدمة لم يرجع resetToken. راجع استجابة forgot-password.':'Service did not return resetToken. Check forgot-password response.'});
        return;
      }
      sessionStorage.setItem('sph_reset_email', e);
      sessionStorage.setItem('sph_reset_token', token);
      setResetToken(token);
      setMsg({type:'success', text:res.data?.message || resetCopy(lang,'codeSent')});
      addNotification(res.data?.message || resetCopy(lang,'codeSent'),'success');
      setTimeout(()=>nav(`/reset-password?email=${encodeURIComponent(e)}`),650);
    }catch(err){ handleError(err); }
    finally{ setLoading(false); }
  };

  const changePassword=async()=>{
    const code=String(otp||'').trim();
    const token=resetToken || readSession('sph_reset_token');
    if(!token){ setMsg({type:'error',text:resetCopy(lang,'expired')}); setTimeout(()=>nav('/forgot-password'),900); return; }
    if(!code){ setMsg({type:'error',text:resetCopy(lang,'needOtp')}); return; }
    if(!/^\d{6}$/.test(code)){ setMsg({type:'error',text:resetCopy(lang,'invalidOtp')}); return; }
    if(score<2){ setMsg({type:'error',text:resetCopy(lang,'weakPassword')}); return; }
    if(newPassword!==confirmPassword){ setMsg({type:'error',text:resetCopy(lang,'mismatch')}); return; }
    setLoading(true); setMsg({type:'',text:''});
    try{
      await authAPI.resetPassword({code, newPassword, resetToken:token});
      ['sph_reset_email','sph_reset_token'].forEach(k=>sessionStorage.removeItem(k));
      setMsg({type:'success', text:resetCopy(lang,'passwordChanged')});
      addNotification(resetCopy(lang,'passwordChanged'),'success');
      setTimeout(()=>nav('/login',{replace:true,state:{success:resetCopy(lang,'passwordChanged')}}),850);
    }catch(err){ handleError(err); }
    finally{ setLoading(false); }
  };
  const submit=()=>isForgot?sendCode():changePassword();

  return <main className="auth-page compact-auth-page reset-auth-page smart-auth-polish">
    <div className="auth-left compact-auth-left reset-auth-left">
      <LogoBlock/>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      <div className="auth-compact-benefits reset-steps two-step-reset">
        <span className={isForgot?'active':''}><ShieldCheck size={18}/>{resetCopy(lang,'step1')}</span>
        <span className={!isForgot?'active':''}><Lock size={18}/>{resetCopy(lang,'step2')}</span>
      </div>
      <div className="reset-security-note"><ShieldCheck size={18}/><span>{resetCopy(lang,'tokenSaved')}</span></div>
    </div>
    <section className="auth-card compact-auth-card auth-real-card reset-auth-card smart-card-polish">
      <div className="auth-tools"><LanguageTheme theme={theme} setTheme={setTheme}/></div>
      <div className="reset-flow-badge"><ShieldCheck size={16}/><span>{resetCopy(lang,'apiReady')}</span></div>
      <h2>{title}</h2>
      <p className="muted-copy reset-subcopy">{subtitle}</p>
      {isForgot && <Field label={resetCopy(lang,'email')}><input type="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="user@example.com" required/></Field>}
      {!isForgot && <>
        <Field label={resetCopy(lang,'email')}><input type="email" value={email} disabled className="readonly-input"/></Field>
        <Field label={resetCopy(lang,'otp')}><input inputMode="numeric" maxLength={6} value={otp} onChange={e=>setOtp(e.target.value.replace(/\D/g,'').slice(0,6))} placeholder="123456" required/></Field>
        <Field label={resetCopy(lang,'newPassword')}><input type="password" value={newPassword} onChange={e=>setNewPassword(e.target.value)} required/></Field>
        <div className={`password-strength strength-${status}`}><span>{resetCopy(lang,'strength')}: {resetCopy(lang,status)}</span><i/><i/><i/></div>
        <Field label={resetCopy(lang,'confirmPassword')}><input autoComplete="new-password" type="password" value={confirmPassword} onChange={e=>setConfirmPassword(e.target.value)} required/></Field>
      </>}
      <button className="primary-btn wide" onClick={submit} disabled={loading}>{loading?t('loadingData'):(isForgot?resetCopy(lang,'sendCode'):resetCopy(lang,'changePassword'))}<ChevronRight size={17}/></button>
      {msg.text && <p className={`auth-message ${msg.type==='error'?'error':'success'}`}>{msg.text}</p>}
      <p className="switch-auth"><button type="button" onClick={()=>nav('/login')}>{resetCopy(lang,'backLogin')}</button>{!isForgot&&<button type="button" onClick={()=>nav('/forgot-password')}>{resetCopy(lang,'requestNew')}</button>}</p>

    </section>
  </main>;
}

function Auth({mode, theme, setTheme}){
  const {t, lang}=useLang();
  const nav=useNavigate();
  const location=useLocation();
  const [authMode,setAuthMode]=useState(mode==='register'?'register':'login');
  const isLogin=authMode==='login';
  const [registerStep,setRegisterStep]=useState('form');
  const [msg,setMsg]=useState({type:'', text:''});
  const [email,setEmail]=useState('');
  const [password,setPassword]=useState('');
  const [confirmPassword,setConfirmPassword]=useState('');
  const [firstName,setFirstName]=useState('');
  const [lastName,setLastName]=useState('');
  const [phone,setPhone]=useState('');
  const [address,setAddress]=useState('');
  const [otp,setOtp]=useState('');
  const [registrationToken,setRegistrationToken]=useState('');
  const [authLoading,setAuthLoading]=useState(false);
  // Google OAuth is handled by backend browser redirect only.
  // Do not initialize Google Identity Services here to avoid popup/CORS based auth flows.

  useEffect(()=>{ setAuthMode(mode==='register'?'register':'login'); setRegisterStep('form'); setMsg({type:'',text:''}); setEmail(''); setPassword(''); setConfirmPassword(''); setFirstName(''); setLastName(''); setPhone(''); setAddress(''); setOtp(''); setRegistrationToken(''); }, [mode]);
  useEffect(()=>{
    // Never prefill auth forms from localStorage/sessionStorage or old account cache.
    // Some browsers inject saved credentials after React renders; clear only while the fields are still untouched.
    const clearAutofilledAuthFields=()=>{
      setEmail(prev=>prev ? prev : '');
      setPassword(prev=>prev ? prev : '');
      setConfirmPassword(prev=>prev ? prev : '');
      const authRoot=document.querySelector('.auth-real-card');
      if(!authRoot) return;
      authRoot.querySelectorAll('input').forEach(input=>{
        if(input.matches('[data-auth-clean="true"]') && !input.dataset.userTouched){ input.value=''; }
      });
    };
    const timers=[80,350,900].map(ms=>setTimeout(clearAutofilledAuthFields,ms));
    return ()=>timers.forEach(clearTimeout);
  }, [authMode, registerStep]);
  useEffect(()=>{
    if(location.state?.success){ setMsg({type:'success',text:location.state.success}); window.history.replaceState({}, document.title, window.location.pathname); }
    const googleError = sessionStorage.getItem('ecosense_google_error');
    if(googleError){
      setMsg({type:'error',text:googleError});
      sessionStorage.removeItem('ecosense_google_error');
    }
  }, [location.state]);

  const routeAfterAuth=(user={})=>{
    nav(routeForRole(user), {replace:true});
  };

  const saveSession=async(data={}, fallbackUser={})=>{
    // A new successful login/register must start from a clean runtime cache.
    clearAccountDataCache({includeAllAccounts:true});
    const token=normalizeIncomingAuthToken(data?.token || data?.accessToken || data?.data?.token);
    let user=data?.user || data?.data?.user || fallbackUser;
    if(!token) throw new Error(lang==='ar'?'لم يرجع الخدمة Token صالح.':'Service did not return a valid token.');
    localStorage.setItem('ecosense_token', token);
    localStorage.setItem('sph_token', token);
    try{
      const me=await authAPI.getMe();
      const meUser=me?.data?.user || me?.data?.data?.user || me?.data?.data || me?.data;
      if(meUser && (meUser.email || meUser._id || meUser.id)) user=meUser;
    }catch(_){ /* login token is already valid; /auth/me may fail if service blocks connection/cookies */ }
    const fullName=user?.name || user?.fullName || [user?.firstName,user?.lastName].filter(Boolean).join(' ') || fallbackUser.name || fallbackUser.email;
    let finalUser = {...fallbackUser,...user,name:fullName};
    // Role and ownership must come from the backend token/auth/me response, not from old local worker caches.
    setLoggedInUser(finalUser, token);
    try { connectSocket(finalUser._id || finalUser.id || finalUser.email); } catch {}
    return finalUser;
  };

  const validateLogin=()=>{
    const identifier=String(email||'').trim();
    const cleanPassword=String(password||'');
    if(!identifier || !cleanPassword) return lang==='ar'?'اكتب البريد الإلكتروني أو اسم المستخدم وكلمة المرور أولًا.':'Please enter email/username and password first.';
    if(identifier.includes('@') && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(identifier)) return lang==='ar'?'اكتب بريد إلكتروني صحيح أو اسم مستخدم صحيح.':'Please enter a valid email address or username.';
    if(cleanPassword.length<6) return lang==='ar'?'كلمة المرور يجب ألا تقل عن 6 أحرف.':'Password must be at least 6 characters.';
    return '';
  };

  const validateRegister=()=>{
    const cleanEmail=String(email||'').trim();
    if(!String(firstName||'').trim()) return lang==='ar'?'اكتب الاسم الأول.':'Please enter first name.';
    if(!String(lastName||'').trim()) return lang==='ar'?'اكتب اسم العائلة.':'Please enter last name.';
    if(!cleanEmail) return lang==='ar'?'اكتب البريد الإلكتروني.':'Please enter email.';
    if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) return lang==='ar'?'اكتب بريد إلكتروني صحيح.':'Please enter a valid email address.';
    if(!String(phone||'').trim()) return lang==='ar'?'اكتب رقم الهاتف.':'Please enter phone number.';
    if(!String(address||'').trim()) return lang==='ar'?'اكتب العنوان.':'Please enter address.';
    if(!password || password.length<6) return lang==='ar'?'كلمة المرور يجب ألا تقل عن 6 أحرف.':'Password must be at least 6 characters.';
    if(password!==confirmPassword) return lang==='ar'?'تأكيد كلمة المرور غير مطابق.':'Confirm password does not match.';
    return '';
  };

  const submit=async()=>{
    const validation=isLogin ? validateLogin() : validateRegister();
    if(validation){ setMsg({type:'error',text:validation}); return; }
    setAuthLoading(true); setMsg({type:'',text:''});
    try{
      if(isLogin){
        const identifier=String(email).trim();
        const res=await authAPI.login({identifier,email:identifier,username:identifier,password});
        const user=await saveSession(res.data,{email:identifier.includes('@')?identifier:'',username:identifier,name:identifier.split('@')[0],role:'owner'});
        setMsg({type:'success',text:t('loggedIn')});
        addUserNotification(user.email || email, t('loggedIn'), 'success');
        setTimeout(()=>routeAfterAuth(user),350);
      }else{
        const payload={
          firstName:String(firstName).trim(),
          lastName:String(lastName).trim(),
          email:String(email).trim(),
          password,
          phone:String(phone).trim(),
          phoneNumber:String(phone).trim(),
          address:String(address).trim(),
        };
        const res=await authAPI.register(payload);
        const data=res.data || {};
        const token=data?.token || data?.accessToken || data?.data?.token || data?.data?.accessToken;
        const regToken=data?.registrationToken || data?.registration_token || data?.data?.registrationToken || data?.data?.registration_token || data?.tokenForVerification || data?.data?.tokenForVerification || '';
        if(token){
          const user=await saveSession(data,{...payload,name:`${payload.firstName} ${payload.lastName}`,role:'owner'});
          setMsg({type:'success',text:t('accountCreated')});
          setTimeout(()=>routeAfterAuth(user),350);
        }else{
          setRegistrationToken(regToken);
          if(regToken) sessionStorage.setItem('ecosense_registration_token', regToken);
          setRegisterStep('otp');
          setMsg({type:'success',text:lang==='ar'?'تم إرسال كود التفعيل إلى بريدك الإلكتروني.':'Verification code has been sent to your email.'});
        }
      }
    }catch(err){
      const status=err?.response?.status;
      const service=err?.response?.data?.message || err?.response?.data?.error || err?.message;
      let friendly=service;
      if(status===401) friendly=lang==='ar'?'البريد الإلكتروني أو كلمة المرور غير صحيحة.':'Email or password is incorrect.';
      if(status===403) friendly=lang==='ar'?'الحساب غير مفعل أو لا تملك صلاحية الدخول.':'Account is not verified or access is forbidden.';
      if(status>=500) friendly=lang==='ar'?'حدث خطأ في السيرفر. حاول مرة أخرى لاحقًا.':'Server error. Please try again later.';
      setMsg({type:'error',text:friendly || (lang==='ar'?'فشل الاتصال بالخدمة.':'Service request failed.')});
      clearAccountDataCache({includeAllAccounts:true}); ['sph_auth','ecosense_token','sph_token','sph_role','sph_user_email','ecosense_user','token','user_token','accessToken'].forEach(k=>localStorage.removeItem(k));
    }finally{ setAuthLoading(false); }
  };

  const verifyOtp=async()=>{
    const code=String(otp||'').trim();
    if(!code){ setMsg({type:'error',text:lang==='ar'?'اكتب كود التفعيل.':'Please enter the OTP code.'}); return; }
    setAuthLoading(true); setMsg({type:'',text:''});
    try{
      const regToken = registrationToken || sessionStorage.getItem('ecosense_registration_token') || '';
      const res=await authAPI.verifyOTP({code, otp:code, registrationToken:regToken, registration_token:regToken, email:String(email).trim()});
      const data=res.data || {};
      const token=data?.token || data?.accessToken || data?.data?.token || data?.data?.accessToken;
      if(token){
        const user=await saveSession(data,{email:String(email).trim(),name:`${firstName} ${lastName}`,role:'owner'});
        sessionStorage.removeItem('ecosense_registration_token');
        setRegistrationToken('');
        setMsg({type:'success',text:lang==='ar'?'تم تفعيل الحساب بنجاح.':'Account verified successfully.'});
        setTimeout(()=>routeAfterAuth(user),450);
      }else{
        setMsg({type:'success',text:lang==='ar'?'تم تفعيل الحساب بنجاح. يمكنك تسجيل الدخول الآن.':'Account verified successfully. You can sign in now.'});
        setTimeout(()=>{ switchMode('login'); setPassword(''); setOtp(''); },700);
      }
    }catch(err){
      const service=err?.response?.data?.message || err?.response?.data?.error || err?.message;
      setMsg({type:'error',text:service || (lang==='ar'?'كود التفعيل غير صحيح.':'OTP code is incorrect.')});
    }finally{ setAuthLoading(false); }
  };

  const handleGoogleToken=async(idToken)=>{
    setAuthLoading(true); setMsg({type:'',text:''});
    try{
      const res=await authAPI.googleAuth({idToken, token:idToken, credential:idToken});
      const user=await saveSession(res.data,{email:'google-user@ecosense.local',name:'Google User',role:'owner'});
      setMsg({type:'success',text:lang==='ar'?'تم تسجيل الدخول باستخدام Google.':'Signed in with Google successfully.'});
      setTimeout(()=>routeAfterAuth(user),350);
    }catch(err){
      const service=err?.response?.data?.message || err?.response?.data?.error || err?.message;
      setMsg({type:'error',text:service || (lang==='ar'?'فشل تسجيل الدخول باستخدام Google.':'Google sign-in failed.')});
      clearAccountDataCache({includeAllAccounts:true}); ['sph_auth','ecosense_token','sph_token','sph_role','sph_user_email','ecosense_user','token','user_token','accessToken'].forEach(k=>localStorage.removeItem(k));
    }finally{ setAuthLoading(false); }
  };

  const startGoogle=(e)=>{
    e?.preventDefault?.();
    setAuthLoading(true);
    setMsg({type:'',text:''});

    // Google OAuth must be a full browser redirect to the backend root.
    // Do not call /auth/google with axios/fetch because the OAuth redirect flow will be blocked by CORS.
    const BACKEND_VERCEL_URL = 'https://ecosense-backend.vercel.app';
    const currentFrontendUrl = window.location.origin;
    sessionStorage.setItem('ecosense_google_oauth_started', String(Date.now()));
    window.location.href = `${BACKEND_VERCEL_URL}/api/auth/google?redirect_to=${encodeURIComponent(currentFrontendUrl)}`;
  };

  const switchMode=(next)=>{ setAuthMode(next); setRegisterStep('form'); setMsg({type:'',text:''}); setEmail(''); setPassword(''); setConfirmPassword(''); setFirstName(''); setLastName(''); setPhone(''); setAddress(''); setOtp(''); setRegistrationToken(''); window.history.replaceState(null,'',next==='login'?'/login':'/register'); };

  const googleDisabled=authLoading;
  return <main className="auth-page compact-auth-page">
    <div className="auth-left compact-auth-left">
      <LogoBlock/>
      <h1>{lang==='ar'?'مرحبًا بك في Ecosense AI':'Welcome to Ecosense AI'}</h1>
      <p>{t('authSubtitle')}</p>
      <div className="auth-compact-benefits">
        <span><ScanSearch size={18}/>{t('diagnosisCenter')}</span>
        <span><ShieldCheck size={18}/>{t('finalStatus')}</span>
        <span><ShieldCheck size={18}/>{t('confidence')}</span>
      </div>
    </div>
    <section className="auth-card compact-auth-card auth-real-card" autoComplete="off" data-form-type="other">
      <div className="auth-tools"><LanguageTheme theme={theme} setTheme={setTheme}/></div>
      <div className="auth-switch-tabs">
        <button type="button" className={isLogin?'active':''} onClick={()=>switchMode('login')}>{t('login')}</button>
        <button type="button" className={!isLogin?'active':''} onClick={()=>switchMode('register')}>{t('signup')}</button>
      </div>
      <h2>{isLogin?t('login'):(registerStep==='otp'?(lang==='ar'?'تأكيد البريد الإلكتروني':'Verify Email'):t('createAccount'))}</h2>
      {(!isLogin && registerStep==='otp') ? <div className="otp-panel">
        <p className="muted-copy">{lang==='ar'?`أدخل كود التفعيل الذي تم إرساله إلى ${email}.`:`Enter the OTP code sent to ${email}.`}</p>
        <Field label={lang==='ar'?'كود التفعيل':'OTP Code'}><input inputMode="numeric" autoComplete="one-time-code" value={otp} onChange={e=>setOtp(e.target.value)} placeholder="123456" required/></Field>
        <button className="primary-btn wide" onClick={verifyOtp} disabled={authLoading}>{authLoading?t('loadingData'):(lang==='ar'?'تأكيد الحساب':'Verify Account')}</button>
        <button className="secondary-btn wide" type="button" onClick={()=>setRegisterStep('form')}>{lang==='ar'?'تعديل بيانات التسجيل':'Edit registration details'}</button>
      </div> : <>
        {!isLogin&&<div className="form-grid compact-auth-fields"><Field label={lang==='ar'?'الاسم الأول':'First Name'}><input name="ecosense_register_first_name" autoComplete="off" data-lpignore="true" value={firstName} onChange={e=>setFirstName(e.target.value)} required/></Field><Field label={lang==='ar'?'اسم العائلة':'Last Name'}><input name="ecosense_register_last_name" autoComplete="off" data-lpignore="true" value={lastName} onChange={e=>setLastName(e.target.value)} required/></Field></div>}
        <Field label={isLogin ? (lang==='ar'?'البريد الإلكتروني أو اسم المستخدم':'Email or Username') : t('email')}><input name={isLogin?'ecosense_auth_identifier_no_autofill':'ecosense_register_email_no_autofill'} autoComplete="off" data-auth-clean="true" data-lpignore="true" data-form-type="other" type={isLogin?'text':'email'} value={email} onFocus={e=>{e.currentTarget.dataset.userTouched='true'}} onChange={e=>setEmail(e.target.value)} required/></Field>
        {!isLogin&&<div className="form-grid compact-auth-fields"><Field label={t('phone')}><input name="ecosense_register_phone" autoComplete="off" data-lpignore="true" value={phone} onChange={e=>setPhone(e.target.value)} required/></Field><Field label={lang==='ar'?'العنوان':'Address'}><input name="ecosense_register_address" autoComplete="off" data-lpignore="true" value={address} onChange={e=>setAddress(e.target.value)} required/></Field></div>}
        <Field label={t('password')}><input name={isLogin?'ecosense_auth_password_no_autofill':'ecosense_register_password_no_autofill'} autoComplete="new-password" data-auth-clean="true" data-lpignore="true" data-form-type="other" type="password" value={password} onFocus={e=>{e.currentTarget.dataset.userTouched='true'}} onChange={e=>setPassword(e.target.value)} required/></Field>
        {!isLogin&&<Field label={t('confirmPassword')}><input name="ecosense_confirm_password_no_autofill" autoComplete="new-password" data-auth-clean="true" data-lpignore="true" data-form-type="other" type="password" value={confirmPassword} onFocus={e=>{e.currentTarget.dataset.userTouched='true'}} onChange={e=>setConfirmPassword(e.target.value)} required/></Field>}
        <div className="auth-row auth-row-clean">{isLogin&&<button type="button" className="link-button" onClick={()=>nav('/forgot-password')}>{t('forgot')}</button>}</div>
        <div className="auth-permission-note pro-permission-note">
          <span><ShieldCheck size={18}/></span>
          <div>
            <b>{lang==='ar'?'حسابك يحدد صلاحياتك تلقائيًا':'Your account controls your access'}</b>
            <small>{lang==='ar'?'المالك يرى كل الأدوات، والمدير يرى إدارة المزرعة، والعامل يرى المهام والقطاع المخصص فقط.':'Owners get full access, managers manage farm operations, and workers only see assigned tasks and sectors.'}</small>
          </div>
        </div>
        <button className="primary-btn wide" onClick={submit} disabled={authLoading}>{authLoading?t('loadingData'):(isLogin?t('login'):(lang==='ar'?'متابعة':'Continue'))}</button>
        <button type="button" className="google-btn compact real-google-btn" onClick={startGoogle} disabled={googleDisabled}>
          <GoogleIcon/>{t('continueGoogle')}
        </button>
      </>}
      {msg.text && <p className={`auth-message ${msg.type==='error'?'error':'success'}`}>{msg.text}</p>}
      <p className="switch-auth">
        {isLogin ? t('noAccount') : t('alreadyHave')}
        <button type="button" onClick={()=>switchMode(isLogin?'register':'login')}>{isLogin?t('signup'):t('login')}</button>
      </p>

    </section>
  </main>
}

function GoogleIcon(){return <svg width="20" height="20" viewBox="0 0 48 48" aria-hidden="true"><path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34.1 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z"/><path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34.1 6.1 29.3 4 24 4 16.3 4 9.6 8.3 6.3 14.7z"/><path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.3 0-9.8-3.4-11.4-8.1l-6.5 5C9.4 39.5 16.2 44 24 44z"/><path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.3 4.2-4.1 5.6l6.2 5.2C36.9 39.3 44 34 44 24c0-1.3-.1-2.4-.4-3.5z"/></svg>}
function tone(s=''){ if(String(s).includes('High')) return 'red'; if(String(s).includes('Moderate')) return 'amber'; return 'green'; }
function labelStatus(s,t){ const v=String(s||''); if(!v||v==='-') return '-'; if(v.includes('High')||v.includes('مرتفع')||v.includes('خطر')||v.toLowerCase().includes('critical')) return t('high'); if(v.includes('Moderate')||v.includes('متوسط')) return t('moderate'); if(v.includes('Healthy')||v.includes('سليم')||v.includes('صحي')) return t('healthy'); if(v.includes('No Image')||v.includes('لا توجد صورة')) return t('noImage'); return translateModelText(v,t); }
function download(name,text,type='application/json'){ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type})); a.download=name; a.click(); }

export default App;
