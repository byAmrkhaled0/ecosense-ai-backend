import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { NavLink, Navigate, Outlet, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import {
  Activity, AlertTriangle, BarChart3, Bell, Camera, CheckCircle2, ChevronRight, Cpu,
  Eye, FileText, Globe2, Home, ImagePlus, Layers3, Leaf, Lock, LogIn, LogOut, Menu, Moon,
  Plus, ScanSearch, ShieldCheck, Sparkles, Sprout, Sun, ThermometerSun, UploadCloud,
  UserPlus, X, Zap
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from 'recharts';
import { authAPI, sectorsAPI, usersAPI, notificationsAPI, aiDiagnosisAPI, devicesAPI, reportsAPI, sensorsAPI, API_BASE_URL } from './services/api';

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
    core3: 'Actionable Advice', core3Text: 'Shows diagnosis, severity, backend flags, visual ratios, and recommendations.',
    finalStatus: 'Final Status', sensorStatus: 'Sensor Status', imageStatus: 'Image Status', confidence: 'Confidence', severity: 'Severity', recommendations: 'Recommendations', actions: 'Actions', apiMode: 'Live API', demoMode: 'Demo Mode',
    healthy: 'Healthy', moderate: 'Moderate Stress', high: 'High Stress', noImage: 'No Image', low: 'Low', medium: 'Medium', sufficient: 'Sufficient',
    plantDiagnosis: 'Plant Health Diagnosis', diagnosisSubtitle: 'Upload a plant image and sensor readings to classify the plant condition.',
    cropType: 'Crop Type', tomato: 'Tomato', corn: 'Corn', pepper: 'Pepper', mint: 'Mint', temperature: 'Air Temperature', humidity: 'Humidity', soilMoisture: 'Soil Moisture', soilTemp: 'Soil Temperature', light: 'Light Level',
    uploadImage: 'Upload Plant Image', chooseImage: 'Choose or drag plant image', analyze: 'Analyze Plant', analyzing: 'Analyzing...', result: 'AI Result', diagnosisText: 'Diagnosis', imageAnalysis: 'Image Analysis', greenRatio: 'Green Tissue', yellowRatio: 'Yellowing', brownRatio: 'Brown Tissue', darkSpotRatio: 'Dark Spots', damagedRatio: 'Visible Damage', diseaseName: 'Disease / Visual Problem', diseaseDetails: 'Disease Details', modelResponse: 'Model Response', userExplanation: 'What this means', dataSent: 'Submitted Sensor Data', visualProblem: 'Visual Problem', healthScore: 'Health Score', recommendationCards: 'Recommended Steps', actionCards: 'Immediate Actions', backendFlagsTitle: 'Backend Flags', riskFactorsTitle: 'Risk Factors', noDiseaseDetected: 'No clear disease detected', statusExplanation: 'The final result is produced by fusing sensor_status and image_status, then applying the safety layer when visual disease or dangerous environmental conditions are detected.',
    dashboardTitle: 'Plant Health Overview', dashboardSubtitle: 'Monitor the latest plant status, sensor summary, alerts, and AI trends.', liveReadings: 'Live Readings', latestReading: 'Latest Reading', modelKeywords: 'Model Keywords', fusionLayer: 'Fusion Layer', safetyLayer: 'Safety Layer', backendFlags: 'Backend Flags', riskFactors: 'Risk Factors',
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
    core3: 'توصيات قابلة للتنفيذ', core3Text: 'عرض التشخيص والخطورة وإشارات الباك إند ونسب تحليل الصورة والتوصيات.',
    finalStatus: 'الحالة النهائية', sensorStatus: 'حالة الحساسات', imageStatus: 'حالة الصورة', confidence: 'نسبة الثقة', severity: 'درجة الخطورة', recommendations: 'التوصيات', actions: 'الإجراءات', apiMode: 'API مباشر', demoMode: 'وضع تجريبي',
    healthy: 'سليم', moderate: 'إجهاد متوسط', high: 'إجهاد مرتفع', noImage: 'لا توجد صورة', low: 'منخفضة', medium: 'متوسطة', sufficient: 'كافية',
    plantDiagnosis: 'تشخيص حالة النبات', diagnosisSubtitle: 'ارفع صورة النبات وأدخل قراءات الحساسات لتصنيف حالة النبات.',
    cropType: 'نوع النبات', tomato: 'طماطم', corn: 'ذرة', pepper: 'فلفل', mint: 'نعناع', temperature: 'حرارة الجو', humidity: 'الرطوبة', soilMoisture: 'رطوبة التربة', soilTemp: 'حرارة التربة', light: 'مستوى الإضاءة',
    uploadImage: 'رفع صورة النبات', chooseImage: 'اختر أو اسحب صورة النبات', analyze: 'تحليل النبات', analyzing: 'جاري التحليل...', result: 'نتيجة الذكاء الاصطناعي', diagnosisText: 'التشخيص', imageAnalysis: 'تحليل الصورة', greenRatio: 'النسيج الأخضر', yellowRatio: 'الاصفرار', brownRatio: 'النسيج البني', darkSpotRatio: 'البقع الداكنة', damagedRatio: 'التلف الظاهري', diseaseName: 'اسم المرض / المشكلة البصرية', diseaseDetails: 'تفاصيل المرض', modelResponse: 'استجابة الموديل', userExplanation: 'ماذا تعني النتيجة؟', dataSent: 'بيانات الحساسات المرسلة', visualProblem: 'المشكلة البصرية', healthScore: 'درجة الصحة', recommendationCards: 'الخطوات المقترحة', actionCards: 'الإجراءات الفورية', backendFlagsTitle: 'إشارات الباك إند', riskFactorsTitle: 'عوامل الخطورة', noDiseaseDetected: 'لا يوجد مرض واضح', statusExplanation: 'النتيجة النهائية يتم تكوينها من دمج sensor_status و image_status ثم تطبيق طبقة الأمان عند وجود مرض بصري أو ظروف بيئية خطرة.',
    dashboardTitle: 'نظرة عامة على صحة النبات', dashboardSubtitle: 'تابع آخر حالة للنبات وملخص الحساسات والتنبيهات واتجاهات الذكاء الاصطناعي.', liveReadings: 'قراءات مباشرة', latestReading: 'آخر قراءة', modelKeywords: 'مصطلحات الموديل', fusionLayer: 'طبقة الدمج', safetyLayer: 'طبقة الأمان', backendFlags: 'إشارات الباك إند', riskFactors: 'عوامل الخطورة',
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
    datasetSubtitle: 'Upload CSV/Excel data to prepare it for plant-condition analysis through your backend.',
    uploadDataset: 'Upload Dataset',
    chooseDataset: 'Choose CSV or Excel file',
    datasetReady: 'Dataset selected and ready to send to the backend.',
    datasetName: 'Dataset Name',
    rowsEstimate: 'Estimated Rows',
    sendDataset: 'Send Dataset',
    datasetNote: 'When connected to the backend, this section can upload a full dataset and return batch plant-health results.',
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
    modelDrivenNote: 'Results come from the backend/model. Demo data appears only when the API is unavailable.',
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
    backendChat: 'Chat uses backend when available',
    askRealAssistant: 'Ask real assistant',
    sensorUserHint: 'For users who have sensor values and want to send them to backend then model.',

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
    endpoint: 'Endpoint',
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
    connectBackend: 'Ready to send Authorization Token with each request.',
    plantPhoto: 'Plant Photo',
    noPlants: 'No saved plants yet.',
    openProfile: 'Open Profile',
    riskReasonStable: 'Current readings are within a stable range.',
    riskReasonWaterHeat: 'Possible stress caused by low moisture or high temperature.',
    riskReasonCritical: 'High risk due to critical environmental conditions.',
    saveSuccess: 'Saved successfully.',
    soundOn: 'Sound enabled',
    soundOff: 'Sound disabled',
    datasetPreviewNote: 'Local preview; backend integration can analyze all rows.',
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
    datasetSubtitle: 'ارفع ملف CSV أو Excel لتجهيزه لتحليل حالة النباتات عبر الباك إند.',
    uploadDataset: 'رفع ملف بيانات',
    chooseDataset: 'اختر ملف CSV أو Excel',
    datasetReady: 'تم اختيار الملف وهو جاهز للإرسال للباك إند.',
    datasetName: 'اسم الملف',
    rowsEstimate: 'عدد الصفوف المتوقع',
    sendDataset: 'إرسال الملف',
    datasetNote: 'عند ربط الباك إند، هذا القسم يرفع داتا سيت كاملة ويعرض نتائج تشخيص النباتات دفعة واحدة.',
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
  refreshReadings: 'تحديث القراءات', analyze: 'تحليل النبات', analyzing: 'جاري التحليل...',
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
  exportData: 'تصدير البيانات', datasetAnalyzer: 'تحليل ملف بيانات', datasetNote: 'ارفع ملف CSV أو Excel ليتم تحليله عند ربط الباك إند.',
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
  connectBackend: 'جاهز لإرسال Authorization Token مع كل طلب.',
  plantPhoto: 'صورة النبات',
  noPlants: 'لا توجد نباتات محفوظة بعد.',
  openProfile: 'فتح الملف',
  riskReasonStable: 'القراءات الحالية ضمن نطاق مستقر.',
  riskReasonWaterHeat: 'يوجد احتمال إجهاد بسبب انخفاض الرطوبة أو ارتفاع الحرارة.',
  riskReasonCritical: 'الخطر مرتفع بسبب ظروف بيئية حرجة وقد يحتاج النبات لتدخل عاجل.',
  saveSuccess: 'تم الحفظ بنجاح.',
  soundOn: 'الصوت مفعل',
  soundOff: 'الصوت مغلق',
  datasetPreviewNote: 'هذه معاينة محلية؛ عند ربط الباك إند سيتم إرسال الملف وتحليل كل الصفوف.',
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
  endpoint: 'المسار',
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
  modelDrivenNote: 'كل نتيجة هنا تأتي من الباك إند المتصل بالموديل، والبيانات التجريبية تظهر فقط لو الـ API غير متاح.',
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
  backendChat: 'الشات متصل بالباك إند عند توفره',
  askRealAssistant: 'اسأل المساعد الحقيقي',
  sensorUserHint: 'هذه الصفحة تخدم المستخدم الذي يملك قيم حساسات ويريد إرسالها للباك إند ثم للموديل.',

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
    appName:'Ecosense AI', goodMorning:'Good morning', goodAfternoon:'Good afternoon', goodEvening:'Good evening', overviewSubtitle:'Here’s a complete overview of your farm.', online:'Online', role:'Role', signOut:'Sign out', currentAccount:'Current Account', accountOnline:'Account online', markDone:'Mark as Done', profile:'Profile', farmPreferences:'Farm Preferences', appearance:'Appearance', profileInfo:'Profile Information', accountRolePermissions:'Account role & permissions', modelConnection:'Model connection', modelFinalStatusNote:'The interface always uses backend final_status as the official plant condition in reports and cards.',
    reopenOnboarding:'Open onboarding', onboardingStepAddSector:'Add sector', onboardingStepLinkDevice:'Link device', onboardingStepFirstDiagnosis:'Run first diagnosis', onboardingStepAddWorker:'Add worker', onboardingStepEnableAlerts:'Enable alerts',
    farmManagementTitle:'Farm Management', farmManagementSub:'Manage sectors, devices, farm status, sensor condition, and sector summaries from one clean production view.', farmOverview:'Farm Overview', sectorsDevicesStatus:'Sectors, devices, and farm status', devices:'Devices', sector:'Sector', sectorOverview:'Sector Overview', backToSectors:'Back to Farm Management', sectorSensors:'Sector Sensors', sensorName:'Sensor Name', sensorType:'Sensor Type', sensorValue:'Sensor Value', sensorUnit:'Unit', addSensor:'Add Sensor', assignedWorkers:'Responsible Workers', workersSectionSub:'Manage all workers, assign them to sectors, create accounts, and control permissions.',
    liveStreamMode:'Live Stream Mode', latestImageMode:'Latest Captured Image', cameraOfflineState:'Camera Offline', liveStreamReady:'Live stream placeholder is ready for the backend/hardware feed.', latestCapturedFallback:'Showing the last captured image until live stream is available.', cameraBackendHint:'Ready to receive stream URL, snapshot URL, and camera status from hardware backend.', lastCapturedImage:'Last captured image', retryCamera:'Retry camera', cameraOnline:'Camera online', cameraOffline:'Camera offline',
    viewSensorHistory:'View Sensor History', sensorHistory:'Sensor Reading History', readingsHistorySub:'Previous readings with time, sensor type, sector filters, and trend chart.', filterBySector:'Filter by sector', filterBySensor:'Filter by sensor', allSectors:'All sectors', allSensors:'All sensors', dateTime:'Date & Time', close:'Close',
    addExistingWorker:'Choose existing worker', createNewWorker:'Create new worker', generatedCredentials:'Generated Credentials', username:'Username', workerStatus:'Status', active:'Active', away:'Away', lastTask:'Last Task', permissionsOwnerOnly:'Sensitive services are owner-only.', ownerPermissions:'Owner permissions', workerPermissions:'Worker permissions', accountPasswordHint:'Copy the generated username and password and give them directly to the worker.',
    criticalAlerts:'Critical Alerts', warningAlerts:'Warning Alerts', deviceAlerts:'Device Alerts', diseaseAlerts:'Disease Alerts', sensorOfflineAlerts:'Sensor Offline Alerts', alertReason:'Alert reason', alertSeverity:'Severity', viewDiagnosis:'View Diagnosis',
    reportPreview:'Diagnosis Report Preview', exportPdf:'Export PDF', exportCsv:'Export CSV', filterByDate:'Filter by date', filterByStatus:'Filter by status', filterByStatusAll:'All statuses', workerTask:'Worker task', diagnosisTime:'Diagnosis time', finalStatusPrimary:'final_status is the primary status used by the website and reports. sensor_status and image_status are explanatory only.',
    mobileQuickDiagnosis:'Quick mobile diagnosis', takePlantPhoto:'Take plant photo', tasks:'Tasks', myTasks:'My Tasks', assignedSector:'Assigned Sector', pending:'Pending', completed:'Completed', startTask:'Start Task', items:'items', ownerOnlyAction:'Owner only action', workerLimitedView:'Worker limited view',
    farmManagerPermissions:'Farm Manager permissions', workerSettingsOnly:'Worker account settings only', activeWorkersNow:'Active workers now', lastSeen:'Last seen', onlineNow:'Online now', offlineNow:'Offline', currentTask:'Current task', deviceDetails:'Device Details', farmAlerts:'Farm Alerts', farmReports:'Farm Reports', operationalReady:'Operational and backend-ready', backendStructureReady:'Backend structure ready', linkedBackendFeature:'This feature is functional locally and ready for backend sync.', deviceReconnectRequested:'Device reconnect request sent.', deviceRemoved:'Device removed.', noBackendRealtime:'Real-time backend is not connected yet; local presence structure is ready.', systemMode:'System Mode', accountStatus:'Account Status', copyUsername:'Copy Username', copyPassword:'Copy Password', copyAll:'Copy All', ownerSensitiveBlocked:'Owner-sensitive action blocked for this role', reportsManagement:'Reports Management', alertsManagement:'Alerts Management', devicesManagement:'Devices Management', tasksManagement:'Tasks Management', viewFullReport:'View Full Report', openPlatform:'Open Platform', lowSoilMoistureAlert:'Alert: soil moisture is low', highTemperatureAlert:'Alert: temperature is high', fungalRiskAlert:'Alert: possible fungal infection',
    takePhoto:'Take Photo', chooseFromGallery:'Choose from Gallery', changeImage:'Change Image', removeImage:'Remove Image', imagePreview:'Image Preview', mobileUploadHint:'On mobile, choose camera or gallery before analysis.', backendSensorReadings:'Backend Sensor Readings', refreshSensors:'Refresh Sensors', simulateReadings:'Simulate Readings', sensorLoading:'Loading sensor readings...', sensorConnected:'Connected to backend sensors', sensorOfflineState:'Sensors offline', sensorErrorState:'Could not load sensor readings', simulationMode:'Simulation Mode', backendMode:'Backend Mode', lastUpdated:'Last Updated', noSensorData:'No sensor data available', retry:'Retry', latestBackendImage:'Latest image from backend', combinedReady:'Image and readings are ready for combined diagnosis', backendRequired:'Backend required', imageRequired:'Please add or capture a plant image first.', diagnosisSuccess:'Diagnosis completed successfully.', diagnosisFallback:'API unavailable. Simulation result displayed clearly.', sensorsUpdated:'Sensor readings updated.', sensorsSimulationUpdated:'Simulation readings refreshed.', sensorsOfflineFallback:'Sensors backend is offline. Simulation mode is available.', imageRemoved:'Image removed.', imageSelected:'Image selected successfully.', cameraCapture:'Open camera', gallerySelect:'Open gallery', takePhotoHint:'Open the mobile camera', galleryHint:'Select an existing image', imageReady:'Image ready', mobileImagePickerNote:'Use camera or gallery, then review the preview before analysis'
  },
  ar: {
    appName:'Ecosense AI', goodMorning:'صباح الخير', goodAfternoon:'مساء الخير', goodEvening:'مساء الخير', overviewSubtitle:'إليك نظرة عامة على حالة مزرعتك.', online:'متصل', role:'الدور', signOut:'تسجيل الخروج', currentAccount:'الحساب الحالي', accountOnline:'الحساب متصل', markDone:'إنهاء المهمة', profile:'البروفايل', farmPreferences:'تفضيلات المزرعة', appearance:'المظهر', profileInfo:'بيانات البروفايل', accountRolePermissions:'نوع الحساب والصلاحيات', modelConnection:'اتصال الموديل', modelFinalStatusNote:'الواجهة تعتمد دائمًا على final_status القادم من الباك إند كحالة النبات الرسمية داخل التقارير والكروت.',
    reopenOnboarding:'فتح التهيئة الأولى', onboardingStepAddSector:'أضف قطاع', onboardingStepLinkDevice:'اربط جهاز', onboardingStepFirstDiagnosis:'اعمل أول تشخيص', onboardingStepAddWorker:'أضف عامل', onboardingStepEnableAlerts:'فعل التنبيهات',
    farmManagementTitle:'إدارة المزرعة', farmManagementSub:'إدارة القطاعات والأجهزة وحالة المزرعة وحالة الحساسات وملخص كل قطاع من صفحة واحدة منظمة.', farmOverview:'نظرة عامة على المزرعة', sectorsDevicesStatus:'القطاعات والأجهزة وحالة المزرعة', devices:'الأجهزة', sector:'القطاع', sectorOverview:'ملخص القطاع', backToSectors:'الرجوع لإدارة المزرعة', sectorSensors:'حساسات القطاع', sensorName:'اسم الحساس', sensorType:'نوع الحساس', sensorValue:'قيمة الحساس', sensorUnit:'الوحدة', addSensor:'إضافة حساس', assignedWorkers:'العمال المسؤولون', workersSectionSub:'إدارة كل العمال وتعيينهم للقطاعات وإنشاء حسابات وصلاحيات لكل عامل.',
    liveStreamMode:'وضع البث المباشر', latestImageMode:'آخر صورة ملتقطة', cameraOfflineState:'الكاميرا غير متصلة', liveStreamReady:'مربع البث جاهز لاستقبال رابط البث من الباك إند والهاردوير.', latestCapturedFallback:'يتم عرض آخر صورة ملتقطة لحين توفر البث المباشر.', cameraBackendHint:'جاهز لاستقبال stream URL و snapshot URL وحالة الكاميرا من باك إند الهاردوير.', lastCapturedImage:'آخر صورة ملتقطة', retryCamera:'إعادة محاولة الكاميرا', cameraOnline:'الكاميرا متصلة', cameraOffline:'الكاميرا غير متصلة',
    viewSensorHistory:'سجل القراءات', sensorHistory:'سجل قراءات الحساسات', readingsHistorySub:'قراءات سابقة بالوقت ونوع الحساس والقطاع مع فلاتر ورسم بياني بسيط.', filterBySector:'فلترة حسب القطاع', filterBySensor:'فلترة حسب الحساس', allSectors:'كل القطاعات', allSensors:'كل الحساسات', dateTime:'الوقت والتاريخ', close:'إغلاق',
    addExistingWorker:'اختيار عامل موجود', createNewWorker:'إنشاء عامل جديد', generatedCredentials:'بيانات الحساب الناتج', username:'اسم المستخدم', workerStatus:'الحالة', active:'نشط', away:'غير متاح', lastTask:'آخر مهمة', permissionsOwnerOnly:'الخدمات الحساسة مخصصة للمالك فقط.', ownerPermissions:'صلاحيات المالك', workerPermissions:'صلاحيات العامل', accountPasswordHint:'انسخ اسم المستخدم وكلمة المرور وسلمهم للعامل مباشرة.',
    criticalAlerts:'تنبيهات حرجة', warningAlerts:'تنبيهات تحذيرية', deviceAlerts:'تنبيهات الأجهزة', diseaseAlerts:'تنبيهات الأمراض', sensorOfflineAlerts:'حساس غير متصل', alertReason:'سبب التنبيه', alertSeverity:'درجة الخطورة', viewDiagnosis:'عرض التشخيص',
    reportPreview:'معاينة تقرير التشخيص', exportPdf:'تصدير PDF', exportCsv:'تصدير CSV', filterByDate:'فلترة بالتاريخ', filterByStatus:'فلترة بالحالة', filterByStatusAll:'كل الحالات', workerTask:'مهمة العامل', diagnosisTime:'وقت التشخيص', finalStatusPrimary:'final_status هي الحالة الأساسية التي يعتمد عليها الموقع والتقارير، أما sensor_status و image_status فهي للشرح فقط.',
    mobileQuickDiagnosis:'تشخيص سريع من الهاتف', takePlantPhoto:'التقاط صورة النبات', tasks:'المهام', myTasks:'مهامي', assignedSector:'القطاع المسؤول عنه', pending:'قيد الانتظار', completed:'مكتملة', startTask:'بدء المهمة', items:'عناصر', ownerOnlyAction:'إجراء خاص بالمالك', workerLimitedView:'عرض محدود للعامل',
    farmManagerPermissions:'صلاحيات مدير المزرعة', workerSettingsOnly:'إعدادات حساب العامل فقط', activeWorkersNow:'العمال النشطين الآن', lastSeen:'آخر ظهور', onlineNow:'متصل الآن', offlineNow:'غير متصل', currentTask:'المهمة الحالية', deviceDetails:'تفاصيل الجهاز', farmAlerts:'تنبيهات المزرعة', farmReports:'تقارير المزرعة', operationalReady:'شغال وجاهز للربط', backendStructureReady:'هيكل جاهز للباك إند', linkedBackendFeature:'الميزة تعمل محليًا وجاهزة للمزامنة مع الباك إند.', deviceReconnectRequested:'تم إرسال طلب إعادة الاتصال بالجهاز.', deviceRemoved:'تم حذف الجهاز.', noBackendRealtime:'لا يوجد باك إند لحظي متصل الآن؛ هيكل حالة النشاط جاهز للربط.', systemMode:'وضع النظام', accountStatus:'حالة الحساب', copyUsername:'نسخ اسم المستخدم', copyPassword:'نسخ كلمة المرور', copyAll:'نسخ الكل', ownerSensitiveBlocked:'تم منع إجراء حساس مخصص للمالك فقط', reportsManagement:'إدارة التقارير', alertsManagement:'إدارة التنبيهات', devicesManagement:'إدارة الأجهزة', tasksManagement:'إدارة المهام', viewFullReport:'عرض التقرير كامل', openPlatform:'دخول المنصة', lowSoilMoistureAlert:'تنبيه: رطوبة التربة منخفضة', highTemperatureAlert:'تنبيه: درجة الحرارة مرتفعة', fungalRiskAlert:'تنبيه: احتمال إصابة فطرية'
  },
  fr: { profile:'Profil', farmPreferences:'Préférences ferme', appearance:'Apparence', profileInfo:'Informations du profil', accountRolePermissions:'Rôle et permissions', modelConnection:'Connexion modèle', modelFinalStatusNote:'L’interface utilise toujours final_status comme état officiel.', reopenOnboarding:'Ouvrir l’onboarding', onboardingStepAddSector:'Ajouter un secteur', onboardingStepLinkDevice:'Lier un appareil', onboardingStepFirstDiagnosis:'Premier diagnostic', onboardingStepAddWorker:'Ajouter un ouvrier', onboardingStepEnableAlerts:'Activer les alertes', farmManagementTitle:'Gestion de la ferme', farmManagementSub:'Gérez secteurs, appareils, état de la ferme et capteurs.', farmOverview:'Vue ferme', sectorsDevicesStatus:'Secteurs, appareils et état', devices:'Appareils', sector:'Secteur', sectorOverview:'Aperçu secteur', backToSectors:'Retour gestion ferme', sectorSensors:'Capteurs secteur', sensorName:'Nom du capteur', sensorType:'Type de capteur', sensorValue:'Valeur', sensorUnit:'Unité', addSensor:'Ajouter capteur', assignedWorkers:'Ouvriers responsables', workersSectionSub:'Gérez les ouvriers, affectations, comptes et permissions.', liveStreamMode:'Mode direct', latestImageMode:'Dernière image', cameraOfflineState:'Caméra hors ligne', liveStreamReady:'Flux prêt pour backend/hardware.', latestCapturedFallback:'Dernière image affichée si le direct est indisponible.', cameraBackendHint:'Prêt pour URL stream, snapshot et statut caméra.', lastCapturedImage:'Dernière capture', retryCamera:'Réessayer', cameraOnline:'Caméra en ligne', cameraOffline:'Caméra hors ligne', viewSensorHistory:'Historique capteurs', sensorHistory:'Historique des mesures', readingsHistorySub:'Mesures précédentes avec filtres et graphique.', filterBySector:'Filtrer par secteur', filterBySensor:'Filtrer par capteur', allSectors:'Tous les secteurs', allSensors:'Tous les capteurs', dateTime:'Date et heure', close:'Fermer', addExistingWorker:'Choisir ouvrier existant', createNewWorker:'Créer nouvel ouvrier', generatedCredentials:'Identifiants générés', username:'Nom utilisateur', workerStatus:'Statut', active:'Actif', away:'Absent', lastTask:'Dernière tâche', permissionsOwnerOnly:'Services sensibles réservés au propriétaire.', ownerPermissions:'Permissions propriétaire', workerPermissions:'Permissions ouvrier', accountPasswordHint:'Copiez l’utilisateur et le mot de passe.', criticalAlerts:'Alertes critiques', warningAlerts:'Alertes avertissement', deviceAlerts:'Alertes appareils', diseaseAlerts:'Alertes maladie', sensorOfflineAlerts:'Capteur hors ligne', alertReason:'Raison', alertSeverity:'Gravité', viewDiagnosis:'Voir diagnostic', reportPreview:'Aperçu du rapport', exportPdf:'Exporter PDF', exportCsv:'Exporter CSV', filterByDate:'Filtrer par date', filterByStatus:'Filtrer par état', filterByStatusAll:'Tous les états', workerTask:'Tâche ouvrier', diagnosisTime:'Heure diagnostic', finalStatusPrimary:'final_status est l’état principal; sensor_status et image_status sont explicatifs.', tasks:'Tâches', items:'éléments' },
  es: { profile:'Perfil', farmPreferences:'Preferencias de granja', appearance:'Apariencia', profileInfo:'Información del perfil', accountRolePermissions:'Rol y permisos', modelConnection:'Conexión del modelo', modelFinalStatusNote:'La interfaz usa siempre final_status como estado oficial.', reopenOnboarding:'Abrir onboarding', onboardingStepAddSector:'Agregar sector', onboardingStepLinkDevice:'Vincular dispositivo', onboardingStepFirstDiagnosis:'Primer diagnóstico', onboardingStepAddWorker:'Agregar trabajador', onboardingStepEnableAlerts:'Activar alertas', farmManagementTitle:'Gestión de granja', farmManagementSub:'Gestiona sectores, dispositivos, estado y sensores.', farmOverview:'Vista de granja', sectorsDevicesStatus:'Sectores, dispositivos y estado', devices:'Dispositivos', sector:'Sector', sectorOverview:'Resumen del sector', backToSectors:'Volver a gestión', sectorSensors:'Sensores del sector', sensorName:'Nombre del sensor', sensorType:'Tipo de sensor', sensorValue:'Valor', sensorUnit:'Unidad', addSensor:'Agregar sensor', assignedWorkers:'Trabajadores responsables', workersSectionSub:'Gestiona trabajadores, asignaciones, cuentas y permisos.', liveStreamMode:'Modo en vivo', latestImageMode:'Última imagen', cameraOfflineState:'Cámara sin conexión', liveStreamReady:'Stream listo para backend/hardware.', latestCapturedFallback:'Mostrando última captura si no hay vivo.', cameraBackendHint:'Listo para URL de stream, snapshot y estado.', lastCapturedImage:'Última captura', retryCamera:'Reintentar', cameraOnline:'Cámara online', cameraOffline:'Cámara offline', viewSensorHistory:'Historial de sensores', sensorHistory:'Historial de lecturas', readingsHistorySub:'Lecturas anteriores con filtros y gráfico.', filterBySector:'Filtrar por sector', filterBySensor:'Filtrar por sensor', allSectors:'Todos los sectores', allSensors:'Todos los sensores', dateTime:'Fecha y hora', close:'Cerrar', addExistingWorker:'Elegir trabajador existente', createNewWorker:'Crear trabajador', generatedCredentials:'Credenciales generadas', username:'Usuario', workerStatus:'Estado', active:'Activo', away:'Ausente', lastTask:'Última tarea', permissionsOwnerOnly:'Servicios sensibles solo para propietario.', ownerPermissions:'Permisos propietario', workerPermissions:'Permisos trabajador', accountPasswordHint:'Copia usuario y contraseña.', criticalAlerts:'Alertas críticas', warningAlerts:'Alertas de advertencia', deviceAlerts:'Alertas de dispositivos', diseaseAlerts:'Alertas de enfermedad', sensorOfflineAlerts:'Sensor sin conexión', alertReason:'Razón', alertSeverity:'Severidad', viewDiagnosis:'Ver diagnóstico', reportPreview:'Vista previa del reporte', exportPdf:'Exportar PDF', exportCsv:'Exportar CSV', filterByDate:'Filtrar por fecha', filterByStatus:'Filtrar por estado', filterByStatusAll:'Todos los estados', workerTask:'Tarea trabajador', diagnosisTime:'Hora diagnóstico', finalStatusPrimary:'final_status es el estado principal; sensor_status e image_status son explicativos.', tasks:'Tareas', items:'elementos' },
  de: { profile:'Profil', farmPreferences:'Farm-Einstellungen', appearance:'Darstellung', profileInfo:'Profilinformationen', accountRolePermissions:'Rolle und Rechte', modelConnection:'Modellverbindung', modelFinalStatusNote:'Die Oberfläche nutzt immer final_status als offiziellen Status.', reopenOnboarding:'Onboarding öffnen', onboardingStepAddSector:'Sektor hinzufügen', onboardingStepLinkDevice:'Gerät verbinden', onboardingStepFirstDiagnosis:'Erste Diagnose', onboardingStepAddWorker:'Arbeiter hinzufügen', onboardingStepEnableAlerts:'Warnungen aktivieren', farmManagementTitle:'Farmverwaltung', farmManagementSub:'Sektoren, Geräte, Farmstatus und Sensorstatus verwalten.', farmOverview:'Farmübersicht', sectorsDevicesStatus:'Sektoren, Geräte und Status', devices:'Geräte', sector:'Sektor', sectorOverview:'Sektorübersicht', backToSectors:'Zur Farmverwaltung', sectorSensors:'Sektorsensoren', sensorName:'Sensorname', sensorType:'Sensortyp', sensorValue:'Wert', sensorUnit:'Einheit', addSensor:'Sensor hinzufügen', assignedWorkers:'Zuständige Arbeiter', workersSectionSub:'Arbeiter, Zuweisungen, Konten und Rechte verwalten.', liveStreamMode:'Live-Stream', latestImageMode:'Letztes Bild', cameraOfflineState:'Kamera offline', liveStreamReady:'Stream bereit für Backend/Hardware.', latestCapturedFallback:'Letzte Aufnahme wird angezeigt.', cameraBackendHint:'Bereit für Stream-URL, Snapshot und Kamerastatus.', lastCapturedImage:'Letzte Aufnahme', retryCamera:'Erneut versuchen', cameraOnline:'Kamera online', cameraOffline:'Kamera offline', viewSensorHistory:'Sensorverlauf', sensorHistory:'Messwertverlauf', readingsHistorySub:'Frühere Messwerte mit Filtern und Diagramm.', filterBySector:'Nach Sektor filtern', filterBySensor:'Nach Sensor filtern', allSectors:'Alle Sektoren', allSensors:'Alle Sensoren', dateTime:'Datum & Zeit', close:'Schließen', addExistingWorker:'Vorhandenen Arbeiter wählen', createNewWorker:'Neuen Arbeiter erstellen', generatedCredentials:'Generierte Zugangsdaten', username:'Benutzername', workerStatus:'Status', active:'Aktiv', away:'Abwesend', lastTask:'Letzte Aufgabe', permissionsOwnerOnly:'Sensible Dienste nur für Eigentümer.', ownerPermissions:'Eigentümerrechte', workerPermissions:'Arbeiterrechte', accountPasswordHint:'Benutzername und Passwort kopieren.', criticalAlerts:'Kritische Warnungen', warningAlerts:'Warnungen', deviceAlerts:'Gerätewarnungen', diseaseAlerts:'Krankheitswarnungen', sensorOfflineAlerts:'Sensor offline', alertReason:'Grund', alertSeverity:'Schweregrad', viewDiagnosis:'Diagnose ansehen', reportPreview:'Berichtsvorschau', exportPdf:'PDF exportieren', exportCsv:'CSV exportieren', filterByDate:'Nach Datum filtern', filterByStatus:'Nach Status filtern', filterByStatusAll:'Alle Status', workerTask:'Arbeiteraufgabe', diagnosisTime:'Diagnosezeit', finalStatusPrimary:'final_status ist der Hauptstatus; sensor_status und image_status dienen der Erklärung.', tasks:'Aufgaben', items:'Elemente' },
  it: { profile:'Profilo', farmPreferences:'Preferenze fattoria', appearance:'Aspetto', profileInfo:'Informazioni profilo', accountRolePermissions:'Ruolo e permessi', modelConnection:'Connessione modello', modelFinalStatusNote:'L’interfaccia usa sempre final_status come stato ufficiale.', reopenOnboarding:'Apri onboarding', onboardingStepAddSector:'Aggiungi settore', onboardingStepLinkDevice:'Collega dispositivo', onboardingStepFirstDiagnosis:'Prima diagnosi', onboardingStepAddWorker:'Aggiungi operaio', onboardingStepEnableAlerts:'Attiva avvisi', farmManagementTitle:'Gestione fattoria', farmManagementSub:'Gestisci settori, dispositivi, stato e sensori.', farmOverview:'Vista fattoria', sectorsDevicesStatus:'Settori, dispositivi e stato', devices:'Dispositivi', sector:'Settore', sectorOverview:'Panoramica settore', backToSectors:'Torna alla gestione', sectorSensors:'Sensori settore', sensorName:'Nome sensore', sensorType:'Tipo sensore', sensorValue:'Valore', sensorUnit:'Unità', addSensor:'Aggiungi sensore', assignedWorkers:'Operatori responsabili', workersSectionSub:'Gestisci operatori, assegnazioni, account e permessi.', liveStreamMode:'Modalità live', latestImageMode:'Ultima immagine', cameraOfflineState:'Camera offline', liveStreamReady:'Stream pronto per backend/hardware.', latestCapturedFallback:'Ultima cattura mostrata se live non disponibile.', cameraBackendHint:'Pronto per URL stream, snapshot e stato.', lastCapturedImage:'Ultima cattura', retryCamera:'Riprova', cameraOnline:'Camera online', cameraOffline:'Camera offline', viewSensorHistory:'Storico sensori', sensorHistory:'Storico letture', readingsHistorySub:'Letture precedenti con filtri e grafico.', filterBySector:'Filtra per settore', filterBySensor:'Filtra per sensore', allSectors:'Tutti i settori', allSensors:'Tutti i sensori', dateTime:'Data e ora', close:'Chiudi', addExistingWorker:'Scegli operaio esistente', createNewWorker:'Crea operaio', generatedCredentials:'Credenziali generate', username:'Username', workerStatus:'Stato', active:'Attivo', away:'Assente', lastTask:'Ultimo task', permissionsOwnerOnly:'Servizi sensibili solo al proprietario.', ownerPermissions:'Permessi proprietario', workerPermissions:'Permessi operaio', accountPasswordHint:'Copia username e password.', criticalAlerts:'Avvisi critici', warningAlerts:'Avvisi warning', deviceAlerts:'Avvisi dispositivi', diseaseAlerts:'Avvisi malattia', sensorOfflineAlerts:'Sensore offline', alertReason:'Motivo', alertSeverity:'Gravità', viewDiagnosis:'Vedi diagnosi', reportPreview:'Anteprima report', exportPdf:'Esporta PDF', exportCsv:'Esporta CSV', filterByDate:'Filtra per data', filterByStatus:'Filtra per stato', filterByStatusAll:'Tutti gli stati', workerTask:'Task operaio', diagnosisTime:'Ora diagnosi', finalStatusPrimary:'final_status è lo stato principale; sensor_status e image_status sono esplicativi.', tasks:'Task', items:'elementi' },
  tr: { profile:'Profil', farmPreferences:'Çiftlik tercihleri', appearance:'Görünüm', profileInfo:'Profil bilgileri', accountRolePermissions:'Rol ve izinler', modelConnection:'Model bağlantısı', modelFinalStatusNote:'Arayüz resmi durum olarak her zaman final_status kullanır.', reopenOnboarding:'Onboarding aç', onboardingStepAddSector:'Sektör ekle', onboardingStepLinkDevice:'Cihaz bağla', onboardingStepFirstDiagnosis:'İlk teşhis', onboardingStepAddWorker:'Çalışan ekle', onboardingStepEnableAlerts:'Uyarıları aç', farmManagementTitle:'Çiftlik Yönetimi', farmManagementSub:'Sektörleri, cihazları, çiftlik durumunu ve sensörleri yönetin.', farmOverview:'Çiftlik görünümü', sectorsDevicesStatus:'Sektörler, cihazlar ve durum', devices:'Cihazlar', sector:'Sektör', sectorOverview:'Sektör özeti', backToSectors:'Çiftlik yönetimine dön', sectorSensors:'Sektör sensörleri', sensorName:'Sensör adı', sensorType:'Sensör tipi', sensorValue:'Değer', sensorUnit:'Birim', addSensor:'Sensör ekle', assignedWorkers:'Sorumlu çalışanlar', workersSectionSub:'Çalışanları, atamaları, hesapları ve izinleri yönetin.', liveStreamMode:'Canlı yayın modu', latestImageMode:'Son yakalanan görüntü', cameraOfflineState:'Kamera çevrimdışı', liveStreamReady:'Backend/hardware canlı yayın için hazır.', latestCapturedFallback:'Canlı yayın yoksa son görüntü gösterilir.', cameraBackendHint:'Stream URL, snapshot ve kamera durumu için hazır.', lastCapturedImage:'Son görüntü', retryCamera:'Kamerayı yeniden dene', cameraOnline:'Kamera çevrimiçi', cameraOffline:'Kamera çevrimdışı', viewSensorHistory:'Sensör geçmişi', sensorHistory:'Okuma geçmişi', readingsHistorySub:'Filtreler ve grafik ile önceki okumalar.', filterBySector:'Sektöre göre filtrele', filterBySensor:'Sensöre göre filtrele', allSectors:'Tüm sektörler', allSensors:'Tüm sensörler', dateTime:'Tarih ve saat', close:'Kapat', addExistingWorker:'Mevcut çalışan seç', createNewWorker:'Yeni çalışan oluştur', generatedCredentials:'Oluşturulan bilgiler', username:'Kullanıcı adı', workerStatus:'Durum', active:'Aktif', away:'Uzakta', lastTask:'Son görev', permissionsOwnerOnly:'Hassas hizmetler sadece sahibindir.', ownerPermissions:'Sahip izinleri', workerPermissions:'Çalışan izinleri', accountPasswordHint:'Kullanıcı adı ve şifreyi kopyalayın.', criticalAlerts:'Kritik uyarılar', warningAlerts:'Uyarılar', deviceAlerts:'Cihaz uyarıları', diseaseAlerts:'Hastalık uyarıları', sensorOfflineAlerts:'Sensör çevrimdışı', alertReason:'Neden', alertSeverity:'Şiddet', viewDiagnosis:'Teşhisi görüntüle', reportPreview:'Rapor önizleme', exportPdf:'PDF dışa aktar', exportCsv:'CSV dışa aktar', filterByDate:'Tarihe göre filtrele', filterByStatus:'Duruma göre filtrele', filterByStatusAll:'Tüm durumlar', workerTask:'Çalışan görevi', diagnosisTime:'Teşhis zamanı', finalStatusPrimary:'final_status ana durumdur; sensor_status ve image_status açıklama içindir.', tasks:'Görevler', items:'öğe' }
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
    backendRequired: 'Backend Required', sensorsApiNotConnected: 'Sensors API is not connected yet', simulationMode: 'Simulation Mode', retryConnection: 'Retry Connection',
    assistantMobileNote: 'On mobile, the assistant opens as a compact bottom sheet.', askAnythingPlaceholder: 'Ask anything about agriculture or plant health...',
    accountAccessHint: 'Your account controls your access. Owners get full access; workers only see assigned tools and tasks.',
    viewFullReport: 'View Full Report', plantPhoto: 'Plant photo', emptyState: 'No data available yet',
    loginSecurityTitle: 'Your account controls your access', loginSecurityBody: 'Owners get full access; workers only see assigned tools and tasks.',
    backendSensorStatus: 'Backend sensor status', noRepeatedRequests: 'Automatic sensor polling is disabled until the backend endpoint is available.'
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
    backendRequired: 'يتطلب الباك إند', sensorsApiNotConnected: 'واجهة حساسات الباك إند غير متصلة حتى الآن', simulationMode: 'وضع المحاكاة', retryConnection: 'إعادة المحاولة',
    assistantMobileNote: 'على الموبايل يفتح المساعد كنافذة سفلية مدمجة.', askAnythingPlaceholder: 'اسأل عن الزراعة أو صحة النبات...',
    accountAccessHint: 'حسابك يحدد صلاحيات الوصول. المالك لديه صلاحيات كاملة، والعامل يرى الأدوات والمهام المخصصة له فقط.',
    viewFullReport: 'عرض التقرير كامل', plantPhoto: 'صورة النبات', emptyState: 'لا توجد بيانات متاحة بعد',
    loginSecurityTitle: 'حسابك يحدد صلاحيات الوصول', loginSecurityBody: 'المالك لديه صلاحيات كاملة، والعامل يرى الأدوات والمهام المخصصة له فقط.',
    backendSensorStatus: 'حالة باك إند الحساسات', noRepeatedRequests: 'تم إيقاف طلبات الحساسات التلقائية حتى يتم توفير Endpoint الباك إند.'
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
      <div><b>${esc(t('workerTask'))}</b><span>${esc(translateModelText(r.workerTask || '-',t))}</span></div>
    </div>
    ${r.image ? `<img class="plant-img" src="${esc(r.image)}" alt="Plant image"/>` : ''}
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
    <section class="cover"><div class="brand"><img class="logo" src="/project-logo.png"/><div><h1>${esc(t(titleKey))}</h1><p>${esc(t('appName'))} — Smart Plant Health</p><p>${esc(t('finalStatusPrimary'))}</p></div></div><button class="no-print" onclick="window.print()">${esc(t('printOrSavePdf'))}</button></section>
    <section class="meta"><div class="meta-grid">
      <div><b>${esc(t('farmName'))}</b><span>${esc(settings.farmName || 'Ecosense Smart Farm')}</span></div>
      <div><b>${esc(t('reportGeneratedBy'))}</b><span>${esc(userName)}</span></div>
      <div><b>${esc(t('role'))}</b><span>${esc(role)}</span></div>
      <div><b>${esc(t('date'))}</b><span>${esc(now)}</span></div>
    </div><p class="note">${esc(t('reportOwnerNameNote'))}</p></section>
    ${renderRows || `<section class="report-card"><p>${esc(t('emptyState'))}</p></section>`}
  </div><script>setTimeout(()=>window.print(),500)</script></body></html>`;
}
function openProfessionalReport(args){
  const html = professionalReportHtml(args);
  const w = window.open('', '_blank', 'width=980,height=900');
  if(w){ w.document.open(); w.document.write(html); w.document.close(); }
  else { download('ecosense-professional-report.html', html, 'text/html'); }
  try{ addNotification(args.t('reportExportReady'),'success'); }catch{}
}

function localizeValue(value, t) {
  const v = String(value ?? '');
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
  return map[v] || value;
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
    const s = String(txt).trim();
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
    sensorsMergedNotice: 'Sensors were moved into Farm Management to reduce repeated pages.',
    devicesAndSensors: 'Devices & Sensors',
    alertsSummary: 'Alerts Summary',
    reportsSummary: 'Reports Summary',
    tasksSummary: 'Tasks Summary',
    currentSensorReadings: 'Current Sensor Readings',
    sensorSource: 'Sensor Source',
    liveSource: 'Live',
    simulationSource: 'Simulation',
    backendRequiredSource: 'Backend Required',
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
    sensorsMergedNotice: 'تم نقل الحساسات داخل إدارة المزرعة لتقليل الصفحات المتكررة.',
    devicesAndSensors: 'الأجهزة والحساسات',
    alertsSummary: 'ملخص التنبيهات',
    reportsSummary: 'ملخص التقارير',
    tasksSummary: 'ملخص المهام',
    currentSensorReadings: 'قراءات الحساسات الحالية',
    sensorSource: 'مصدر الحساسات',
    liveSource: 'مباشر',
    simulationSource: 'محاكاة',
    backendRequiredSource: 'يتطلب باك إند',
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

const initialReadings = [
  { id: 1, time: '10:15', cropType: 'Tomato', temperature: 25, humidity: 60, soilMoisture: 45, soilTemp: 24, light: 'Sufficient', final_status: 'Healthy', confidence: .91 },
  { id: 2, time: '09:00', cropType: 'Pepper', temperature: 34, humidity: 42, soilMoisture: 29, soilTemp: 30, light: 'Medium', final_status: 'Moderate Stress', confidence: .82 },
  { id: 3, time: '07:40', cropType: 'Corn', temperature: 39, humidity: 24, soilMoisture: 17, soilTemp: 35, light: 'Low', final_status: 'High Stress', confidence: .88 }
];
const initialSectors = [
  { id: 1, name: 'Greenhouse A', location: 'North Field', crop: 'Tomato', status: 'Healthy',
    sensors: { temperature: 25, humidity: 60, soilMoisture: 45, soilTemp: 24, light: 'Sufficient' },
    diagnosis: { condition: 'Healthy', disease: 'No Clear Disease Detected', confidence: .91, summary: 'Stable plant condition with balanced readings.' }, workers:[{id:1,name:'Ahmed Ali',role:'Farm Manager',phone:'01000000001'}], tasks:[{id:1,name:'Inspect leaves',priority:'Medium'},{id:2,name:'Check irrigation system',priority:'High'}], notes:['Sector is stable. Continue weekly leaf inspection.'], rows:['Healthy','Healthy','Moderate Stress','Healthy'], equipment:['Irrigation Pump','Monitoring Camera','Water Valve'], devices:[{id:1,name:'Soil Moisture Sensor',type:'Soil Moisture',value:45,unit:'%',status:'Online'},{id:2,name:'Temperature Sensor',type:'Temperature',value:25,unit:'°C',status:'Online'}] },
  { id: 2, name: 'Mint Zone', location: 'East Side', crop: 'Mint', status: 'Moderate Stress',
    sensors: { temperature: 33, humidity: 42, soilMoisture: 28, soilTemp: 29, light: 'Medium' },
    diagnosis: { condition: 'Moderate Stress', disease: 'General Visual Stress', confidence: .82, summary: 'Moisture is below the preferred range and requires monitoring.' }, workers:[{id:2,name:'Sara Mohamed',role:'Irrigation Worker',phone:'01000000002'}], tasks:[{id:3,name:'Adjust light and ventilation',priority:'Medium'},{id:4,name:'Inspect leaves',priority:'Medium'}], notes:['Mint sector needs moisture follow-up.'], rows:['Healthy','Moderate Stress','Moderate Stress','Healthy'], equipment:['Irrigation Pump','Controller','Water Valve'], devices:[{id:3,name:'Soil Moisture Sensor',type:'Soil Moisture',value:28,unit:'%',status:'Online'},{id:4,name:'Light Sensor',type:'Light',value:'Medium',unit:'',status:'Online'}] },
  { id: 3, name: 'Pepper Sector', location: 'South Greenhouse', crop: 'Pepper', status: 'High Stress',
    sensors: { temperature: 39, humidity: 24, soilMoisture: 17, soilTemp: 35, light: 'Low' },
    diagnosis: { condition: 'High Stress', disease: 'Leaf Spot / Fungal Suspicion', confidence: .88, summary: 'Critical heat and low soil moisture indicate urgent plant risk.' }, workers:[{id:3,name:'Mahmoud Hassan',role:'Plant Care Worker',phone:'01000000003'}], tasks:[{id:5,name:'Inspect leaves',priority:'High'},{id:6,name:'Clean plant area',priority:'Medium'}], notes:['High risk area. Inspect disease symptoms today.'], rows:['High Stress','Moderate Stress','High Stress','Healthy'], equipment:['Monitoring Camera','Irrigation Pump','Controller'], devices:[{id:5,name:'Temperature Sensor',type:'Temperature',value:39,unit:'°C',status:'Online'},{id:6,name:'Soil Moisture Sensor',type:'Soil Moisture',value:17,unit:'%',status:'Online'}] }
];

const initialPlants = [
  { id: 1, name: 'Tomato Plant 1', crop: 'Tomato', sectorId: 1, age: '45 days', plantedAt: '2026-01-01', lastDiagnosis: 'Healthy', disease: 'No Clear Disease Detected', image: '', timeline: [
    { id: 1, time: '10:15', status: 'Healthy', disease: 'No Clear Disease Detected', summary: 'Stable plant condition with balanced readings.' }
  ]},
  { id: 2, name: 'Pepper Bed A', crop: 'Pepper', sectorId: 3, age: '38 days', plantedAt: '2026-01-08', lastDiagnosis: 'High Stress', disease: 'Leaf Spot / Fungal Suspicion', image: '', timeline: [
    { id: 2, time: '09:40', status: 'High Stress', disease: 'Leaf Spot / Fungal Suspicion', summary: 'Critical heat and low soil moisture indicate urgent plant risk.' }
  ]}
];


const initialTasks = [
  { id: 1, title: 'Inspect leaves', status: 'pending', priority: 'High', assignedTo: 'Mahmoud Hassan', assignedToEmail: 'mahmoud.worker@ecosense.local', createdAt: '2026-01-25 09:00', dueDate: '2026-01-25', sector: 'Pepper Sector', details: 'Inspect leaves for dark spots and upload proof if needed.' },
  { id: 2, title: 'Check irrigation system', status: 'inProgress', priority: 'Medium', assignedTo: 'Sara Mohamed', assignedToEmail: 'sara.worker@ecosense.local', createdAt: '2026-01-26 08:30', dueDate: '2026-01-26', sector: 'Mint Zone', details: 'Check pump, valve, and soil moisture trend.' },
  { id: 3, title: 'Clean plant area', status: 'completed', priority: 'Low', assignedTo: 'Ahmed Ali', assignedToEmail: 'ahmed.manager@ecosense.local', createdAt: '2026-01-23 10:15', completedAt: '2026-01-23 12:40', dueDate: '2026-01-23', sector: 'Greenhouse A', details: 'Clean plant bed and remove dead leaves.' }
];

const getStore = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) || fallback; } catch { return fallback; } };
const setStore = (key, value) => localStorage.setItem(key, JSON.stringify(value));
const currentUser = () => getStore('ecosense_user', { email: 'demo@smartplant.local', name: 'Demo Owner', role: 'owner' });
const currentUserEmail = () => (currentUser()?.email || localStorage.getItem('sph_user_email') || 'demo@smartplant.local').toLowerCase();
const currentUserName = () => currentUser()?.name || currentUser()?.fullName || currentUserEmail();
const scopedKey = (key, email=currentUserEmail()) => `${key}_${String(email || 'demo@smartplant.local').toLowerCase()}`;
const getUserStore = (key, fallback, email=currentUserEmail()) => getStore(scopedKey(key, email), fallback);
const setUserStore = (key, value, email=currentUserEmail()) => setStore(scopedKey(key, email), value);
const normalizeTaskStatus = (status='pending') => status === 'todo' ? 'pending' : status === 'done' ? 'completed' : status;
const getGlobalTasks = () => getStore('sph_tasks_global', initialTasks).map(t => ({...t, status: normalizeTaskStatus(t.status)}));
const setGlobalTasks = (tasks) => setStore('sph_tasks_global', tasks.map(t => ({...t, status: normalizeTaskStatus(t.status)})));
const workerAccounts = () => getStore('sph_worker_accounts', []);
const findLocalAccount = (identifier='', passwordInput='') => {
  const id = String(identifier || '').trim().toLowerCase();
  if(!id) return null;
  return workerAccounts().find(acc => {
    const username = String(acc.username || acc.email || '').toLowerCase();
    const email = String(acc.email || '').toLowerCase();
    return (username === id || email === id) && String(acc.password || '') === String(passwordInput || '');
  }) || null;
};
const setLoggedInUser = (user, token='local-demo-token') => {
  const role = normalizeRole(user.role || user.jobTitle, user.email || user.username);
  const account = {...user, role};
  localStorage.setItem('ecosense_user', JSON.stringify(account));
  localStorage.setItem('sph_auth','true');
  localStorage.setItem('sph_user_email', (account.email || account.username || 'demo@smartplant.local').toLowerCase());
  localStorage.setItem('sph_token', token || 'local-demo-token');
  localStorage.setItem('sph_role', role);
};
const taskAssignedToCurrent = (task) => {
  const email = currentUserEmail();
  const name = String(currentUserName()).toLowerCase();
  return String(task.assignedToEmail || '').toLowerCase() === email || String(task.assignedTo || '').toLowerCase() === name;
};
const latestAutoReading = () => {
  const readings = getStore('sph_readings', initialReadings);
  return readings[0] || initialReadings[0];
};
const notificationStore = (email=currentUserEmail()) => getUserStore('sph_notifications', [], email);
const setNotificationStore = (items, email=currentUserEmail()) => setUserStore('sph_notifications', items, email);
const playNotificationSound = (type='warning') => {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    const gain = ctx.createGain();
    gain.connect(ctx.destination);
    const pattern = type === 'danger' ? [980, 740, 980, 520, 980] : [760, 620, 760];
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(type === 'danger' ? 0.38 : 0.22, ctx.currentTime + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 1.25);
    pattern.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      osc.type = i % 2 ? 'square' : 'sawtooth';
      osc.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.18);
      osc.connect(gain);
      osc.start(ctx.currentTime + i * 0.18);
      osc.stop(ctx.currentTime + i * 0.18 + 0.14);
    });
    if (navigator.vibrate && type === 'danger') navigator.vibrate([250,120,250,120,350]);
  } catch {}
};
const addUserNotification = (email, message, type='warning') => {
  const target = String(email || currentUserEmail()).toLowerCase();
  const all = [{ id: Date.now(), time: new Date().toLocaleString(), message, type, read:false }, ...notificationStore(target)].slice(0, 40);
  setNotificationStore(all, target);
  if (target === currentUserEmail()) {
    if (getStore('sph_settings', {soundAlerts:true}).soundAlerts !== false) playNotificationSound(type);
    try { window.dispatchEvent(new CustomEvent('sph-notification', { detail: { message, type } })); } catch {}
  }
  return all;
};
const addNotification = (message, type='warning') => addUserNotification(currentUserEmail(), message, type);
const latestHomeAlert = (t) => {
  const n = notificationStore()[0];
  return n ? n.message : t('noCriticalAlerts');
};
const isDanger = (status='') => String(status).includes('High') || String(status).toLowerCase().includes('critical') || String(status).includes('خطر');

const currentRole = () => localStorage.getItem('sph_role') || currentUser()?.role || 'owner';
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
  ['sph_auth','ecosense_token','sph_role'].forEach(k => localStorage.removeItem(k));
  window.location.href = '/login';
};

const rolePermissions = {
  owner: { diagnose:true, reports:true, exportReports:true, sensors:true, devices:true, alerts:true, tasks:true, assignTasks:true, workers:true, sectors:true, editOperationalSector:true, deleteSector:true, deleteDevice:true, managePermissions:true, ownerSensitive:true, settingsFarm:true },
  farm_manager: { diagnose:true, reports:true, exportReports:true, sensors:true, devices:true, alerts:true, tasks:true, assignTasks:true, workersView:true, sectors:true, editOperationalSector:true, deleteSector:false, deleteDevice:false, managePermissions:false, ownerSensitive:false, settingsFarm:false },
  worker: { diagnose:false, reports:false, exportReports:false, sensors:false, devices:false, alerts:true, tasks:true, assignTasks:false, workers:false, sectors:true, editOperationalSector:false, deleteSector:false, deleteDevice:false, managePermissions:false, ownerSensitive:false, settingsFarm:false }
};
const hasPermission = (permission) => !!(rolePermissions[currentRole()] || rolePermissions.worker)[permission];
const normalizeRole = (role, email='') => {
  const r = String(role || '').toLowerCase().trim();
  if (['farm_manager','farm manager','manager','مدير مزرعة','farm-manager'].includes(r)) return 'farm_manager';
  if (['worker','عامل','employee','staff','plant care worker','irrigation worker'].includes(r)) return 'worker';
  if (['owner','مالك','admin','administrator'].includes(r)) return 'owner';
  const e = String(email || '').toLowerCase();
  if(e.includes('manager')) return 'farm_manager';
  if(e.includes('worker')) return 'worker';
  return 'owner';
};

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

  return <LangContext.Provider value={value}>
    <Routes>
      <Route path="/" element={<Landing theme={theme} setTheme={setTheme} />} />
      <Route path="/login" element={<Auth mode="login" theme={theme} setTheme={setTheme} />} />
      <Route path="/register" element={<Auth mode="register" theme={theme} setTheme={setTheme} />} />
      <Route element={<ProtectedRoute><AppShell theme={theme} setTheme={setTheme} /></ProtectedRoute>}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/diagnosis" element={<Diagnosis />} />
        <Route path="/sensors" element={<Navigate to="/farm-management" replace />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/reports" element={<Navigate to="/farm-management" replace />} />
        <Route path="/library" element={<Navigate to="/diagnosis" replace />} />
        <Route path="/farm-management" element={<FarmManagement />} />
        <Route path="/sectors" element={<Navigate to="/farm-management" replace />} />
        <Route path="/sectors/:id" element={<SectorDetails />} />
        <Route path="/workers" element={<RoleGate any={["workers","workersView"]}><WorkerAccountsPage /></RoleGate>} />
        <Route path="/devices" element={<Navigate to="/farm-management" replace />} />
        <Route path="/plants" element={<OwnerOnly><PlantProfiles /></OwnerOnly>} />
        <Route path="/tasks" element={<MyTasks />} />
        <Route path="/admin" element={<Navigate to="/farm-management" replace />} />
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

function ProtectedRoute({children}) {
  const authed = localStorage.getItem('sph_auth') === 'true';
  const loc = useLocation();
  const onboarded = localStorage.getItem('sph_onboarded') === 'true';
  if (!authed) return <Navigate to="/login" replace />;
  if (!onboarded && loc.pathname !== '/onboarding') return <Navigate to="/onboarding" replace />;
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
    const loadBackendNotifications = async()=>{
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
        /* backend notifications unavailable: local notifications stay active */
      }
    };
    sync(); loadBackendNotifications();
    window.addEventListener('sph-notification', sync);
    window.addEventListener('storage', sync);
    const timer = setInterval(loadBackendNotifications, 30000);
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
    return ()=>document.body.classList.remove('sidebar-open');
  },[open]);
  const page = [
    ['/diagnosis','diagnosisCenter'], ['/dashboard','dashboard'], ['/plants','myDiagnosis'], ['/alerts','alerts'], ['/farm-management','farmManagementTitle'], ['/workers','workers'], ['/sectors','farmManagementTitle'], ['/more','more'], ['/settings','settings']
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
    <MobileBottomNav />
    <FloatingAssistant/><NotificationToast/>
  </div>
}

function Sidebar({ open, close }) {
  const { t } = useLang();
  const ownerItems = [ ['/dashboard', Home, 'dashboard'], ['/diagnosis', ScanSearch, 'diagnosisCenter'], ['/plants', Sprout, 'myDiagnosis'], ['/farm-management', Layers3, 'farmManagementTitle'], ['/workers', UserPlus, 'workers'], ['/settings', ShieldCheck, 'settings'] ];
  const managerItems = [ ['/dashboard', Home, 'dashboard'], ['/diagnosis', ScanSearch, 'diagnosisCenter'], ['/farm-management', Layers3, 'farmManagementTitle'], ['/tasks', CheckCircle2, 'myTasks'], ['/alerts', Bell, 'alerts'], ['/workers', UserPlus, 'workers'], ['/settings', ShieldCheck, 'settings'] ];
  const workerItems = [ ['/dashboard', Home, 'dashboard'], ['/tasks', CheckCircle2, 'myTasks'], ['/farm-management', Layers3, 'assignedSector'], ['/alerts', Bell, 'alerts'], ['/settings', ShieldCheck, 'settings'] ];
  const items = isOwner() ? ownerItems : isFarmManager() ? managerItems : workerItems;
  return <aside className={`sidebar ${open?'open':''}`}>
    <button className="side-close" onClick={close}><X/></button>
    <LogoBlock/>
    <nav>{items.map(([to,Icon,key])=><NavLink key={to} to={to} onClick={close} className={({isActive})=>`side-link ${isActive?'active':''}`}><Icon size={20}/><span>{t(key)}</span></NavLink>)}</nav>
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
  const baseItems = [
    ['/dashboard', Home, 'dashboard'],
    ['/diagnosis', ScanSearch, 'diagnosis'],
    ['/farm-management', Layers3, 'farm'],
    ['/alerts', Bell, 'alerts'],
    ['/more', Menu, 'more']
  ];
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
  const readings=getStore('sph_readings',initialReadings); 
  const latest=latestAutoReading();
  const alerts=notificationStore().length;
  const risky=readings.filter(r=>r.final_status!=='Healthy').length;
  const latestDiagnosis = latest.diagnosis || (latest.final_status==='Healthy' ? 'Stable plant condition with balanced readings.' : 'Water or heat stress indicators detected.');
  return <>
    <PageHead title={currentRole()==='owner'?t('ownerDashboard'):currentRole()==='farm_manager'?t('farmManagerDashboard'):t('workerDashboard')} sub={t('mainProductFocus')} action={<NavLink to="/diagnosis" className="primary-btn"><ScanSearch size={18}/>{t('startDiagnosis')}</NavLink>}/>
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
      <Kpi icon={<Activity/>} label={t('soilMoisture')} value={`${latest.soilMoisture}%`}/>
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
function ReadingGrid({r}){ const {t}=useLang(); return <div className="reading-grid"><Box label={t('cropType')} value={r.cropType}/><Box label={t('temperature')} value={`${r.temperature}°C`}/><Box label={t('humidity')} value={`${r.humidity}%`}/><Box label={t('soilMoisture')} value={`${r.soilMoisture}%`}/><Box label={t('soilTemp')} value={`${r.soilTemp}°C`}/><Box label={t('light')} value={r.light}/></div> }
function Box({label,value}){ const {t}=useLang(); return <div className="box"><span>{label}</span><b>{localizeValue(value,t)}</b></div> }

function normalizeBackendSensorPayload(payload={}){
  const data = payload.data || payload.readings || payload.latest || payload.sensor_readings || payload;
  const imageUrl = payload.image_url || payload.imageUrl || payload.latest_image || payload.latestImage || payload.snapshot_url || payload.snapshotUrl || data.image_url || data.imageUrl || data.snapshot_url;
  const now = new Date().toLocaleString();
  const reading = {
    id: Date.now(),
    time: data.time || data.timestamp || data.created_at || new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}),
    lastUpdated: data.lastUpdated || data.last_updated || data.timestamp || now,
    cropType: data.cropType || data.crop_type || data.crop || 'Tomato',
    temperature: Number(data.temperature ?? data.airTemperature ?? data.temp ?? 25),
    humidity: Number(data.humidity ?? 60),
    soilMoisture: Number(data.soilMoisture ?? data.soil_moisture ?? data.moisture ?? 45),
    soilTemp: Number(data.soilTemp ?? data.soil_temp ?? data.soilTemperature ?? 24),
    light: data.light || data.lightLevel || data.light_level || 'Sufficient',
    sector: data.sector || data.sectorName || data.sector_name || 'Greenhouse A',
    final_status: data.final_status || data.finalStatus || 'Healthy',
    confidence: Number(data.confidence ?? .91),
    source: payload.source || data.source || 'backend'
  };
  return { reading, imageUrl };
}

function makeSimulationReading(){
  const samples=[
    { cropType:'Tomato', temperature:25, humidity:60, soilMoisture:45, soilTemp:24, light:'Sufficient', final_status:'Healthy', confidence:.91 },
    { cropType:'Pepper', temperature:34, humidity:40, soilMoisture:28, soilTemp:30, light:'Medium', final_status:'Moderate Stress', confidence:.82 },
    { cropType:'Corn', temperature:39, humidity:24, soilMoisture:17, soilTemp:35, light:'Low', final_status:'High Stress', confidence:.88 },
    { cropType:'Mint', temperature:29, humidity:55, soilMoisture:36, soilTemp:27, light:'Sufficient', final_status:'Moderate Stress', confidence:.79 }
  ];
  const reading = samples[Math.floor(Math.random()*samples.length)];
  const now = new Date();
  return { id: Date.now(), time: now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}), lastUpdated: now.toLocaleString(), sector:'Greenhouse A', source:'simulation', ...reading };
}

async function fileFromImageUrl(url){
  const response = await fetch(url, { mode:'cors' });
  if(!response.ok) throw new Error('image-fetch-failed');
  const blob = await response.blob();
  return new File([blob], 'backend-captured-plant-image.jpg', { type: blob.type || 'image/jpeg' });
}

function Diagnosis(){ 
  const {t}=useLang(); 
  const cameraInputRef=useRef(null);
  const galleryInputRef=useRef(null);
  const [diagnosisMode,setDiagnosisMode]=useState('combined'); 
  const [sensorData,setSensorData]=useState(latestAutoReading()); 
  const [file,setFile]=useState(null); 
  const [preview,setPreview]=useState(''); 
  const [backendImageUrl,setBackendImageUrl]=useState('');
  const [result,setResult]=useState(null); 
  const [loading,setLoading]=useState(false); 
  const [mode,setMode]=useState('apiMode'); 
  const [sensorState,setSensorState]=useState({status:'loading', source:'backend', lastUpdated:'', error:''});

  const applyReading=(reading, source='backend')=>{
    const clean={...reading, lastUpdated: reading.lastUpdated || new Date().toLocaleString(), source};
    setSensorData(clean);
    const all=[clean,...getStore('sph_readings',initialReadings)].slice(0,30);
    setStore('sph_readings', all);
    try { window.dispatchEvent(new CustomEvent('sph-sensors-updated', { detail: clean })); } catch {}
  };

  const loadBackendSensors=async({silent=false, fallback=false}={})=>{
    if(!silent) setSensorState(s=>({...s,status:'loading', error:''}));
    try{
      const response=await sensorsAPI.getLatest(sensorData.sectorId || sensorData.sector || '');
      const {reading,imageUrl}=normalizeBackendSensorPayload(response.data || {});
      applyReading(reading,'backend');
      if(imageUrl){ setBackendImageUrl(imageUrl); setPreview(imageUrl); }
      setSensorState({status:'connected', source:'backend', lastUpdated: reading.lastUpdated, error:''});
      if(!silent) addNotification(t('sensorsUpdated'),'success');
    }catch(err){
      if(fallback){
        const reading=makeSimulationReading();
        applyReading(reading,'simulation');
        setSensorState({status:'simulation', source:'simulation', lastUpdated: reading.lastUpdated, error:String(err?.message||'offline')});
        if(!silent) addNotification(t('sensorsSimulationUpdated'),'warning');
      }else{
        setSensorState(s=>({...s,status:'offline', source:'backend', error:String(err?.message||'offline')}));
        if(!silent) addNotification(t('sensorsOfflineFallback'),'warning');
      }
    }
  };

  useEffect(()=>{
    const reading = makeSimulationReading();
    applyReading(reading,'simulation');
    setSensorState({status:'simulation', source:'simulation', lastUpdated:reading.lastUpdated, error:'Backend sensor endpoint is not connected yet'});
  },[]);

  const choose=e=>{ 
    const f=e.target.files?.[0]; 
    if(!f) return;
    setFile(f); setBackendImageUrl('');
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
  const clearImage=()=>{ setFile(null); setPreview(''); setBackendImageUrl(''); addNotification(t('imageRemoved'),'success'); };
  const change=(k,v)=>setSensorData({...sensorData,[k]:v,lastUpdated:new Date().toLocaleString(),source:'manual'});
  const refresh=()=>{ const next=makeSimulationReading(); applyReading(next,'simulation'); setSensorState({status:'simulation',source:'simulation',lastUpdated:next.lastUpdated,error:t('sensorsApiNotConnected')}); addNotification(t('sensorsSimulationUpdated'),'success'); };
  const simulate=()=>{ const next=makeSimulationReading(); applyReading(next,'simulation'); setSensorState({status:'simulation',source:'simulation',lastUpdated:next.lastUpdated,error:''}); addNotification(t('sensorsSimulationUpdated'),'success'); };

  async function analyze(){ 
    setLoading(true); setResult(null); 
    const form = {
      cropType: sensorData.cropType, temperature: sensorData.temperature, humidity: sensorData.humidity,
      soilMoisture: sensorData.soilMoisture, soilTemp: sensorData.soilTemp, light: sensorData.light
    };
    let res, apiOk=false; 
    try{ 
      if (diagnosisMode === 'sensors') {
        res = await aiDiagnosisAPI.predictSensors(form);
      } else {
        let imageFile=file;
        if(!imageFile && backendImageUrl){
          imageFile = await fileFromImageUrl(backendImageUrl);
        }
        if (!imageFile) {
          addNotification(t('imageRequired'),'danger');
          throw new Error('Image file is required for image diagnosis modes');
        }

        const fd = new FormData();
        fd.append('file', imageFile);
        fd.append('cropType', form.cropType);

        if (diagnosisMode === 'combined') {
          fd.append('temperature', String(form.temperature));
          fd.append('humidity', String(form.humidity));
          fd.append('soilMoisture', String(form.soilMoisture));
          fd.append('soilTemp', String(form.soilTemp));
          fd.append('light', form.light);
        }

        res = diagnosisMode === 'image' ? await aiDiagnosisAPI.predictImage(fd) : await aiDiagnosisAPI.predictWithImage(fd);
      }

      if(!res.ok) throw new Error('API');
      const data=await res.json(); 
      const normalized=normalize(data, form, diagnosisMode !== 'sensors' && (!!file || !!backendImageUrl));
      normalized.diagnosis_mode=data.source || diagnosisMode; normalized.image_preview=preview;
      setResult(normalized); apiOk=true; 
      persistDiagnosis(normalized, form, t);
      triggerModelAlerts(normalized,t);
      addNotification(t('diagnosisSuccess'),'success');
    } catch (error) { 
      const demo=demoResult(form,diagnosisMode !== 'sensors' && (!!file || !!backendImageUrl));
      demo.diagnosis_mode=diagnosisMode; demo.image_preview=preview;
      demo.suspicious_regions = demo.suspicious_regions || (preview && isDanger(demo.final_status) ? [{x:54,y:25,w:26,h:22,label:'Leaf spots'}] : []);
      setResult(demo); persistDiagnosis(demo, form, t); triggerModelAlerts(demo,t);
      addNotification(t('diagnosisFallback'),'warning');
    } finally { setMode(apiOk?'apiMode':'demoMode'); setLoading(false); }
  } 

  const sensorTone = sensorState.status==='connected' ? 'success' : sensorState.status==='simulation' ? 'warning' : sensorState.status==='loading' ? 'info' : 'danger';
  const sensorLabel = sensorState.status==='connected' ? t('sensorConnected') : sensorState.status==='simulation' ? t('simulationMode') : sensorState.status==='loading' ? t('sensorLoading') : t('sensorOfflineState');
  const imageRequired = (diagnosisMode === 'image' || diagnosisMode === 'combined') && !file && !backendImageUrl;
  return <>
    <PageHead title={t('diagnosisCenter')} sub={t('modelDrivenNote')} action={<span className="mode-pill">{t(mode)}</span>}/>
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
          <div><b>{sensorLabel}</b><span>{t('lastUpdated')}: {sensorState.lastUpdated || sensorData.lastUpdated || '-'}</span><span>{t('sensorSource')}: {sensorState.status==='connected'?t('liveSource'):sensorState.status==='simulation'?t('simulationSource'):t('backendRequiredSource')}</span></div>
        </div>
        <p className="muted-copy">{sensorState.status==='simulation' ? t('sensorsOfflineFallback') : t('sensorUserHint')}</p>
        <div className="form-grid">
          <Field label={t('cropType')}><select value={sensorData.cropType} onChange={e=>change('cropType',e.target.value)}><option value="Tomato">{t('tomato')}</option><option value="Corn">{t('corn')}</option><option value="Pepper">{t('pepper')}</option><option value="Mint">{t('mint')}</option></select></Field>
          <Field label={t('temperature')}><input type="number" value={sensorData.temperature} onChange={e=>change('temperature',Number(e.target.value))}/></Field>
          <Field label={t('humidity')}><input type="number" value={sensorData.humidity} onChange={e=>change('humidity',Number(e.target.value))}/></Field>
          <Field label={t('soilMoisture')}><input type="number" value={sensorData.soilMoisture} onChange={e=>change('soilMoisture',Number(e.target.value))}/></Field>
          <Field label={t('soilTemp')}><input type="number" value={sensorData.soilTemp} onChange={e=>change('soilTemp',Number(e.target.value))}/></Field>
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
        <div className={`dropzone image-scan-zone ${preview?'has-preview':''}`}>{preview?<><img src={preview} alt={t('imagePreview')}/><span className="image-source-pill">{backendImageUrl ? t('latestBackendImage') : t('imagePreview')}</span><span className="image-ready-pill"><CheckCircle2 size={14}/>{t('imageReady')}</span></>:<div><UploadCloud size={48}/><b>{t('chooseImage')}</b><small>{t('mobileImagePickerNote')}</small></div>}</div>
        {preview && <div className="mobile-action-row"><button type="button" className="secondary-btn" onClick={openGallery}>{t('changeImage')}</button><button type="button" className="danger-btn" onClick={clearImage}><X size={16}/>{t('removeImage')}</button></div>}
        {backendImageUrl && <p className="muted-copy"><CheckCircle2 size={14}/> {t('combinedReady')}</p>}
      </Panel>}
    </div>
    <button className="primary-btn wide" onClick={analyze} disabled={loading || imageRequired}>{loading?<><Activity className="spin" size={18}/>{t('analyzing')}</>:<><ScanSearch size={18}/>{t('sendToModel')}</>}</button>
    {imageRequired && <p className="form-helper warning-text">{t('imageRequired')}</p>}
    {result && <Result result={result}/>}
  </>
}
function Field({label,children}){ return <label className="field"><span>{label}</span>{children}</label> }
function normalize(r, inputData={}, hasImage=false){
  const ia = r.image_analysis || r.imageAnalysis || r.analysis || {};
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
    .86;

  const diagnosisText = diagnosisObj
    ? (diagnosisObj.explanation || diagnosisObj.primary_issue || diagnosisObj.visual_problem_ar || diagnosisObj.visual_problem || '-')
    : (r.diagnosis || r.summary || r.message || '-');

  return {
    final_status: r.final_status || r.finalStatus || r.status || ia.final_status || 'Healthy',
    sensor_status: r.sensor_status || r.sensorStatus || '-',
    image_status: r.image_status || r.imageStatus || ia.image_stress || ia.final_status || (hasImage ? 'Analyzed' : 'No Image'),
    disease_name: disease || 'No Clear Disease Detected',
    confidence: Number(confidenceValue || 0),
    final_confidence: r.final_confidence ?? Number(confidenceValue || 0),
    sensor_confidence: r.sensor_confidence || r.confidence || null,
    image_confidence: r.image_confidence ?? ia.confidence ?? null,
    severity: r.severity || '-',
    diagnosis: diagnosisText,
    diagnosis_details: diagnosisObj || null,
    recommendations: Array.isArray(r.recommendations) ? r.recommendations : (Array.isArray(r.recommendations_ar) ? r.recommendations_ar : [r.recommendations || r.recommendations_ar || 'Keep monitoring sensor readings.']),
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
function demoResult(f, hasImage){
  let s='Healthy', sev='low', diag='Readings are stable. No urgent plant-health risk detected.', disease='No Clear Disease Detected';
  let flags=[], risks=[];
  if(f.soilMoisture<30||f.temperature>33){s='Moderate Stress';sev='medium';diag='The plant may be under water or heat stress. Monitoring and corrective care are recommended.'; risks.push('Low soil moisture or high temperature'); disease=hasImage?'General Visual Stress':'No Clear Disease Detected'}
  if(f.soilMoisture<=18||f.temperature>=38||f.light==='Low'){s='High Stress';sev='high';diag='The plant is in a dangerous condition. Low moisture, high heat, or weak light may cause fast deterioration.'; flags.push('Urgent plant risk'); risks.push('Critical low soil moisture','Low light','Heat stress'); disease=hasImage?'Leaf Spot / Fungal Suspicion':'Severe Environmental Stress'}
  return {final_status:s,sensor_status:s,image_status:hasImage?s:'No Image',disease_name:disease,confidence:s==='Healthy'?.91:.84,severity:sev,diagnosis:diag,recommendations:s==='Healthy'?['Keep monitoring sensor readings.','Maintain stable watering and light exposure.','Recheck leaves weekly.']:['Adjust irrigation according to soil moisture.','Improve airflow and reduce heat exposure.','Inspect leaves for spots, yellowing, or brown tissue.'],actions:s==='Healthy'?['Continue normal care.']:['Water gradually if soil is dry.','Move plant away from extreme heat.','Isolate visibly damaged plants if disease symptoms spread.'],backend_flags:flags,risk_factors:risks,input_data:f,image_analysis:hasImage?{green_ratio:.68,yellow_ratio:.14,brown_ratio:.08,dark_spot_ratio:.05,damaged_ratio:.27,health_score:.72,visual_problem:disease}:{}};
}
function persistDiagnosis(result, form, t){
  const readings=getStore('sph_readings', initialReadings);
  const r={ id: Date.now(), time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}), ...form, final_status:result.final_status, confidence:result.confidence, disease:result.disease_name };
  setStore('sph_readings',[r,...readings].slice(0,30));
  const sectors=getStore('sph_sectors', initialSectors);
  const idx=sectors.findIndex(s=>s.crop===form.cropType);
  if(idx>=0){
    sectors[idx]={...sectors[idx], status:result.final_status, sensors:form, diagnosis:{condition:result.final_status,disease:result.disease_name,confidence:result.confidence,summary:result.diagnosis}};
    setStore('sph_sectors', sectors);
  }
  if(isDanger(result.final_status)){
    addNotification(`${t('highRiskWarning')} (${form.cropType})`, 'danger');
  }
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
  const {t}=useLang();
  const ia = result.image_analysis || {};
  const input = result.input_data || {};
  const sectors = getStore('sph_sectors', initialSectors);
  const [sectorId,setSectorId]=useState(sectors[0]?.id || '');
  const linkToSector=()=>{
    const all=getStore('sph_sectors', initialSectors);
    const updated=all.map(s=>Number(s.id)===Number(sectorId)?{...s,status:result.final_status,diagnosis:{condition:result.final_status,disease:result.disease_name,confidence:result.confidence,summary:result.diagnosis},sensors:input}:s);
    setStore('sph_sectors',updated); alert(t('linkedToSector'));
  };
  const firstAction = result.actions?.[0];
  const secondAction = result.actions?.[1];
  const plan = result.treatment_plan || {
    now: firstAction
      ? `${getActionTitle(firstAction, t)}${getActionDetails(firstAction, t) ? ' - ' + getActionDetails(firstAction, t) : ''}`
      : result.recommendations?.[0],
    next24h: secondAction
      ? `${getActionTitle(secondAction, t)}${getActionDetails(secondAction, t) ? ' - ' + getActionDetails(secondAction, t) : ''}`
      : result.recommendations?.[1] || result.recommendations?.[0],
    weekly: result.monitoring?.[0] || result.recommendations?.[2] || t('taskInspectLeaves')
  };
  const modelAlerts = [
    ...(Array.isArray(result.model_alerts) ? result.model_alerts : []),
    ...(isDanger(result.final_status) ? [t('alarmAlert')] : [])
  ].filter(Boolean);

  return <Panel title={t('userFriendlyResult')}>
    <div className={`model-alert-strip ${tone(result.final_status)}`}>
      <Bell size={20}/>
      <div>
        <b>{t('modelNotificationBar')}</b>
        <span>{modelAlerts.length ? modelAlerts.map(a=>displayText(a,t)).join(' • ') : t('noModelAlerts')}</span>
      </div>
    </div>

    <div className={`diagnosis-hero compact ${tone(result.final_status)}`}>
      <div>
        <span>{t('condition')}</span>
        <h2>{labelStatus(result.final_status,t)}</h2>
        <p>{t('compactDiagnosisView')}</p>
      </div>
      <div className="disease-card">
        <span>{t('diseaseName')}</span>
        <b>{translateModelText(result.disease_name || t('noDiseaseDetected'), t)}</b>
      </div>
    </div>

    <div className="result-main rich compact-metrics">
      <Box label={t('confidence')} value={`${Math.round(result.confidence*100)}%`}/>
      <Box label={t('severity')} value={String(result.severity).toUpperCase()}/>
      <Box label={t('diagnosisMode')} value={result.diagnosis_mode||'combined'}/>
      <Box label={t('healthScore')} value={`${Math.round((ia.health_score ?? result.confidence ?? .8)*100)}%`}/>
    </div>

    {result.image_preview && <div className="image-result-wrap compact-image-result">
      <h4>{t('uploadedImageData')}</h4>
      <div className="suspicion-image">
        <img src={result.image_preview}/>
        {(result.suspicious_regions||[]).map((r,i)=><span key={i} className="suspect-box" style={{left:`${r.x||r.left||20}%`,top:`${r.y||r.top||20}%`,width:`${r.w||r.width||25}%`,height:`${r.h||r.height||20}%`}}>{r.label||t('possibleDisease')}</span>)}
      </div>
    </div>}

    <div className="result-sections treatment clean-grid">
      <section className="detail-card wide-card"><h4>{t('diagnosisSummary')}</h4><p>{displayText(result.diagnosis, t)}</p></section>
      <section className="detail-card"><h4>{t('modelLikelyCause')}</h4><p>{displayText(result.likely_cause || getLikelyCause(result,t), t)}</p></section>
      <section className="detail-card"><h4>{t('treatmentPlan')}</h4><div className="treatment-steps"><div><b>{t('nowAction')}</b><span>{displayText(plan.now,t)}</span></div><div><b>{t('next24h')}</b><span>{displayText(plan.next24h,t)}</span></div><div><b>{t('weeklyFollowup')}</b><span>{displayText(plan.weekly,t)}</span></div></div></section>
      <section className="detail-card"><h4>{t('recommendedCare')}</h4><ul className="rec-list pro">{(result.recommendations||[]).slice(0,4).map((r,i)=><li key={i}>{displayText(r,t)}</li>)}</ul></section>
    </div>

    <div className="result-actions compact-actions">
      <button className="primary-btn" onClick={()=>saveDiagnosisResult(result,t)}><CheckCircle2 size={18}/>{t('saveDiagnosis')}</button>
      <button className="secondary-btn" onClick={()=>downloadDiagnosisReport(result,t)}><FileText size={18}/>{t('downloadPdf')}</button>
      <select value={sectorId} onChange={e=>setSectorId(e.target.value)}>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}</option>)}</select>
      <button className="secondary-btn" onClick={linkToSector}><Layers3 size={18}/>{t('linkDiagnosisToSector')}</button>
    </div>

    <div className="grid-2 result-extra compact-extra">
      <section className="detail-card"><h4>{t('sentReadings')}</h4><div className="data-grid compact-data">{['cropType','temperature','humidity','soilMoisture','soilTemp','light'].map(k=><Box key={k} label={t(k)} value={formatInputValue(k,input[k])}/>) }</div></section>
      <section className="detail-card"><h4>{t('imageIndicators')}</h4>{Object.keys(ia||{}).length>0 ? <Bars data={ia}/> : <p className="muted-small">{t('noImage')}</p>}</section>
    </div>
  </Panel>
}
function getLikelyCause(result,t){
  if(isDanger(result.final_status)) return t('riskReasonCritical');
  if(String(result.final_status).includes('Moderate')) return t('riskReasonWaterHeat');
  return t('riskReasonStable');
}
function saveDiagnosisResult(result,t){
  const plants=getUserStore('sph_plants', initialPlants);
  const first=plants[0] || initialPlants[0];
  const updated={...first,lastDiagnosis:result.final_status,disease:result.disease_name,timeline:[{id:Date.now(),time:new Date().toLocaleString(),status:result.final_status,disease:result.disease_name,summary:result.diagnosis},...(first.timeline||[])]};
  setUserStore('sph_plants',[updated,...plants.filter(p=>p.id!==first.id)]);
  alert(t('saveSuccess'));
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
  const html=`<html dir="rtl"><head><meta charset="UTF-8"><title>${t('plantDiagnosis')}</title></head><body><h1>${t('plantDiagnosis')}</h1><h2>${t('condition')}: ${labelStatus(result.final_status,t)}</h2><p>${t('diseaseProblem')}: ${translateModelText(result.disease_name,t)}</p><p>${t('diagnosisSummary')}: ${translateModelText(result.diagnosis,t)}</p><h3>${t('recommendedCare')}</h3><ul>${result.recommendations.map(x=>`<li>${translateModelText(x,t)}</li>`).join('')}</ul></body></html>`;
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([html],{type:'text/html'})); a.download='plant-diagnosis-report.html'; a.click();
}
function formatInputValue(k,v){ if(v==null || v==='') return '—'; if(['temperature','soilTemp'].includes(k)) return `${v}°C`; if(['humidity','soilMoisture'].includes(k)) return `${v}%`; return v; }
function Bars({data}){ const {t}=useLang(); const keys=[['green_ratio','greenRatio'],['yellow_ratio','yellowRatio'],['brown_ratio','brownRatio'],['dark_spot_ratio','darkSpotRatio'],['damaged_ratio','damagedRatio']]; return <div className="bars">{keys.filter(([k])=>data[k]!=null).map(([k,l])=>{ const v=Math.round(Number(data[k])*100); return <div key={k}><span>{t(l)}</span><em><i style={{width:`${v}%`}}/></em><b>{v}%</b></div> })}</div> }
function Sensors(){ 
  const {t}=useLang(); 
  const [readings,setReadings]=useState(getStore('sph_readings',initialReadings)); 
  const [device,setDevice]=useState({deviceId:'SPH-DEVICE-001',interval:'5'});
  const [form,setForm]=useState({cropType:'Tomato',temperature:25,humidity:60,soilMoisture:45,soilTemp:24,light:'Sufficient'});
  const [result,setResult]=useState(null);
  const [historyOpen,setHistoryOpen]=useState(false);
  const change=(k,v)=>setForm({...form,[k]:v});
  const sendSensors=async()=>{ 
    let normalized;
    try{
      const res=await aiDiagnosisAPI.predictSensors(form);
      if(!res.ok) throw new Error('API');
      normalized=normalize(await res.json(), form, false);
    }catch{
      normalized=demoResult(form,false);
    }
    setResult(normalized);
    const r={...form,id:Date.now(),time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),final_status:normalized.final_status,confidence:normalized.confidence,solution:normalized.recommendations?.[0]};
    const all=[r,...readings].slice(0,30);
    setReadings(all); setStore('sph_readings',all);
    triggerModelAlerts(normalized,t);
  };
  const simulate=()=>{
    const next={cropType:['Tomato','Corn','Pepper','Mint'][Math.floor(Math.random()*4)],temperature:Math.round(23+Math.random()*17),humidity:Math.round(25+Math.random()*45),soilMoisture:Math.round(16+Math.random()*45),soilTemp:Math.round(22+Math.random()*14),light:['Low','Medium','Sufficient'][Math.floor(Math.random()*3)]};
    setForm(next);
  };
  return <>
    <PageHead title={t('connectedSensors')} sub={t('sensorUserHint')} action={<button className="primary-btn" onClick={sendSensors}><ScanSearch size={18}/>{t('sendToModel')}</button>}/>
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
          <Field label={t('cropType')}><select value={form.cropType} onChange={e=>change('cropType',e.target.value)}><option>Tomato</option><option>Corn</option><option>Pepper</option><option>Mint</option></select></Field>
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
  const sectors=getStore('sph_sectors',initialSectors);
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

function DataTable({readings}){ const {t}=useLang(); return <div className="table-wrap"><table><thead><tr><th>{t('time')}</th><th>{t('cropType')}</th><th>{t('temperature')}</th><th>{t('humidity')}</th><th>{t('soilMoisture')}</th><th>{t('condition')}</th></tr></thead><tbody>{readings.map(r=><tr key={r.id}><td>{r.time}</td><td>{r.cropType}</td><td>{r.temperature}°C</td><td>{r.humidity}%</td><td>{r.soilMoisture}%</td><td><span className={`tag ${tone(r.final_status)}`}>{labelStatus(r.final_status,t)}</span></td></tr>)}</tbody></table></div> }
function Sectors({embedded=false}){ 
  const {t}=useLang(); 
  const [items,setItems]=useState(getStore('sph_sectors',initialSectors)); 
  const [form,setForm]=useState({name:'',location:'',crop:'Tomato'});
  const [workerForm,setWorkerForm]=useState({name:'',role:'Plant Care Worker',phone:'',sectorId:''}); 
  const save=()=>{ 
    if(!form.name.trim()) return; 
    const sensors={temperature:25,humidity:60,soilMoisture:45,soilTemp:24,light:'Sufficient'};
    const all=[{...form,id:Date.now(),status:'Healthy',sensors,diagnosis:{condition:'Healthy',disease:'No Clear Disease Detected',confidence:.91,summary:'New sector is ready for automatic monitoring.'},workers:[],tasks:[],notes:[],rows:['Healthy','Healthy','Healthy','Healthy'],equipment:['Irrigation Pump','Water Valve'],devices:[]},...items]; 
    setItems(all); setStore('sph_sectors',all); setForm({name:'',location:'',crop:'Tomato'}); 
  }; 
  const del=id=>{ const all=items.filter(x=>x.id!==id); setItems(all); setStore('sph_sectors',all); }; 
  const healthy=items.filter(s=>s.status==='Healthy').length;
  const warning=items.filter(s=>String(s.status).includes('Moderate')).length;
  const danger=items.filter(s=>isDanger(s.status)).length;
  return <>
    {!embedded && <PageHead title={t('sectorsAsService')} sub={t('sectorsAsServiceSub')} action={<span className="mode-pill">{t('userDataOnly')}</span>}/>}
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
          <Field label={t('crop')}><select value={form.crop} onChange={e=>setForm({...form,crop:e.target.value})}><option>Tomato</option><option>Corn</option><option>Pepper</option><option>Mint</option></select></Field>
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
  const id=Number(loc.pathname.split('/').pop());
  const [sectors,setSectors]=useState(getStore('sph_sectors',initialSectors));
  const sector=sectors.find(s=>Number(s.id)===id) || sectors[0];
  const [worker,setWorker]=useState({name:'',role:'',phone:''});
  const [device,setDevice]=useState({name:'',type:'',value:'',unit:'',status:'Online'});
  const [task,setTask]=useState({name:'',priority:'Medium'});
  const [note,setNote]=useState('');
  const updateSector=(next)=>{
    const all=sectors.map(s=>s.id===sector.id?next:s);
    setSectors(all); setStore('sph_sectors',all);
  };
  const addWorker=()=>{
    if(!worker.name.trim()) return;
    updateSector({...sector, workers:[...(sector.workers||[]),{...worker,id:Date.now()}]});
    setWorker({name:'',role:'',phone:''});
  };
  const addDevice=()=>{
    if(!device.name.trim()) return;
    updateSector({...sector, devices:[...(sector.devices||[]),{...device,id:Date.now()}]});
    setDevice({name:'',type:'',value:'',unit:'',status:'Online'});
  };
  const addTask=()=>{
    if(!task.name.trim()) return;
    updateSector({...sector, tasks:[...(sector.tasks||[]),{...task,id:Date.now(),status:'todo',assignedTo:(sector.workers||[])[0]?.name||''}]});
    addNotification(`${t('taskSent')} ${task.name} - ${t('workerMobileNotification')}`,'warning');
    setTask({name:'',priority:'Medium'});
  };
  const addNote=()=>{
    if(!note.trim()) return;
    updateSector({...sector, notes:[note,...(sector.notes||[])]});
    setNote('');
  };
  const rows=sector.rows||['Healthy','Healthy','Moderate Stress','Healthy'];
  const nextIrrigation=isDanger(sector.status)?t('irrigationUrgent'):t('irrigationNormal');
  return <>
    <PageHead title={`${t('farmView')} - ${localizeValue(sector.name,t)}`} sub={t('createFullFarmExperience')} action={<NavLink className="secondary-btn" to="/sectors">{t('backToSectors')}</NavLink>}/>

    <section className={`farm-hero ${tone(sector.status)}`}>
      <div>
        <span>{t('sectorOverview')}</span>
        <h1>{localizeValue(sector.name,t)}</h1>
        <p>{localizeValue(sector.location,t)} • {localizeValue(sector.crop,t)}</p>
        <div className="farm-hero-tags">
          <b>{labelStatus(sector.status,t)}</b>
          <b>{translateModelText(sector.diagnosis?.disease,t)}</b>
          <b>{Math.round((sector.diagnosis?.confidence ?? .8)*100)}%</b>
        </div>
      </div>
      <LiveCamera sector={sector} />
    </section>

    <div className="grid-2">
      <Panel title={t('fieldMap')}>
        <div className="field-map">
          {rows.map((r,i)=><div key={i} className={`plant-row ${tone(r)}`}>
            <span>{i+1}</span>
            {[1,2,3,4,5,6].map(n=><i key={n}/>)}
            <b>{labelStatus(r,t)}</b>
          </div>)}
        </div>
      </Panel>
      <Panel title={t('sectorEnvironment')}>
        <ReadingGrid r={{cropType:sector.crop,...(sector.sensors||{})}}/>
        <div className="irrigation-box"><Sprout/><div><b>{t('nextIrrigation')}</b><span>{nextIrrigation}</span></div></div>
      </Panel>
    </div>

    <div className="grid-2">
      <Panel title={t('latestDiagnosis')}>
        <div className={`diagnosis-hero ${tone(sector.status)}`}>
          <div><span>{t('condition')}</span><h2>{labelStatus(sector.status,t)}</h2><p>{translateModelText(sector.diagnosis?.summary,t)}</p></div>
          <div className="disease-card"><span>{t('diseaseProblem')}</span><b>{translateModelText(sector.diagnosis?.disease,t)}</b></div>
        </div>
      </Panel>
      <Panel title={t('equipment')}>
        <div className="equipment-grid">{(sector.equipment||[]).map((e,i)=><div key={i} className="equipment-card"><Cpu size={20}/><b>{localizeValue(e,t)}</b><span>{t('sensorOnline')}</span></div>)}</div>
      </Panel>
    </div>

    <div className="grid-2">
      <Panel title={t('sectorSensors')}>
        <div className="device-list">{(sector.devices||[]).map(d=><div className="device-row" key={d.id}><Cpu size={18}/><div><b>{localizeValue(d.name,t)}</b><span>{localizeValue(d.type,t)} • {localizeValue(d.value,t)}{d.unit}</span></div><em>{localizeValue(d.status,t)}</em></div>)}</div>
        <div className="form-grid compact">
          <Field label={t('sensorName')}><input value={device.name} onChange={e=>setDevice({...device,name:e.target.value})}/></Field>
          <Field label={t('sensorType')}><input value={device.type} onChange={e=>setDevice({...device,type:e.target.value})}/></Field>
          <Field label={t('sensorValue')}><input value={device.value} onChange={e=>setDevice({...device,value:e.target.value})}/></Field>
          <Field label={t('sensorUnit')}><input value={device.unit} onChange={e=>setDevice({...device,unit:e.target.value})}/></Field>
        </div>
        <button className="primary-btn" onClick={addDevice}><Plus size={18}/>{t('addSensor')}</button>
      </Panel>
      <Panel title={t('assignedWorkers')}>
        <div className="sector-worker-summary">
          {(sector.workers||[]).length ? (sector.workers||[]).map((w,i)=><article className="sector-worker-card" key={w.id || i}>
            <span className="worker-avatar-mini">{(w.name||'W').split(' ').map(x=>x[0]).slice(0,2).join('').toUpperCase()}</span>
            <div>
              <b>{w.name}</b>
              <span>{localizeValue(w.role||w.jobTitle,t)} • {w.phone || w.email || '-'}</span>
              <em>{t('lastTask')}: {localizeValue((sector.tasks||[])[i]?.name || (sector.tasks||[])[0]?.name || 'Inspect leaves',t)}</em>
            </div>
            <strong className="tag green">{t('active')}</strong>
          </article>) : <p className="muted-copy">{t('emptyState')}</p>}
        </div>
        {isOwner() ? <NavLink className="secondary-btn wide" to="/workers"><UserPlus size={18}/>{t('addWorker')}</NavLink> : <span className="role-pill worker">{t('workerLimitedView')}</span>}
      </Panel>
    </div>

    <div className="grid-2">
      <Panel title={t('careTasks')}>
        <div className="task-list">{(sector.tasks||[]).map(x=><WorkerTaskCard key={x.id} task={x} sector={sector} updateSector={updateSector} t={t}/>)}</div>
        <div className="form-grid compact">
          <Field label={t('taskName')}><input value={task.name} onChange={e=>setTask({...task,name:e.target.value})}/></Field>
          <Field label={t('priority')}><select value={task.priority} onChange={e=>setTask({...task,priority:e.target.value})}><option>High</option><option>Medium</option><option>Low</option></select></Field>
        </div>
        <button className="primary-btn" onClick={addTask}><Plus size={18}/>{t('addTask')}</button>
      </Panel>
      <Panel title={t('notes')}>
        <div className="notes-list">{(sector.notes||[]).map((n,i)=><p key={i}>{translateModelText(n,t)}</p>)}</div>
        <Field label={t('noteText')}><input value={note} onChange={e=>setNote(e.target.value)}/></Field>
        <button className="primary-btn wide" onClick={addNote}><Plus size={18}/>{t('addNote')}</button>
      </Panel>
    </div>

    <Panel title={t('sectorChat')}>
      <MiniChat context={sector}/>
    </Panel>
  </>
}


function LiveCamera({sector}){
  const {t}=useLang();
  const [mode,setMode]=useState('live');
  const [online,setOnline]=useState(Boolean(sector?.cameraOnline ?? false));
  const snapshot = sector?.lastCameraImage || '';
  const retry=()=>{ setOnline(true); setMode('live'); addNotification(t('cameraOnline'),'success'); };
  return <div className={`live-camera pro-camera ${online?'online':'offline'}`}>
    <div className="camera-topline">
      <span><Camera size={18}/>{t('liveCamera')}</span>
      <b className={`tag ${online?'green':'amber'}`}>{online?t('cameraOnline'):t('cameraOffline')}</b>
    </div>
    <div className="camera-mode-tabs">
      <button className={mode==='live'?'active':''} onClick={()=>setMode('live')}><Activity size={16}/>{t('liveStreamMode')}</button>
      <button className={mode==='latest'?'active':''} onClick={()=>setMode('latest')}><ImagePlus size={16}/>{t('latestImageMode')}</button>
      <button className={mode==='offline'?'active':''} onClick={()=>{setMode('offline');setOnline(false);}}><AlertTriangle size={16}/>{t('cameraOfflineState')}</button>
    </div>
    <div className="camera-frame">
      {mode==='live' && online ? <div className="stream-placeholder"><span className="pulse-dot"/><Camera size={46}/><b>{t('liveStreamMode')}</b><p>{t('liveStreamReady')}</p></div> :
       mode==='latest' || !online ? <div className="snapshot-placeholder">{snapshot ? <img src={snapshot} alt={t('lastCapturedImage')}/> : <><ImagePlus size={46}/><b>{t('lastCapturedImage')}</b><p>{t('latestCapturedFallback')}</p></>} </div> :
       <div className="offline-placeholder"><AlertTriangle size={46}/><b>{t('cameraOfflineState')}</b><p>{t('cameraBackendHint')}</p></div>}
    </div>
    <div className="camera-actions">
      <button className="secondary-btn" onClick={retry}><Zap size={16}/>{t('retryCamera')}</button>
      <span>{t('cameraBackendHint')}</span>
    </div>
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
  const sectors=getUserStore('sph_sectors', getStore('sph_sectors',initialSectors));
  const [plants,setPlants]=useState(getUserStore('sph_plants',initialPlants));
  const [form,setForm]=useState({name:'',crop:'Tomato',sectorId:sectors[0]?.id||1,age:'',plantedAt:''});
  const save=()=>{
    if(!form.name.trim()) return;
    const p={...form,id:Date.now(),lastDiagnosis:'Healthy',disease:'No Clear Disease Detected',timeline:[]};
    const all=[p,...plants]; setPlants(all); setUserStore('sph_plants',all); setForm({name:'',crop:'Tomato',sectorId:sectors[0]?.id||1,age:'',plantedAt:''});
  };
  return <>
    <PageHead title={t('plantProfiles')} sub={t('userDataOnly')} action={null}/>
    <Panel title={t('addPlant')}>
      <div className="form-grid">
        <Field label={t('plantName')}><input value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></Field>
        <Field label={t('cropType')}><select value={form.crop} onChange={e=>setForm({...form,crop:e.target.value})}><option>Tomato</option><option>Corn</option><option>Pepper</option><option>Mint</option></select></Field>
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
  const sectors=getUserStore('sph_sectors', getStore('sph_sectors',initialSectors));
  const plants=getUserStore('sph_plants',initialPlants);
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
  const storedUser = getStore('ecosense_user', {}) || {};
  const defaultProfile = {
    soundAlerts:true,
    tempUnit:'C',
    profileName: storedUser.name || storedUser.fullName || 'Smart Plant User',
    phone: storedUser.phone || '',
    avatar: storedUser.picture || storedUser.avatar || storedUser.photoURL || '',
    farmName:'Ecosense Smart Farm',
    defaultCrop:'Tomato',
    timezone:'Africa/Cairo',
    twoFactor:false,
    sessionAlerts:true,
    reportAccess:'owner',
    themeMode: theme || 'light'
  };
  const [settings,setSettings]=useState({...defaultProfile,...getStore('sph_settings', defaultProfile)});
  const [uploadState,setUploadState]=useState('');
  const userEmail = storedUser.email || currentUserEmail();
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
    rolePermissions: lang==='ar'?'الدور والصلاحيات':'Role & Permissions', backendReady: lang==='ar'?'جاهز للربط بالباك إند':'Backend-ready', languageTheme: lang==='ar'?'اللغة والمظهر':'Language & theme', defaultCrop: lang==='ar'?'المحصول الافتراضي':'Default crop', farmLocation: lang==='ar'?'موقع المزرعة':'Farm location', timezone: lang==='ar'?'المنطقة الزمنية':'Timezone',
    soundAlerts: t('soundAlerts'), sessionAlerts: lang==='ar'?'تنبيهات تسجيل الدخول':'Login alerts', twoFactor: lang==='ar'?'تفعيل حماية إضافية':'Extra security protection', reportAccess: lang==='ar'?'صلاحية التقارير':'Report access', ownerOnly: lang==='ar'?'المالك فقط':'Owner only', lightMode: lang==='ar'?'وضع نهاري':'Light mode', darkMode: lang==='ar'?'وضع ليلي':'Dark mode'
  };
  const save=()=>{
    setStore('sph_settings',settings);
    const mergedUser = {...storedUser, name:settings.profileName, fullName:settings.profileName, phone:settings.phone, avatar:settings.avatar, picture:settings.avatar, email:userEmail};
    setStore('ecosense_user', mergedUser);
    setUploadState(lang==='ar'?'تم حفظ الإعدادات بنجاح.':'Settings saved successfully.');
    addNotification(lang==='ar'?'تم حفظ إعدادات الحساب.':'Account settings saved.','success');
  };
  const uploadProfilePhotoToBackend=async(file,preview)=>{
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
      setUploadState(lang==='ar'?'تمت معاينة الصورة. سيتم رفعها للباك إند عند توفر Endpoint.':'Image preview added. Backend storage will be used when endpoint is ready.');
      const backendUrl=await uploadProfilePhotoToBackend(file,preview);
      if(backendUrl){ setSettings(prev=>({...prev,avatar:backendUrl})); setUploadState(lang==='ar'?'تم رفع الصورة وتخزينها بنجاح.':'Profile image uploaded successfully.'); }
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
          <small>{roleDesc}</small>
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

        <section className="settings-card-v2 account-section">
          <header><span><ShieldCheck size={20}/></span><div><h3>{L.account}</h3><p>{L.rolePermissions}</p></div></header>
          <div className="settings-mini-list"><span>{t('role')}</span><b>{roleTitle}</b></div>
          <div className="settings-mini-list"><span>{t('assignedSector')}</span><b>{localizeValue(assignedSector,t)}</b></div>
          {!isWorker() && <div className="backend-storage-card small"><Cpu size={18}/><div><b>{L.backendReady}</b><p>{t('modelFinalStatusNote')}</p></div></div>}
          {isWorker() && <div className="worker-settings-safe-note"><Lock size={18}/><span>{lang==='ar'?'تم إخفاء إعدادات الإدارة والمالك من حساب العامل.':'Owner/admin settings are hidden from the worker account.'}</span></div>}
        </section>

        {!isWorker() && <section className="settings-card-v2 farm-section">
          <header><span><Layers3 size={20}/></span><div><h3>{L.farm}</h3><p>{lang==='ar'?'تفضيلات المزرعة الأساسية':'Main farm preferences'}</p></div></header>
          <Field label={t('farmName')}><input value={settings.farmName || ''} onChange={e=>setSettings({...settings,farmName:e.target.value})} placeholder={t('farmName')} disabled={!isOwner()}/></Field>
          <Field label={L.defaultCrop}><select value={settings.defaultCrop || 'Tomato'} onChange={e=>setSettings({...settings,defaultCrop:e.target.value})} disabled={!isOwner()}><option>Tomato</option><option>Corn</option><option>Pepper</option><option>Mint</option></select></Field>
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
  const [sectors,setSectors]=useState(getStore('sph_sectors',initialSectors));
  const [workers,setWorkers]=useState(getStore('sph_worker_accounts',[]));
  const [lastAccount,setLastAccount]=useState(null);
  const [form,setForm]=useState({name:'',email:'',phone:'',jobTitle:'Plant Care Worker',sectorId:String(initialSectors[0]?.id || ''), permissions:{tasks:true,alerts:true,upload:true,finishTask:true}});
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
    if(found) setForm(prev=>({...prev,name:found.name||'',email:found.email||'',phone:found.phone||'',jobTitle:found.jobTitle||found.role||'Plant Care Worker',sectorId:String(found.sectorId||prev.sectorId)}));
  };
  const createWorker=async()=>{
    if(!canManageWorkers){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    if(!form.name.trim() || !form.email.trim()){
      addNotification(lang==='ar'?'اكتب اسم العامل والإيميل أولًا.':'Enter worker name and email first.','warning');
      return;
    }
    const password=generateWorkerPassword();
    const account={
      id:Date.now(), name:form.name.trim(), email:form.email.trim(), username:form.email.trim(), password,
      phone:form.phone.trim(), role: normalizeRole(form.jobTitle, form.email), jobTitle:form.jobTitle || 'Plant Care Worker', status:'Active', permissions:form.permissions || {},
      sectorId:selectedSector?.id || form.sectorId, sector:selectedSector?.name || '-', lastTask:(selectedSector?.tasks||[])[0]?.name || 'Inspect leaves', createdAt:new Date().toLocaleString()
    };
    const token=localStorage.getItem('ecosense_token');
    if(token){
      try{
        await usersAPI.addWorker({name:account.name, fullName:account.name, email:account.email, password:account.password, phone:account.phone, role:account.role, jobTitle:account.jobTitle, sectorId:account.sectorId});
      }catch(err){ /* local worker account fallback */ }
    }
    const nextWorkers=[account,...workers];
    setWorkers(nextWorkers); setStore('sph_worker_accounts',nextWorkers); setLastAccount(account);
    const nextSectors=sectors.map(s=>String(s.id)===String(account.sectorId)?{...s,workers:[...(s.workers||[]),{id:account.id,name:account.name,role:account.jobTitle,phone:account.phone,email:account.email,status:'Active'}]}:s);
    setSectors(nextSectors); setStore('sph_sectors',nextSectors);
    setForm({name:'',email:'',phone:'',jobTitle:'Plant Care Worker',sectorId:String(selectedSector?.id || ''),permissions:{tasks:true,alerts:true,upload:true,finishTask:true}});
    addNotification(lang==='ar'?'تم إنشاء حساب العامل. انسخ الإيميل والباسورد وابعتهمله.':'Worker account created. Copy email and password for the worker.','success');
  };
  const removeWorker=(id)=>{
    const next=workers.filter(w=>w.id!==id); setWorkers(next); setStore('sph_worker_accounts',next);
    const nextSectors=sectors.map(s=>({...s,workers:(s.workers||[]).filter(w=>w.id!==id)})); setSectors(nextSectors); setStore('sph_sectors',nextSectors);
    if(lastAccount?.id===id) setLastAccount(null);
  };
  const toggleStatus=(worker)=>{
    const nextStatus = worker.status === 'Disabled' ? 'Active' : 'Disabled';
    const next=workers.map(w=>String(w.id)===String(worker.id)?{...w,status:nextStatus}:w);
    setWorkers(next); setStore('sph_worker_accounts',next);
    const nextSectors=sectors.map(s=>({...s,workers:(s.workers||[]).map(w=>String(w.id)===String(worker.id)?{...w,status:nextStatus}:w)}));
    setSectors(nextSectors); setStore('sph_sectors',nextSectors);
  };
  const assignWorker=(worker)=>{
    const sectorId = prompt(lang==='ar'?'اكتب رقم القطاع المراد تعيين العامل له':'Enter sector id to assign this worker', String(worker.sectorId || sectors[0]?.id || ''));
    const target=sectors.find(s=>String(s.id)===String(sectorId));
    if(!target) return;
    const nextWorkers=workers.map(w=>String(w.id)===String(worker.id)?{...w,sectorId:target.id,sector:target.name}:w);
    setWorkers(nextWorkers); setStore('sph_worker_accounts',nextWorkers);
    const nextSectors=sectors.map(s=>{
      const without=(s.workers||[]).filter(w=>String(w.id)!==String(worker.id));
      return String(s.id)===String(target.id) ? {...s,workers:[...without,{id:worker.id,name:worker.name,role:worker.jobTitle||worker.role,phone:worker.phone,email:worker.email,status:worker.status||'Active'}]} : {...s,workers:without};
    });
    setSectors(nextSectors); setStore('sph_sectors',nextSectors);
    addNotification(lang==='ar'?'تم تعيين العامل للقطاع.':'Worker assigned to sector.','success');
  };
  const editWorker=(worker)=>{
    setForm({name:worker.name||'',email:worker.email||'',phone:worker.phone||'',jobTitle:worker.jobTitle||worker.role||'Plant Care Worker',sectorId:String(worker.sectorId||sectors[0]?.id||''),permissions:worker.permissions||{tasks:true,alerts:true,upload:true,finishTask:true}});
    window.scrollTo({top:0,behavior:'smooth'});
  };
  const copyText=(text)=>{ try{ navigator.clipboard.writeText(text); addNotification(lang==='ar'?'تم نسخ بيانات الحساب.':'Account data copied.','success'); }catch{} };
  const copyAllAccount=(acc)=>copyText(`${t('workerName')}: ${acc.name}\n${t('role')}: ${localizeValue(acc.jobTitle || acc.role,t)}\n${t('sector')}: ${localizeValue(acc.sector,t)}\n${t('accountStatus')}: ${localizeValue(acc.status,t)}\n${t('username')}: ${acc.username || acc.email}\n${t('password')}: ${acc.password}`);
  return <>
    <PageHead title={L.title} sub={L.subtitle}/>
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

      {canManageWorkers && <div className="worker-management-grid-v2">
        <section className="worker-form-card-v2">
          <header><span><UserPlus size={22}/></span><div><h3>{L.addAssign}</h3><p>{t('permissionsOwnerOnly')}</p></div></header>
          <div className="worker-form-grid-v2">
            <Field label={L.existing}><select defaultValue="" onChange={e=>selectExistingWorker(e.target.value)}><option value="">{L.create}</option>{existingWorkers.map(w=><option key={w.id} value={w.id}>{w.name} - {localizeValue(w.sector,t)}</option>)}</select></Field>
            <Field label={t('workerName')}><input value={form.name} onChange={e=>setForm({...form,name:e.target.value})} placeholder={lang==='ar'?'مثال: أحمد محمد':'Example: Ahmed Mohamed'}/></Field>
            <Field label={t('email')}><input type="email" value={form.email} onChange={e=>setForm({...form,email:e.target.value})} placeholder="worker@ecosense.ai"/></Field>
            <Field label={t('phone')}><input value={form.phone} onChange={e=>setForm({...form,phone:e.target.value})} placeholder="01000000000"/></Field>
            <Field label={L.role}><select value={form.jobTitle} onChange={e=>setForm({...form,jobTitle:e.target.value})}><option>Farm Manager</option><option>Irrigation Worker</option><option>Plant Care Worker</option></select></Field>
            <Field label={t('selectSector')}><select value={form.sectorId} onChange={e=>setForm({...form,sectorId:e.target.value})}>{sectors.map(s=><option key={s.id} value={s.id}>{localizeValue(s.name,t)}</option>)}</select></Field>
          </div>
          <div className="worker-permissions-grid v2">
            {[['tasks',t('workerOnlyTasks')],['alerts',t('alerts')],['upload',t('proofUpload')],['finishTask',t('completeTask')]].map(([key,label])=><label key={key}><input type="checkbox" checked={form.permissions?.[key]!==false} onChange={e=>setForm({...form,permissions:{...(form.permissions||{}),[key]:e.target.checked}})}/><span>{label}</span></label>)}
          </div>
          <button className="primary-btn worker-create-submit" onClick={createWorker}><Lock size={18}/>{L.credentials}</button>
        </section>

        <section className="worker-credentials-card-v2 account-result-card clean-generated-account-card">
          <div className="credentials-icon"><ShieldCheck size={30}/></div>
          <h3>{t('workerAccountDataTitle')}</h3>
          {lastAccount ? <div className="account-result-content">
            <div className="generated-account-summary">
              <div className="worker-avatar-v2 small"><span>{(lastAccount.name||'W').slice(0,1).toUpperCase()}</span></div>
              <div className="generated-account-fields">
                <div><span>{t('workerName')}</span><b>{lastAccount.name}</b></div>
                <div><span>{t('role')}</span><b>{localizeValue(lastAccount.jobTitle || lastAccount.role,t)}</b></div>
                <div><span>{t('sector')}</span><b>{localizeValue(lastAccount.sector,t)}</b></div>
                <div><span>{t('accountStatus')}</span><b>{localizeValue(lastAccount.status,t)}</b></div>
              </div>
            </div>
            <div className="account-divider" />
            <div className="credentials-box modern clean-credentials-box">
              <div className="credential-line">
                <div><span>{t('username')}</span><code>{lastAccount.username || lastAccount.email}</code></div>
                <button onClick={()=>copyText(lastAccount.username || lastAccount.email)}>{t('copyUsername')}</button>
              </div>
              <div className="credential-line">
                <div><span>{t('password')}</span><code>{lastAccount.password}</code></div>
                <button onClick={()=>copyText(lastAccount.password)}>{t('copyPassword')}</button>
              </div>
            </div>
            <button className="primary-btn wide" onClick={()=>copyAllAccount(lastAccount)}><FileText size={16}/>{t('copyAll')}</button>
            <p>{t('accountPasswordHint')}</p>
          </div> : <p className="credentials-empty">{t('workerCredentialsEmpty')}</p>}
        </section>
      </div>}

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
                  <span><b>{L.role}</b>{localizeValue(w.jobTitle || w.role,t)}</span>
                  <span><b>{L.phone}</b>{w.phone || '-'}</span>
                  <span><b>{L.sector}</b>{localizeValue(w.sector || selectedSector?.name,t)}</span>
                  <span><b>{L.lastTask}</b>{localizeValue(w.lastTask || 'Inspect leaves',t)}</span>
                </div>
              </div>
              <div className="worker-actions-v2">
                {canManageWorkers ? <><button className="secondary-btn" onClick={()=>editWorker(w)}><FileText size={15}/>{L.edit}</button>
                <button className="secondary-btn" onClick={()=>assignWorker(w)}><Layers3 size={15}/>{L.assign}</button>
                {isStored ? <button className="danger-soft-btn" onClick={()=>toggleStatus(w)}><X size={15}/>{L.disable}</button> : <button className="danger-soft-btn" onClick={()=>removeWorker(w.id)}><X size={15}/>{t('delete')}</button>}</> : <span className="manager-view-only"><Eye size={15}/>{t('workerLimitedView')}</span>}
              </div>
            </article>}) : <div className="farm-empty-focus small"><p>{t('emptyState')}</p></div>}
        </div>
      </section>
    </section>
  </>
}


function FarmSensorsPanel({sectors=[]}){
  const {t}=useLang();
  const [historyOpen,setHistoryOpen]=useState(false);
  const readings=getStore('sph_readings',initialReadings).map((r,i)=>({...r,sector:r.sector || sectors[i%Math.max(sectors.length,1)]?.name || 'Greenhouse A'}));
  const sensorRows=sectors.map((sector,i)=>{
    const s=sector.sensors || {};
    return {
      id:sector.id,
      sector:sector.name,
      crop:sector.crop,
      status:sector.status,
      source: sector.source || (sector.backendLive ? 'Live' : 'Simulation'),
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
  const [sectors,setSectors]=useState(getStore('sph_sectors',initialSectors));
  const [backendLive,setBackendLive]=useState(false);
  const [form,setForm]=useState({name:'',location:'',crop:'Tomato'});
  const [workerForm,setWorkerForm]=useState({name:'',role:'Plant Care Worker',phone:'',sectorId:String(initialSectors[0]?.id || '')});
  const workers=sectors.flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name})));
  const visibleSectors=isWorker()?sectors.filter(s=>String(s.id)===String(currentUser()?.sectorId) || String(s.name)===String(currentUser()?.sector) || (s.workers||[]).some(w=>String(w.email||'').toLowerCase()===currentUserEmail())):sectors;
  const devices=visibleSectors.flatMap(s=>(s.devices||[]).map(d=>({...d,sector:s.name,sectorId:s.id})));
  const tasks=getGlobalTasks().filter(task=>!isWorker() || taskAssignedToCurrent(task));
  const savedFarmSettings=getStore('sph_settings',{});
  const [adminFarm,setAdminFarm]=useState({farmName:savedFarmSettings.farmName || 'Ecosense Smart Farm', crop:savedFarmSettings.defaultCrop || 'Tomato', location:savedFarmSettings.farmLocation || 'North Field', manager:savedFarmSettings.farmManager || workers[0]?.name || ''});
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
          crop:item.crop || item.cropType || 'Tomato',
          status:item.status || item.final_status || item.healthStatus || 'Healthy',
          sensors:item.sensors || {temperature:25,humidity:60,soilMoisture:45,soilTemp:24,light:'Sufficient'},
          diagnosis:item.diagnosis || {condition:item.status || 'Healthy', disease:item.disease || 'No Clear Disease Detected', confidence:item.confidence || .9, summary:item.summary || 'Backend sector loaded.'},
          workers:item.workers || [], tasks:item.tasks || [], notes:item.notes || [], rows:item.rows || ['Healthy','Healthy','Healthy','Healthy'], equipment:item.equipment || ['Irrigation Pump','Water Valve'], devices:item.devices || []
        }));
        setSectors(normalized); setStore('sph_sectors',normalized); setBackendLive(true);
      }
    }).catch(()=>{});
    devicesAPI.getAll().then(({data})=>{
      const list = Array.isArray(data) ? data : (data.devices || data.data || []);
      if(mounted && list.length){
        const current=getStore('sph_sectors', initialSectors);
        const next=current.map(sec=>({...sec, devices:[...(sec.devices||[]), ...list.filter(d=>String(d.sectorId || d.sector || '')===String(sec.id) || String(d.sectorName || '')===String(sec.name)).map((d,i)=>({id:d._id||d.id||`${sec.id}-${i}`, name:d.name||d.deviceName||'Farm device', type:d.type||d.deviceType||'Sensor Device', status:d.status||'Online', value:d.lastReading?.value ?? d.value ?? '-', unit:d.lastReading?.unit ?? d.unit ?? '', lastContact:d.lastContact||d.updatedAt||new Date().toLocaleString(), battery:d.battery}))]}));
        setSectors(next); setStore('sph_sectors',next); setBackendLive(true);
      }
    }).catch(()=>{});
    return()=>{mounted=false};
  },[]);
  const ownerLabel = lang==='ar' ? 'مالك' : t('owner');
  const saveSector=async()=>{
    if(!form.name.trim()) return;
    const sensors={temperature:25,humidity:60,soilMoisture:45,soilTemp:24,light:'Sufficient'};
    let created={...form,id:Date.now(),status:'Healthy',sensors,diagnosis:{condition:'Healthy',disease:'No Clear Disease Detected',confidence:.91,summary:'New sector is ready for automatic monitoring.'},workers:[],tasks:[],notes:[],rows:['Healthy','Healthy','Healthy','Healthy'],equipment:['Irrigation Pump','Water Valve'],devices:[]};
    const token=localStorage.getItem('ecosense_token');
    if(token){
      try{
        const {data}=await sectorsAPI.create({name:form.name,location:form.location,crop:form.crop,cropType:form.crop});
        const item=data.sector || data.data || data;
        created={...created,...item,id:item._id || item.id || created.id,name:item.name || form.name,location:item.location || form.location,crop:item.crop || item.cropType || form.crop};
        setBackendLive(true);
      }catch(err){ /* local sector fallback */ }
    }
    const next=[created,...sectors];
    setSectors(next); setStore('sph_sectors',next); setForm({name:'',location:'',crop:'Tomato'}); setShowAdd(false); setView('sectors');
  };
  const saveAdminFarm=()=>{
    const current=getStore('sph_settings',{});
    setStore('sph_settings',{...current,farmName:adminFarm.farmName,defaultCrop:adminFarm.crop,farmLocation:adminFarm.location,farmManager:adminFarm.manager});
    addNotification(lang==='ar'?'تم تحديث بيانات المزرعة من لوحة الإدارة.':'Farm data updated from admin dashboard.','success');
  };
  const deleteSectorAdmin=(id)=>{
    if(!hasPermission('deleteSector')){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    const next=sectors.filter(s=>String(s.id)!==String(id));
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم حذف القطاع.':'Sector deleted.','warning');
  };
  const editSectorAdmin=(sector)=>{
    const name=prompt(lang==='ar'?'اسم القطاع الجديد':'New sector name', sector.name);
    if(!name) return;
    const crop=prompt(lang==='ar'?'نوع المحصول':'Crop type', sector.crop) || sector.crop;
    const location=prompt(lang==='ar'?'بيانات أو موقع القطاع':'Sector details or location', sector.location) || sector.location;
    const next=sectors.map(s=>String(s.id)===String(sector.id)?{...s,name,crop,location}:s);
    setSectors(next); setStore('sph_sectors',next);
    addNotification(lang==='ar'?'تم تعديل بيانات القطاع.':'Sector updated.','success');
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
    const token=localStorage.getItem('ecosense_token');
    if(token){
      try{
        const {data}=await usersAPI.addWorker({name:workerForm.name,role:workerForm.role,phone:workerForm.phone,sectorId:targetId});
        const item=data.worker || data.user || data.data || data;
        newWorker={...newWorker,...item,id:item._id || item.id || newWorker.id};
        setBackendLive(true);
      }catch(err){ /* local worker fallback */ }
    }
    const next = sectors.map(s=>Number(s.id)===targetId ? {...s, workers:[...(s.workers||[]), newWorker]} : s);
    setSectors(next); setStore('sph_sectors',next); setWorkerForm({name:'',role:'Plant Care Worker',phone:'',sectorId:targetId});
  };
  const deleteWorker=async(workerId)=>{
    if(!isOwner()) return;
    const token=localStorage.getItem('ecosense_token');
    if(token){ try{ await usersAPI.deleteWorker(workerId); setBackendLive(true); } catch(err){ /* local delete fallback */ } }
    const next = sectors.map(s=>({...s, workers:(s.workers||[]).filter(w=>w.id!==workerId)}));
    setSectors(next); setStore('sph_sectors',next);
  };
  const cards=[
    {key:'overview', icon:Layers3, label:t('farmOverview'), count:backendLive?t('apiConnected'):t('apiFallback'), hint:t('sectorsDevicesStatus')},
    {key:'sectors', icon:Sprout, label:t('sectors'), count:visibleSectors.length, hint:t('sectorHealth')},
    {key:'devices', icon:Cpu, label:t('devices'), count:devices.length, hint:t('devicesAndSensors')},
    {key:'sensors', icon:ThermometerSun, label:t('farmSensors'), count:visibleSectors.length, hint:t('farmSensorsSub')},
    {key:'tasks', icon:CheckCircle2, label:t('farmTasks'), count:tasks.length, hint:t('careTasks')},
    {key:'alerts', icon:Bell, label:t('farmAlerts'), count:notificationStore().length, hint:t('alertsManagement')},
    {key:'reports', icon:FileText, label:t('farmReports'), count:getStore('sph_readings',initialReadings).length, hint:t('reportsManagement')},
    {key:'admin', icon:ShieldCheck, label:t('adminPanel'), count:ownerLabel, hint:t('ownerPermissions')}
  ].filter(card => isOwner() || (isFarmManager() && ['overview','sectors','devices','sensors','tasks','alerts','reports'].includes(card.key)) || (isWorker() && ['sectors','tasks','alerts'].includes(card.key)));
  return <div className="farm-slim-page">
    <PageHead title={t('farmManagementTitle')} sub={t('farmManagementSub')} action={isOwner()?<button className="farm-add-fab" onClick={()=>setShowAdd(!showAdd)}><Plus size={18}/>{t('addSector')}</button>:<AccountRoleBadge/>}/>

    <section className="farm-service-row">
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
        <Field label={t('crop')}><select value={form.crop} onChange={e=>setForm({...form,crop:e.target.value})}><option>Tomato</option><option>Corn</option><option>Pepper</option><option>Mint</option></select></Field>
        <button className="primary-btn" onClick={saveSector}><Plus size={18}/>{t('save')}</button>
      </div>
    </section>}

    {view==='overview' && <section className="farm-production-overview">
      <div className="farm-overview-hero"><Layers3 size={42}/><div><h3>{t('sectorsDevicesStatus')}</h3><p>{t('farmManagementSub')}</p></div></div>
      <div className="farm-overview-metrics">
        <Box label={t('totalSectors')} value={visibleSectors.length}/>
        <Box label={t('devices')} value={devices.length}/>
        <Box label={t('latestDiagnosis')} value={labelStatus(sectors[0]?.status,t)}/>
        <Box label={t('apiStatus')} value={backendLive?t('apiConnected'):t('apiFallback')}/>
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
    {view==='alerts' && <FarmAlertsPanel sectors={visibleSectors} devices={devices}/>}
    {view==='reports' && <FarmReportsPanel sectors={visibleSectors}/>}
    {view==='admin' && isOwner() && <section className="admin-dashboard-v2">
      <div className="admin-hero-v2">
        <div className="admin-hero-icon"><ShieldCheck size={34}/></div>
        <div><span>{t('adminPanel')}</span><h3>{lang==='ar'?'لوحة صلاحيات كاملة للمالك':'Owner production control dashboard'}</h3><p>{t('allFarmAccess')}</p></div>
        <strong>{ownerLabel}</strong>
      </div>
      <div className="admin-sections-grid-v2">
        <section className="admin-section-card-v2 farm-data">
          <header><Layers3 size={20}/><div><h4>{lang==='ar'?'بيانات المزرعة':'Farm Data'}</h4><p>{lang==='ar'?'تعديل اسم المزرعة، نوع المحصول، المدير، والبيانات الأساسية.':'Edit farm name, crop type, manager, and core details.'}</p></div></header>
          <div className="admin-form-grid-v2">
            <Field label={t('farmName')}><input value={adminFarm.farmName} onChange={e=>setAdminFarm({...adminFarm,farmName:e.target.value})}/></Field>
            <Field label={t('cropType')}><select value={adminFarm.crop} onChange={e=>setAdminFarm({...adminFarm,crop:e.target.value})}><option>Tomato</option><option>Corn</option><option>Pepper</option><option>Mint</option></select></Field>
            <Field label={t('location')}><input value={adminFarm.location} onChange={e=>setAdminFarm({...adminFarm,location:e.target.value})}/></Field>
            <Field label={lang==='ar'?'مدير المزرعة':'Farm Manager'}><select value={adminFarm.manager} onChange={e=>setAdminFarm({...adminFarm,manager:e.target.value})}><option value="">{lang==='ar'?'اختر مدير':'Select manager'}</option>{workers.map(w=><option key={w.id || w.name} value={w.name}>{w.name}</option>)}</select></Field>
          </div>
          <button className="primary-btn" onClick={saveAdminFarm}><ShieldCheck size={18}/>{t('save')}</button>
        </section>

        <section className="admin-section-card-v2 permissions">
          <header><Lock size={20}/><div><h4>{lang==='ar'?'إدارة الصلاحيات':'User Permissions'}</h4><p>{lang==='ar'?'الخدمات الحساسة للمالك فقط، والعامل يرى مهامه وقطاعه فقط.':'Sensitive services are owner-only; workers only see tasks and assigned sector.'}</p></div></header>
          <div className="admin-permission-list-v2">
            {[t('addSector'),lang==='ar'?'حذف قطاع':'Delete sector',lang==='ar'?'تعديل قطاع':'Edit sector',t('addWorker'),t('generatedCredentials'),t('exportPdf'),t('exportCsv'),t('alerts')].map((x,i)=><span key={i}><CheckCircle2 size={15}/>{x}</span>)}
          </div>
        </section>

        <section className="admin-section-card-v2 sectors-control">
          <header><Sprout size={20}/><div><h4>{lang==='ar'?'إدارة القطاعات':'Sector Control'}</h4><p>{lang==='ar'?'إضافة، تعديل، حذف، وتعيين مسؤول لكل قطاع.':'Add, edit, delete, and assign supervisors to sectors.'}</p></div></header>
          <div className="admin-sector-list-v2">
            {sectors.map(s=><article key={s.id}>
              <div><b>{localizeValue(s.name,t)}</b><span>{localizeValue(s.crop,t)} • {localizeValue(s.location,t)}</span></div>
              <em className={`tag ${tone(s.status)}`}>{labelStatus(s.status,t)}</em>
              <div className="admin-row-actions-v2">
                <button onClick={()=>editSectorAdmin(s)}><FileText size={15}/>{lang==='ar'?'تعديل':'Edit'}</button>
                <button onClick={()=>assignAdminWorkerToSector(s)}><UserPlus size={15}/>{lang==='ar'?'تعيين':'Assign'}</button>
                <button className="danger" onClick={()=>deleteSectorAdmin(s.id)}><X size={15}/>{t('delete')}</button>
              </div>
            </article>)}
          </div>
        </section>

        <section className="admin-section-card-v2 devices-reports">
          <header><Cpu size={20}/><div><h4>{lang==='ar'?'الأجهزة والتنبيهات والتقارير':'Devices, Alerts & Reports'}</h4><p>{lang==='ar'?'ربط أو حذف جهاز، التحكم في التنبيهات، وإدارة التصدير.':'Link/remove devices, control alerts, and manage exports.'}</p></div></header>
          <div className="admin-action-tiles-v2">
            <button onClick={()=>setView('devices')}><Cpu size={18}/>{lang==='ar'?'إدارة الأجهزة':'Manage devices'}</button>
            <button onClick={()=>addNotification(t('ownerSensitiveBlocked'),'warning')}><X size={18}/>{lang==='ar'?'حذف جهاز حساس':'Protected device removal'}</button>
            <button onClick={()=>setView('alerts')}><Bell size={18}/>{lang==='ar'?'إدارة التنبيهات':'Manage alerts'}</button>
            <button onClick={()=>setView('reports')}><FileText size={18}/>{lang==='ar'?'إدارة التقارير':'Manage reports'}</button>
          </div>
        </section>
      </div>
    </section>}
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
  const {t}=useLang();
  const sectors=getStore('sph_sectors',initialSectors);
  const workers=sectors.flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name})));
  return <>{!embedded && <PageHead title={t('workers')} sub={t('assignedWorkers')}/>}<Panel title={t('workers')}><div className="table-wrap"><table><thead><tr><th>{t('workerName')}</th><th>{t('workerRole')}</th><th>{t('phone')}</th><th>{t('sector')}</th></tr></thead><tbody>{workers.map(w=><tr key={w.id}><td>{w.name}</td><td>{localizeValue(w.role,t)}</td><td>{w.phone}</td><td>{localizeValue(w.sector,t)}</td></tr>)}</tbody></table></div></Panel></>
}

function DevicesPage({embedded=false}){
  const {t}=useLang();
  const sectors=getStore('sph_sectors',initialSectors);
  const devices=sectors.flatMap(s=>(s.devices||[]).map(d=>({...d,sector:s.name})));
  return <>{!embedded && <PageHead title={t('devices')} sub={t('sectorSensors')}/>}<Panel title={t('devices')}><div className="table-wrap"><table><thead><tr><th>{t('sensorName')}</th><th>{t('sensorType')}</th><th>{t('sensorValue')}</th><th>{t('status')}</th><th>{t('sector')}</th></tr></thead><tbody>{devices.map(d=><tr key={d.id}><td>{localizeValue(d.name,t)}</td><td>{localizeValue(d.type,t)}</td><td>{localizeValue(d.value,t)}{d.unit}</td><td>{localizeValue(d.status,t)}</td><td>{localizeValue(d.sector,t)}</td></tr>)}</tbody></table></div></Panel></>
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
  const readings=getStore('sph_readings',initialReadings); 
  const sectors=getStore('sph_sectors',initialSectors);
  const [status,setStatus]=useState('all');
  const [sector,setSector]=useState('all');
  const [date,setDate]=useState('');
  const reportRows=readings.map((r,i)=>({
    ...r,
    sector: r.sector || sectors[i % Math.max(sectors.length,1)]?.name || 'Greenhouse A',
    date: r.date || new Date().toISOString().slice(0,10),
    diagnosis: r.diagnosis || (r.final_status==='Healthy' ? 'Stable plant condition with balanced readings.' : 'Water or heat stress indicators detected.'),
    recommendations: r.recommendations || [r.solution || 'Adjust irrigation according to soil moisture.'],
    actions: r.actions || [r.final_status==='High Stress' ? 'Create treatment task' : 'Monitor readings'],
    image: r.image || '',
    workerTask: r.workerTask || (r.final_status==='High Stress' ? 'Inspect leaves' : 'Weekly follow-up')
  })).filter(r => (status==='all' || r.final_status===status) && (sector==='all' || String(r.sector)===String(sector)) && (!date || r.date===date));
  const total=reportRows.length;
  const healthy=reportRows.filter(r=>r.final_status==='Healthy').length;
  const warning=reportRows.filter(r=>String(r.final_status).includes('Moderate')).length;
  const danger=reportRows.filter(r=>isDanger(r.final_status)).length;
  const preview=reportRows[0] || readings[0];
  const csv=()=>{
    const header=['time','sector','cropType','final_status','temperature','humidity','soilMoisture','soilTemp','light','diagnosis','recommendations','actions','workerTask'];
    const lines=[header.join(','),...reportRows.map(r=>header.map(k=>`"${String(Array.isArray(r[k])?r[k].join(' | '):(r[k]??'')).replaceAll('"','""')}"`).join(','))];
    download('ecosense-diagnosis-report.csv', lines.join('\n'), 'text/csv');
  };
  const pdf=()=>{
    openProfessionalReport({ t, lang, rows: reportRows, titleKey:'farmReports' });
  };
  return <>
    <PageHead title={t('farmReports')} sub={t('reportsCombinedNotice')} action={<div className="report-actions"><button className="secondary-btn" onClick={pdf}><FileText size={18}/>{t('exportPdf')}</button><button className="primary-btn" onClick={csv}><BarChart3 size={18}/>{t('exportCsv')}</button></div>}/>
    <div className="report-filter-panel panel">
      <Field label={t('filterByDate')}><input type="date" value={date} onChange={e=>setDate(e.target.value)}/></Field>
      <Field label={t('filterByStatus')}><select value={status} onChange={e=>setStatus(e.target.value)}><option value="all">{t('filterByStatusAll')}</option><option>Healthy</option><option>Moderate Stress</option><option>High Stress</option></select></Field>
      <Field label={t('filterBySector')}><select value={sector} onChange={e=>setSector(e.target.value)}><option value="all">{t('allSectors')}</option>{sectors.map(s=><option key={s.id} value={s.name}>{localizeValue(s.name,t)}</option>)}</select></Field>
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
            <div className="report-preview-topline">
              <span className={`tag ${tone(preview.final_status)}`}>{labelStatus(preview.final_status,t)}</span>
              <small>{t('diagnosisTime')}: {preview.date} {preview.time}</small>
            </div>
            <h3>{localizeValue(preview.sector || sectors[0]?.name,t)}</h3>
            <p className="report-diagnosis-short"><b>{t('diagnosisSummary')}:</b> {translateModelText(String(preview.diagnosis || '').slice(0,120),t)}{String(preview.diagnosis||'').length>120?'...':''}</p>
            <div className="report-mini-readings">
              <span>{t('temperature')}: <b>{formatInputValue('temperature',preview.temperature)}</b></span>
              <span>{t('humidity')}: <b>{formatInputValue('humidity',preview.humidity)}</b></span>
              <span>{t('soilMoisture')}: <b>{formatInputValue('soilMoisture',preview.soilMoisture)}</b></span>
            </div>
            <div className="report-preview-actions">
              <button className="secondary-btn" onClick={()=>alert(`${t('finalStatus')}: ${labelStatus(preview.final_status,t)}\n${t('diagnosisSummary')}: ${translateModelText(preview.diagnosis,t)}`)}><Eye size={16}/>{lang==='ar'?'عرض التقرير كامل':'View Full Report'}</button>
              <button className="primary-btn" onClick={pdf}><FileText size={16}/>{t('exportPdf')}</button>
            </div>
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
  const [notes,setNotes]=useState(notificationStore().map(n=>({...n,read:n.read||false}))); 
  const sectors=getStore('sph_sectors',initialSectors);
  const risky=getStore('sph_readings',initialReadings).filter(r=>r.final_status!=='Healthy');
  const typedAlerts=[
    ...notes.map((n,i)=>({id:n.id,type:n.type==='danger'?'criticalAlerts':'warningAlerts',reason:n.message,sector:sectors[i%sectors.length]?.name,time:n.time,severity:n.type==='danger'?'High':'Medium',read:n.read})),
    ...risky.map((r,i)=>({id:`r-${r.id}`,type:isDanger(r.final_status)?'criticalAlerts':'warningAlerts',reason:labelStatus(r.final_status,t),sector:sectors[i%sectors.length]?.name,time:r.time,severity:isDanger(r.final_status)?'High':'Medium',reading:r})),
    {id:'device-1',type:'deviceAlerts',reason:t('cameraOffline'),sector:sectors[0]?.name,time:new Date().toLocaleTimeString(),severity:'Medium'},
    {id:'disease-1',type:'diseaseAlerts',reason:t('leafSpotFungal'),sector:sectors[2]?.name,time:new Date().toLocaleTimeString(),severity:'High'},
    {id:'sensor-1',type:'sensorOfflineAlerts',reason:t('sensorOffline'),sector:sectors[1]?.name,time:new Date().toLocaleTimeString(),severity:'Medium'}
  ];
  const counts=['criticalAlerts','warningAlerts','deviceAlerts','diseaseAlerts','sensorOfflineAlerts'].map(k=>[k,typedAlerts.filter(a=>a.type===k).length]);
  const markAll=()=>{const all=notes.map(n=>({...n,read:true})); setNotes(all); setStore('sph_notifications',all);};
  const createTask=(a)=>{
    const workers=workerAccounts();
    const worker=workers[0] || {};
    const next={id:Date.now(),title:a.reason,details:a.reason,status:'pending',priority:a.severity,assignedTo:worker.name || 'Plant Care Worker',assignedToEmail:worker.email || worker.username || '',createdAt:new Date().toLocaleString(),dueDate:new Date(Date.now()+86400000).toISOString().slice(0,10),sector:a.sector};
    setGlobalTasks([next,...getGlobalTasks()]);
    if(next.assignedToEmail) addUserNotification(next.assignedToEmail, taskToastMessage('created',lang), 'success');
    addNotification(t('treatmentTaskCreated'),'success');
  };
  return <>
    <PageHead title={t('alertsTitle')} sub={t('alertsSubtitle')} action={<button className="secondary-btn" onClick={markAll}>{t('markAllRead')}</button>}/>
    <div className="notice-types pro-alert-types">
      {counts.map(([k,c])=><span key={k}>{t(k)} <b>{c}</b></span>)}
    </div>
    <div className="alert-card-list">
      {typedAlerts.map(a=><article className={`alert-pro-card ${String(a.severity).toLowerCase()}`} key={a.id}>
        <div className="alert-pro-icon"><AlertTriangle/></div>
        <div className="alert-pro-body">
          <div className="alert-pro-head"><b>{t(a.type)}</b><span className={`tag ${a.severity==='High'?'red':'amber'}`}>{localizeValue(a.severity,t)}</span></div>
          <p><strong>{t('alertReason')}:</strong> {translateModelText(a.reason,t)}</p>
          <small>{t('sector')}: {localizeValue(a.sector,t)} • {t('time')}: {a.time} • {t('alertSeverity')}: {localizeValue(a.severity,t)}</small>
        </div>
        <div className="alert-pro-actions"><button className="secondary-btn" onClick={()=>nav('/diagnosis')}><Eye size={16}/>{t('viewDiagnosis')}</button><button className="primary-btn" onClick={()=>createTask(a)}><CheckCircle2 size={16}/>{t('createTask')}</button></div>
      </article>)}
      {!typedAlerts.length && <Panel title={t('noCriticalAlerts')}/>} 
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
  const readings=getStore('sph_readings',initialReadings);
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
    <div className="compact-section-head"><div><h3>{t('farmAlerts')}</h3><p>{t('linkedBackendFeature')}</p></div><span className="compact-head-icon"><Bell/></span></div>
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
  const readings=getStore('sph_readings',initialReadings).map((r,i)=>({...r,sector:r.sector || sectors[i%Math.max(sectors.length,1)]?.name || '-', diagnosis:r.diagnosis || (r.final_status==='Healthy'?'Stable plant condition with balanced readings.':'Water or heat stress indicators detected.'), date:r.date || new Date().toISOString().slice(0,10)}));
  const bySector=sectors.map(s=>({name:s.name,count:readings.filter(r=>String(r.sector)===String(s.name)).length,status:s.status}));
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
        </div>
      </article>)}
    </div>
  </section>
}

function DeviceInventory({devices=[], sectors=[]}){
  const {t,lang}=useLang();
  const [localDevices,setLocalDevices]=useState(devices);
  const [selected,setSelected]=useState(null);
  useEffect(()=>setLocalDevices(devices),[devices]);
  const syncDevices=(next)=>{
    setLocalDevices(next);
    const allSectors=getStore('sph_sectors',initialSectors).map(sec=>({...sec,devices:next.filter(d=>String(d.sectorId)===String(sec.id) || String(d.sector)===String(sec.name)).map(({sector,sectorId,...rest})=>rest)}));
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
  const remove=(device)=>{
    if(!hasPermission('deleteDevice')){ addNotification(t('ownerSensitiveBlocked'),'warning'); return; }
    const next=enriched.filter(d=>String(d.id || d.name)!==String(device.id || device.name));
    syncDevices(next);
    addNotification(t('deviceRemoved'),'warning');
  };
  return <section className="compact-section device-inventory-section">
    <div className="compact-section-head"><div><h3>{t('devices')}</h3><p>{t('linkedBackendFeature')}</p></div><span className="compact-head-icon"><Cpu/></span></div>
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
      </article>):<div className="farm-empty-focus small"><p>{t('emptyState')}</p><small>{t('backendStructureReady')}</small></div>}
    </div>
    {selected && <div className="modal-backdrop"><section className="modal-card device-detail-modal"><button className="modal-close" onClick={()=>setSelected(null)}><X size={18}/></button><h3>{t('deviceDetails')}</h3><ReadingGrid r={{cropType:selected.type,temperature:selected.value||'-',humidity:selected.battery||'-',soilMoisture:selected.lastReading||'-',soilTemp:selected.lastContact||'-',light:selected.status||'-'}}/><p className="muted-copy">{t('linkedBackendFeature')}</p></section></div>}
  </section>
}

function CareTasksPanel({tasks=[], workers=[], sectors=[]}){
  const {t,lang}=useLang();
  const [all,setAll]=useState(getGlobalTasks());
  useEffect(()=>setAll(getGlobalTasks()),[]);
  const assign=(task)=>{
    const workerName=prompt(lang==='ar'?'اسم العامل المسؤول':'Responsible worker', task.assignedTo || workers[0]?.name || '');
    if(!workerName) return;
    const worker=workers.find(w=>String(w.name).toLowerCase()===String(workerName).toLowerCase()) || workerAccounts().find(w=>String(w.name).toLowerCase()===String(workerName).toLowerCase());
    const next=all.map(x=>String(x.id)===String(task.id)?{...x,assignedTo:workerName,assignedToEmail:worker?.email || x.assignedToEmail,status:'pending'}:x);
    setAll(next); setGlobalTasks(next);
    if(worker?.email) addUserNotification(worker.email, taskToastMessage('created',lang), 'success');
    addNotification(lang==='ar'?'تم تعيين المهمة.':'Task assigned.','success');
  };
  const list=(tasks.length?tasks:all).map(t=>({...t,status:normalizeTaskStatus(t.status)}));
  return <section className="compact-section care-tasks-section-pro">
    <div className="compact-section-head"><div><h3>{t('farmTasks')}</h3><p>{lang==='ar'?'مهام الرعاية منظمة بدون ازدحام وتظهر للعامل كإشعار.':'Care tasks are structured and sent to assigned workers.'}</p></div><span className="compact-head-icon"><CheckCircle2/></span></div>
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
          <button className="secondary-btn" onClick={()=>assign(task)}><UserPlus size={15}/>{task.assignedTo?t('reassign'):t('assign')}</button>
          <button className="secondary-btn" onClick={()=>alert(`${translateModelText(task.title || task.name,t)}\n${translateModelText(task.details || '',t)}`)}><Eye size={15}/>{t('viewDetails')}</button>
        </div>
      </article>):<div className="farm-empty-focus small"><p>{t('emptyState')}</p></div>}
    </div>
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
  const [tasks,setTasks]=useState(getGlobalTasks().filter(taskAssignedToCurrent));
  const [note,setNote]=useState('');
  const [proof,setProof]=useState('');
  const choose=e=>{const f=e.target.files?.[0]; if(!f) return; const reader=new FileReader(); reader.onload=()=>setProof(reader.result); reader.readAsDataURL(f);};
  const refresh=()=>setTasks(getGlobalTasks().filter(taskAssignedToCurrent));
  const updateTask=(task,status)=>{
    const nextAll=getGlobalTasks().map(x=>String(x.id)===String(task.id)?{...x,status,note:note || x.note,proof:proof || x.proof,completedAt:status==='completed'?new Date().toLocaleString():x.completedAt}:x);
    setGlobalTasks(nextAll);
    setTasks(nextAll.filter(taskAssignedToCurrent));
    addNotification(status==='completed'?taskToastMessage('done',lang):taskToastMessage('updated',lang), status==='completed'?'success':'warning');
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
      </article>):<div className="farm-empty-focus small"><p>{t('emptyState')}</p></div>}
    </div>
  </section>
}

function TaskBoard({embedded=false}){
  const {t,lang}=useLang();
  const [tasks,setTasks]=useState(getGlobalTasks());
  const sectors=getStore('sph_sectors',initialSectors);
  const accountWorkers=workerAccounts();
  const sectorWorkers=sectors.flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name,sectorId:s.id})));
  const workers=[...accountWorkers,...sectorWorkers.filter(w=>!accountWorkers.some(a=>String(a.email||a.id)===String(w.email||w.id)) )];
  const [form,setForm]=useState({title:'',assignedToEmail:workers[0]?.email || '',sector:sectors[0]?.name || 'Greenhouse A',priority:'Medium',details:''});
  const save=()=>{
    if(!form.title.trim()) return;
    const worker=workers.find(w=>String(w.email || w.username || w.name)===String(form.assignedToEmail)) || workers[0] || {};
    const task={id:Date.now(),title:form.title,details:form.details,status:'pending',priority:form.priority,assignedTo:worker.name || form.assignedToEmail,assignedToEmail:worker.email || worker.username || form.assignedToEmail,dueDate:'',createdAt:new Date().toLocaleString(),sector:form.sector};
    const all=[task,...tasks];
    setTasks(all); setGlobalTasks(all);
    if(task.assignedToEmail) addUserNotification(task.assignedToEmail, taskToastMessage('created',lang), 'success');
    addNotification(taskToastMessage('created',lang),'success');
    setForm({title:'',assignedToEmail:workers[0]?.email || '',sector:sectors[0]?.name || 'Greenhouse A',priority:'Medium',details:''});
  };
  const move=(id,status)=>{
    const all=tasks.map(x=>x.id===id?{...x,status:normalizeTaskStatus(status),completedAt:normalizeTaskStatus(status)==='completed'?new Date().toLocaleString():x.completedAt}:x);
    setTasks(all); setGlobalTasks(all); addNotification(taskToastMessage(normalizeTaskStatus(status)==='completed'?'done':'updated',lang),'success');
  };
  const cols=[['pending',t('pending')],['inProgress',t('inProgress')],['completed',t('completed')]];
  return <>
    {!embedded && <PageHead title={t('taskBoard')} sub={t('createTreatmentTask')} action={null}/>} 
    <Panel title={t('createTask')}>
      <div className="form-grid">
        <Field label={t('taskName')}><input value={form.title} onChange={e=>setForm({...form,title:e.target.value})}/></Field>
        <Field label={t('assignedTo')}><select value={form.assignedToEmail} onChange={e=>setForm({...form,assignedToEmail:e.target.value})}>{workers.map(w=><option key={w.email || w.id || w.name} value={w.email || w.username || w.name}>{w.name || w.email}</option>)}</select></Field>
        <Field label={t('sector')}><select value={form.sector} onChange={e=>setForm({...form,sector:e.target.value})}>{sectors.map(s=><option key={s.id} value={s.name}>{localizeValue(s.name,t)}</option>)}</select></Field>
        <Field label={t('priority')}><select value={form.priority} onChange={e=>setForm({...form,priority:e.target.value})}><option>High</option><option>Medium</option><option>Low</option></select></Field>
        <Field label={lang==='ar'?'المطلوب تنفيذه':'Task details'}><input value={form.details} onChange={e=>setForm({...form,details:e.target.value})}/></Field>
      </div>
      <button className="primary-btn" onClick={save}><Plus size={18}/>{t('createTask')}</button>
    </Panel>
    <div className="kanban">{cols.map(([key,label])=><section className="kanban-col" key={key}><h3>{label}</h3>{tasks.filter(x=>normalizeTaskStatus(x.status)===key).map(task=><article className="task-card" key={task.id}><b>{translateModelText(task.title,t)}</b><span>{t('assignedTo')}: {translateModelText(task.assignedTo,t)}</span><span>{t('sector')}: {translateModelText(task.sector,t)}</span><span>{lang==='ar'?'وقت الإنشاء':'Created'}: {task.createdAt || '-'}</span><em>{localizeValue(task.priority,t)}</em><div><button onClick={()=>move(task.id,'pending')}>{t('pending')}</button><button onClick={()=>move(task.id,'inProgress')}>{t('inProgress')}</button><button onClick={()=>move(task.id,'completed')}>{t('completed')}</button></div></article>)}</section>)}</div>
  </>
}

function AdminPanel({embedded=false}){
  const {t}=useLang();
  const role=currentRole();
  const workers=getStore('sph_sectors',initialSectors).flatMap(s=>(s.workers||[]).map(w=>({...w,sector:s.name})));
  return <>
    {!embedded && <PageHead title={t('adminPanel')} sub={role==='owner'?t('allFarmAccess'):t('assignedOnly')} action={<AccountRoleBadge/>}/>}
    {role!=='owner'?<Unauthorized/>:<>
      <div className="kpis"><Kpi icon={<Lock/>} label={t('role')} value={t('owner')}/><Kpi icon={<ShieldCheck/>} label={t('permissions')} value={`${t('canView')} / ${t('canAdd')} / ${t('canEdit')} / ${t('canDelete')}`}/><Kpi icon={<UserPlus/>} label={t('workers')} value={workers.length}/><Kpi icon={<Layers3/>} label={t('sectors')} value={getStore('sph_sectors',initialSectors).length}/></div>
      <Panel title={t('workers')}><div className="table-wrap"><table><thead><tr><th>{t('workerName')}</th><th>{t('workerRole')}</th><th>{t('sector')}</th><th>{t('permissions')}</th></tr></thead><tbody>{workers.map(w=><tr key={w.id}><td>{w.name}</td><td>{translateModelText(w.role,t)}</td><td>{translateModelText(w.sector,t)}</td><td>{t('canView')} / {t('canEdit')}</td></tr>)}</tbody></table></div></Panel>
    </>}
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

function Auth({mode, theme, setTheme}){
  const {t, lang}=useLang();
  const nav=useNavigate();
  const [authMode,setAuthMode]=useState(mode==='register'?'register':'login');
  const isLogin=authMode==='login';
  const [msg,setMsg]=useState('');
  const [email,setEmail]=useState('');
  const [password,setPassword]=useState('');
  const [fullName,setFullName]=useState('');
  const [phone,setPhone]=useState('');
  const [authLoading,setAuthLoading]=useState(false);
  useEffect(()=>{ setAuthMode(mode==='register'?'register':'login'); }, [mode]);
  const submit=async()=>{
    setAuthLoading(true);
    try{
      if(isLogin){
        const localAccount = findLocalAccount(email, password);
        if(localAccount){
          setLoggedInUser(localAccount, 'local-worker-token');
          setMsg(t('loggedIn'));
          addUserNotification(localAccount.email || localAccount.username, t('loggedIn'), 'success');
          setTimeout(()=>nav('/dashboard'),450);
          return;
        }
      }
      let data=null;
      if(isLogin){
        const res=await authAPI.login({email,password});
        data=res.data;
      }else{
        const res=await authAPI.register({name:fullName,fullName,email,password,phone});
        data=res.data;
      }
      const token=data?.token || data?.accessToken;
      const user=data?.user || data?.data?.user || {email,name:fullName,phone};
      if(token) localStorage.setItem('ecosense_token', token);
      setLoggedInUser(user, token || 'local-demo-token');
      setMsg(isLogin?t('loggedIn'):t('accountCreated'));
      setTimeout(()=>nav('/dashboard'),450);
    }catch(err){
      /* backend auth unavailable: local auth fallback */
      const fallbackRole = isLogin ? normalizeRole(null, email) : 'owner';
      setLoggedInUser({email, name:fullName || (email ? String(email).split('@')[0].replace(/[._-]+/g,' ') : 'Mahmoud'), phone, role:fallbackRole}, 'local-demo-token');
      setMsg(isLogin?t('loggedIn'):t('accountCreated'));
      setTimeout(()=>nav('/dashboard'),450);
    }finally{ setAuthLoading(false); }
  };
  const switchMode=(next)=>{ setAuthMode(next); setMsg(''); window.history.replaceState(null,'',next==='login'?'/login':'/register'); };
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
    <section className="auth-card compact-auth-card">
      <div className="auth-tools"><LanguageTheme theme={theme} setTheme={setTheme}/></div>
      <div className="auth-switch-tabs">
        <button className={isLogin?'active':''} onClick={()=>switchMode('login')}>{t('login')}</button>
        <button className={!isLogin?'active':''} onClick={()=>switchMode('register')}>{t('signup')}</button>
      </div>
      <h2>{isLogin?t('login'):t('createAccount')}</h2>
      {!isLogin&&<div className="form-grid compact-auth-fields"><Field label={t('fullName')}><input value={fullName} onChange={e=>setFullName(e.target.value)} required/></Field><Field label={t('phone')}><input value={phone} onChange={e=>setPhone(e.target.value)} required/></Field></div>}
      <Field label={t('email')}><input type="email" value={email} onChange={e=>setEmail(e.target.value)} required/></Field>
      <Field label={t('password')}><input type="password" value={password} onChange={e=>setPassword(e.target.value)} required/></Field>
      {!isLogin&&<Field label={t('confirmPassword')}><input type="password" required/></Field>}
      <div className="auth-row"><label><input type="checkbox"/> {t('remember')}</label>{isLogin&&<a>{t('forgot')}</a>}</div>
      <div className="auth-permission-note pro-permission-note">
        <span><ShieldCheck size={18}/></span>
        <div>
          <b>{lang==='ar'?'حسابك يحدد صلاحياتك تلقائيًا':'Your account controls your access'}</b>
          <small>{lang==='ar'?'المالك يرى كل الأدوات، والعامل يرى المهام والصفحات المخصصة له فقط.':'Owners get full access; workers only see assigned tools and tasks.'}</small>
        </div>
      </div>
      <button className="primary-btn wide" onClick={submit} disabled={authLoading}>{authLoading?t('loadingData'):(isLogin?t('login'):t('createAccount'))}</button>
      <button className="google-btn compact" onClick={submit}><Globe2 size={18}/>{t('continueGoogle')}</button>
      {msg && <p className="success-msg">{msg}</p>}
      <p className="switch-auth">
        {isLogin ? t('noAccount') : t('alreadyHave')}
        <button type="button" onClick={()=>switchMode(isLogin?'register':'login')}>{isLogin?t('signup'):t('login')}</button>
      </p>
    </section>
  </main>
}
function tone(s=''){ if(String(s).includes('High')) return 'red'; if(String(s).includes('Moderate')) return 'amber'; return 'green'; }
function labelStatus(s,t){ const v=String(s||''); if(!v||v==='-') return '-'; if(v.includes('High')||v.includes('مرتفع')||v.includes('خطر')||v.toLowerCase().includes('critical')) return t('high'); if(v.includes('Moderate')||v.includes('متوسط')) return t('moderate'); if(v.includes('Healthy')||v.includes('سليم')||v.includes('صحي')) return t('healthy'); if(v.includes('No Image')||v.includes('لا توجد صورة')) return t('noImage'); return translateModelText(v,t); }
function download(name,text,type='application/json'){ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type})); a.download=name; a.click(); }

export default App;
