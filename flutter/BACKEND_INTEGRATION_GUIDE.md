# Backend Integration Guide for Ecosense

## Overview
This guide walks you through integrating a real backend API with your Ecosense Flutter app. The app is already structured to receive data from an API.

---

## Step 1: Choose Your Backend Technology

### Option A: Node.js + Express (Recommended for Quick Start)
```bash
npm init -y
npm install express cors dotenv jsonwebtoken bcryptjs
```

### Option B: Python + Flask
```bash
pip install flask flask-cors flask-jwt-extended
```

### Option C: Firebase (Quickest)
- Go to console.firebase.google.com
- Create new project
- Enable Authentication and Firestore
- Download service account key

---

## Step 2: Create Authentication API (Node.js Example)

### Install Dependencies
```bash
npm install express cors dotenv jsonwebtoken bcryptjs mysql2
```

### Create .env file
```
PORT=3000
JWT_SECRET=your_secret_key_here
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=password
DB_NAME=ecosense
```

### Create server.js
```javascript
const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const mysql = require('mysql2/promise');

const app = express();
app.use(cors());
app.use(express.json());

const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
});

// Login endpoint
app.post('/api/v1/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    const connection = await pool.getConnection();
    const [users] = await connection.query(
      'SELECT * FROM users WHERE email = ?',
      [email]
    );
    
    if (users.length === 0) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    const user = users[0];
    const validPassword = await bcrypt.compare(password, user.password);
    
    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    const token = jwt.sign(
      { id: user.id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );
    
    connection.release();
    
    res.json({
      success: true,
      token,
      user: { id: user.id, name: user.name, email: user.email }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Signup endpoint
app.post('/api/v1/auth/signup', async (req, res) => {
  try {
    const { name, email, password } = req.body;
    
    const connection = await pool.getConnection();
    const hashedPassword = await bcrypt.hash(password, 10);
    
    await connection.query(
      'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
      [name, email, hashedPassword]
    );
    
    connection.release();
    
    res.json({ success: true, message: 'User created' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(process.env.PORT, () => {
  console.log(`Server running on port ${process.env.PORT}`);
});
```

---

## Step 3: Create Database Schema

### MySQL Setup
```sql
-- Create database
CREATE DATABASE ecosense;
USE ecosense;

-- Users table
CREATE TABLE users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Plants table
CREATE TABLE plants (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT NOT NULL,
  name VARCHAR(255) NOT NULL,
  species VARCHAR(255),
  health_status VARCHAR(50),
  health_percentage DECIMAL(5,2),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Sensors table
CREATE TABLE sensors (
  id INT PRIMARY KEY AUTO_INCREMENT,
  plant_id INT NOT NULL,
  sensor_type VARCHAR(50),
  reading DECIMAL(10,2),
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (plant_id) REFERENCES plants(id)
);

-- Predictions table
CREATE TABLE predictions (
  id INT PRIMARY KEY AUTO_INCREMENT,
  plant_id INT NOT NULL,
  prediction_type VARCHAR(50),
  confidence DECIMAL(5,2),
  recommendation TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (plant_id) REFERENCES plants(id)
);
```

---

## Step 4: Update Flutter App Configuration

### Update lib/config/constants.dart
```dart
class ApiConstants {
  // API Base URL - Change this to your backend
  static const String baseUrl = 'http://your-backend-url:3000/api/v1';
  
  // Auth endpoints
  static const String login = '$baseUrl/auth/login';
  static const String signup = '$baseUrl/auth/signup';
  static const String refreshToken = '$baseUrl/auth/refresh';
  static const String logout = '$baseUrl/auth/logout';
  
  // Plant endpoints
  static const String plants = '$baseUrl/plants';
  static const String plant(int id) => '$baseUrl/plants/$id';
  
  // Sensor endpoints
  static const String sensors = '$baseUrl/sensors';
  
  // Prediction endpoints
  static const String predictions = '$baseUrl/predictions';
  
  // Timeout duration
  static const Duration timeout = Duration(seconds: 30);
}
```

### Update lib/core/network/api_client.dart
```dart
import 'package:dio/dio.dart';

class ApiClient {
  late Dio _dio;
  
  ApiClient() {
    _dio = Dio(
      BaseOptions(
        baseUrl: ApiConstants.baseUrl,
        connectTimeout: ApiConstants.timeout,
        receiveTimeout: ApiConstants.timeout,
        headers: {
          'Content-Type': 'application/json',
        },
      ),
    );
    
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) {
          // Add token to headers
          String? token = _getToken();
          if (token != null) {
            options.headers['Authorization'] = 'Bearer $token';
          }
          return handler.next(options);
        },
        onError: (error, handler) {
          if (error.response?.statusCode == 401) {
            // Handle token refresh
            return _refreshToken().then((_) {
              return _retryRequest(error.requestOptions);
            });
          }
          return handler.next(error);
        },
      ),
    );
  }
  
  String? _getToken() {
    // Get token from Hive storage
    return null; // Implement token retrieval
  }
  
  Future<void> _refreshToken() async {
    // Implement token refresh logic
  }
  
  Future<Response> _retryRequest(RequestOptions requestOptions) {
    return _dio.request(
      requestOptions.path,
      data: requestOptions.data,
      queryParameters: requestOptions.queryParameters,
      options: Options(method: requestOptions.method),
    );
  }
  
  Future<dynamic> get(String endpoint) async {
    try {
      final response = await _dio.get(endpoint);
      return response.data;
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<dynamic> post(String endpoint, {required dynamic data}) async {
    try {
      final response = await _dio.post(endpoint, data: data);
      return response.data;
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<dynamic> put(String endpoint, {required dynamic data}) async {
    try {
      final response = await _dio.put(endpoint, data: data);
      return response.data;
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<dynamic> delete(String endpoint) async {
    try {
      final response = await _dio.delete(endpoint);
      return response.data;
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  dynamic _handleError(dynamic error) {
    if (error is DioException) {
      if (error.response != null) {
        return Exception(
          error.response?.data['error'] ?? 'An error occurred'
        );
      }
    }
    return Exception('Network error');
  }
}
```

---

## Step 5: Implement API Calls in Repositories

### Example: UserRepository
```dart
import '../core/network/api_client.dart';
import '../models/user_model.dart';

class UserRepository {
  final ApiClient _apiClient;
  
  UserRepository(this._apiClient);
  
  Future<UserModel> login({
    required String email,
    required String password,
  }) async {
    try {
      final response = await _apiClient.post(
        '/auth/login',
        data: {
          'email': email,
          'password': password,
        },
      );
      
      // Save token to Hive
      // await _saveToken(response['token']);
      
      return UserModel.fromJson(response['user']);
    } catch (e) {
      throw Exception('Login failed: $e');
    }
  }
  
  Future<UserModel> signup({
    required String name,
    required String email,
    required String password,
  }) async {
    try {
      final response = await _apiClient.post(
        '/auth/signup',
        data: {
          'name': name,
          'email': email,
          'password': password,
        },
      );
      
      return UserModel.fromJson(response['user']);
    } catch (e) {
      throw Exception('Signup failed: $e');
    }
  }
  
  Future<void> logout() async {
    try {
      await _apiClient.post('/auth/logout', data: {});
      // Clear stored token
      // await _clearToken();
    } catch (e) {
      throw Exception('Logout failed: $e');
    }
  }
}
```

### Example: PlantRepository
```dart
class PlantRepository {
  final ApiClient _apiClient;
  
  PlantRepository(this._apiClient);
  
  Future<List<PlantModel>> getPlants() async {
    try {
      final response = await _apiClient.get('/plants');
      return (response as List)
          .map((p) => PlantModel.fromJson(p))
          .toList();
    } catch (e) {
      throw Exception('Failed to fetch plants: $e');
    }
  }
  
  Future<PlantModel> getPlantById(int id) async {
    try {
      final response = await _apiClient.get('/plants/$id');
      return PlantModel.fromJson(response);
    } catch (e) {
      throw Exception('Failed to fetch plant: $e');
    }
  }
  
  Future<PlantModel> createPlant({required PlantModel plant}) async {
    try {
      final response = await _apiClient.post(
        '/plants',
        data: plant.toJson(),
      );
      return PlantModel.fromJson(response);
    } catch (e) {
      throw Exception('Failed to create plant: $e');
    }
  }
  
  Future<PlantModel> updatePlant({
    required int id,
    required PlantModel plant,
  }) async {
    try {
      final response = await _apiClient.put(
        '/plants/$id',
        data: plant.toJson(),
      );
      return PlantModel.fromJson(response);
    } catch (e) {
      throw Exception('Failed to update plant: $e');
    }
  }
  
  Future<void> deletePlant(int id) async {
    try {
      await _apiClient.delete('/plants/$id');
    } catch (e) {
      throw Exception('Failed to delete plant: $e');
    }
  }
}
```

---

## Step 6: Update Providers to Use Repositories

### Example: AuthProvider
```dart
class AuthProvider extends ChangeNotifier {
  UserModel? _user;
  bool _isLoading = false;
  String? _error;
  
  final UserRepository _userRepository;
  
  AuthProvider(this._userRepository);
  
  UserModel? get user => _user;
  bool get isLoading => _isLoading;
  String? get error => _error;
  bool get isAuthenticated => _user != null;
  
  Future<bool> login({
    required String email,
    required String password,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();
    
    try {
      _user = await _userRepository.login(
        email: email,
        password: password,
      );
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _user = null;
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
  
  Future<bool> signup({
    required String name,
    required String email,
    required String password,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();
    
    try {
      _user = await _userRepository.signup(
        name: name,
        email: email,
        password: password,
      );
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _user = null;
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
  
  void logout() {
    _user = null;
    _error = null;
    notifyListeners();
  }
}
```

---

## Step 7: Test the Integration

### Test Login Flow
```dart
// In main.dart or test file
void testLoginFlow() async {
  final apiClient = ApiClient();
  final userRepository = UserRepository(apiClient);
  final authProvider = AuthProvider(userRepository);
  
  final success = await authProvider.login(
    email: 'test@example.com',
    password: 'password123',
  );
  
  if (success) {
    print('Login successful: ${authProvider.user?.name}');
  } else {
    print('Login failed: ${authProvider.error}');
  }
}
```

---

## Step 8: Deploy Backend

### Local Testing
```bash
# Terminal 1: Run backend
node server.js

# Terminal 2: Run Flutter app
flutter run
```

### Deploy to Cloud
Options:
1. **Heroku** - Easy for Node.js
2. **AWS** - Scalable option
3. **Firebase** - Quick serverless
4. **DigitalOcean** - Affordable VPS
5. **Railway/Render** - Modern platforms

### Update API URL After Deployment
```dart
// In lib/config/constants.dart
static const String baseUrl = 'https://your-deployed-backend.com/api/v1';
```

---

## Step 9: Add Token Management

### Create TokenService
```dart
import 'package:hive/hive.dart';

class TokenService {
  static const String _tokenBox = 'tokens';
  static const String _tokenKey = 'auth_token';
  
  static Future<void> saveToken(String token) async {
    final box = await Hive.openBox(_tokenBox);
    await box.put(_tokenKey, token);
  }
  
  static Future<String?> getToken() async {
    final box = await Hive.openBox(_tokenBox);
    return box.get(_tokenKey);
  }
  
  static Future<void> clearToken() async {
    final box = await Hive.openBox(_tokenBox);
    await box.delete(_tokenKey);
  }
  
  static Future<bool> hasToken() async {
    final token = await getToken();
    return token != null && token.isNotEmpty;
  }
}
```

---

## Checklist

- [ ] Backend server created
- [ ] Database schema set up
- [ ] Authentication endpoints working
- [ ] API base URL configured in Flutter
- [ ] Repositories implementing API calls
- [ ] Providers using repositories
- [ ] Token management implemented
- [ ] Login flow tested
- [ ] Logout flow tested
- [ ] Plant data API endpoints created
- [ ] Sensor data API endpoints created
- [ ] Error handling implemented
- [ ] Backend deployed to cloud
- [ ] Production API URL configured
- [ ] Testing on real device
- [ ] Security audit completed

---

## Troubleshooting

### CORS Issues
Add to Node.js:
```javascript
app.use(cors({
  origin: '*',
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  credentials: true,
}));
```

### Connection Refused
- Check if backend is running
- Verify correct IP/URL
- Check firewall settings
- Use adb reverse for Android emulator:
  ```bash
  adb reverse tcp:3000 tcp:3000
  ```

### Token Expiration
Implement refresh token logic in interceptors:
```dart
if (error.response?.statusCode == 401) {
  // Refresh token and retry request
  await refreshToken();
  return _retryRequest(error.requestOptions);
}
```

---

## Next Steps

1. Choose backend technology
2. Create database and schema
3. Build authentication API
4. Update Flutter configuration
5. Implement repositories
6. Test integration
7. Deploy backend
8. Update app with production URL
9. Final testing on devices
10. Release to app stores

---

**Good luck with your backend integration!**
