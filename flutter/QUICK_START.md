# Ecosense - Quick Start Guide

## Project Overview
Ecosense is an AI-powered plant health monitoring system built with Flutter, featuring real-time sensor integration, health predictions, and intelligent recommendations.

---

## Current Features

### ✅ Implemented
- **Authentication:** Login/Signup with validation
- **Logout Confirmation:** Safe logout with confirmation dialog
- **Dark Mode:** Full dark/light theme support
- **Home Dashboard:** Plant statistics and health overview
- **Sidebar Navigation:** Complete menu with all features
- **Profile Section:** User information display
- **Responsive Design:** Works on mobile and tablet

### 🔄 In Development (Next Tasks)
- API integration
- Real-time sensor data
- AI predictions
- Chat assistant
- Data analytics

---

## Project Structure

```
lib/
├── main.dart                          # App entry point
├── config/
│   ├── theme.dart                    # Theme colors and styles
│   ├── constants.dart                # App constants
│   └── app_config.dart               # Configuration
├── core/
│   ├── network/
│   │   ├── api_client.dart          # API client (Ready for integration)
│   │   └── interceptors.dart        # Request/response interceptors
│   └── utils/
│       ├── logger.dart              # Logging utility
│       └── validators.dart          # Input validation
├── models/                           # Data models
├── providers/                        # State management (Provider)
├── repositories/                     # Data layer (Ready for API)
├── screens/
│   ├── splash_screen.dart          # App splash
│   ├── auth/
│   │   ├── login_screen.dart       # ✅ Implemented
│   │   └── signup_screen.dart      # ✅ Implemented
│   └── main/
│       ├── main_layout.dart        # ✅ Main app layout
│       ├── home_screen.dart        # ✅ Dashboard
│       ├── dataset_screen.dart     # Empty (Ready)
│       ├── analysis_screen.dart    # Empty (Ready)
│       ├── prediction_screen.dart  # Empty (Ready)
│       └── about_screen.dart       # Empty (Ready)
├── widgets/
│   ├── common/
│   │   ├── header.dart            # ✅ Top app bar
│   │   ├── sidebar.dart           # ✅ Navigation drawer
│   │   └── bottom_nav.dart        # ✅ Bottom navigation
│   ├── cards/                      # Card components
│   └── dialogs/
│       ├── logout_dialog.dart     # ✅ Logout confirmation
│       └── menu_navigation.dart   # Navigation data
```

---

## How to Use the App

### 1. Login/Signup
- Email: `test@example.com` (any email works in dev mode)
- Password: `password` (any password works in dev mode)
- After login, you enter the main app

### 2. Navigate Using Sidebar
- Open drawer with menu icon
- Select different sections
- Close drawer by selecting a menu item

### 3. Dark Mode Toggle
- Click the sun/moon icon in the header
- App theme toggles between light and dark

### 4. Logout
- Open sidebar
- Tap "Logout" at the bottom
- Confirmation dialog appears
- Confirm to logout and return to login screen

---

## Running the App

```bash
# Clean build
flutter clean

# Get dependencies
flutter pub get

# Run in debug mode
flutter run

# Run in release mode
flutter run --release

# Run on specific device
flutter run -d <device_id>
```

---

## Setting Up for API Integration

### Step 1: Create Backend Server
Choose one:
- **Node.js + Express** (Recommended for quick start)
- **Python + Flask/Django**
- **Go + Gin**
- **Ruby on Rails**

### Step 2: Configure API Base URL
Edit `lib/config/constants.dart`:
```dart
const String API_BASE_URL = 'http://your-backend-url/api/v1';
```

### Step 3: Implement API Endpoints
Required endpoints:
```
POST   /auth/login          # Login
POST   /auth/signup         # Register
POST   /auth/refresh        # Refresh token
GET    /plants              # Get all plants
GET    /plants/:id          # Get plant details
POST   /plants              # Create plant
PUT    /plants/:id          # Update plant
DELETE /plants/:id          # Delete plant
GET    /sensors             # Get sensor data
GET    /predictions         # Get AI predictions
```

### Step 4: Update Repositories
Replace mock calls in repositories with actual API calls:
```dart
// Example: lib/repositories/plant_repository.dart
Future<List<PlantModel>> getPlants() async {
  try {
    final response = await apiClient.get('/plants');
    return (response as List)
        .map((p) => PlantModel.fromJson(p))
        .toList();
  } catch (e) {
    throw Exception('Failed to fetch plants: $e');
  }
}
```

---

## Key Files to Modify First

1. **lib/config/app_config.dart** - API configuration
2. **lib/core/network/api_client.dart** - API base setup
3. **lib/repositories/*.dart** - Implement actual API calls
4. **lib/providers/*.dart** - Update to use repositories

---

## Important: Logout Implementation

The logout flow is already set up with proper safety checks:

```dart
// In logout dialog
if (context.mounted) {
  Provider.of<AuthProvider>(context, listen: false).logout();
  Navigator.of(context).pushNamedAndRemoveUntil(
    '/login',
    (route) => false,
  );
}
```

This ensures:
- ✅ BuildContext is still valid after async operations
- ✅ Safe navigation back to login
- ✅ Clears navigation stack
- ✅ Removes all routes except login

---

## Common Tasks

### Add a New Screen
1. Create file in `lib/screens/main/`
2. Create StatefulWidget or StatelessWidget
3. Add to sidebar menu in `lib/widgets/common/sidebar.dart`
4. Add route navigation if needed

### Add a New Provider
1. Create file in `lib/providers/`
2. Extend ChangeNotifier
3. Define methods for state changes
4. Add to providers list in `main.dart`

### Add a New API Endpoint
1. Create method in appropriate repository
2. Call from provider
3. Update UI to use provider data
4. Add error handling

---

## Error Handling

The app has built-in error handling:
```dart
// In providers
try {
  // API call
  notifyListeners();
} catch (e) {
  _error = e.toString();
  notifyListeners();
}
```

Display errors in UI:
```dart
if (provider.error != null) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(content: Text(provider.error!)),
  );
}
```

---

## Testing

### Test Login Flow
1. Run app
2. Enter any email and password
3. Click Login (2-second delay in dev mode)
4. Should navigate to home screen

### Test Logout
1. Open sidebar
2. Click Logout
3. Dialog appears asking for confirmation
4. Click "Yes, Logout"
5. Should return to login screen

### Test Dark Mode
1. Click sun/moon icon in header
2. Theme should toggle
3. All screens should adapt

---

## Debugging

Enable debug prints:
```dart
// In any file
console.log("[v0] Debug message:", variable);
```

Check logs:
- Android: `flutter logs`
- iOS: `flutter logs`
- Chrome DevTools: `flutter run -d chrome`

---

## Next Steps

1. ✅ Set up backend server
2. ✅ Create authentication API endpoints
3. ✅ Update API base URL in config
4. ✅ Implement plant API integration
5. ✅ Test with real data
6. ✅ Add sensor data API
7. ✅ Implement WebSocket for real-time updates
8. ✅ Add AI prediction API
9. ✅ Set up push notifications
10. ✅ Deploy to app stores

---

## Support & Resources

- **Flutter Docs:** https://flutter.dev/docs
- **Provider Documentation:** https://pub.dev/packages/provider
- **Dio Package:** https://pub.dev/packages/dio
- **Hive Documentation:** https://docs.hivedb.dev/

---

## Performance Tips

1. Use `const` for widgets where possible
2. Implement image caching
3. Use proper list pagination
4. Minimize rebuilds with `Consumer`
5. Cache API responses in Hive
6. Use FutureBuilder for async data

---

## License
MIT License - Feel free to use this project as a template

**Last Updated:** January 2026
