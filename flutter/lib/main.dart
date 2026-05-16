import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:provider/provider.dart';

import 'config/constants.dart';
import 'config/theme.dart';
import 'core/network/api_client.dart';
import 'core/services/token_storage_service.dart';
import 'providers/ai_chat_provider.dart';
import 'providers/auth_provider.dart';
import 'providers/dataset_provider.dart';
import 'providers/home_provider.dart';
import 'providers/plant_provider.dart';
import 'providers/prediction_provider.dart';
import 'providers/sensor_provider.dart';
import 'providers/theme_provider.dart';
import 'providers/language_provider.dart';
import 'repositories/ai_chat_repository.dart';
import 'repositories/auth_repository.dart';
import 'screens/splash_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final apiClient = ApiClient(baseUrl: AppConstants.apiBaseUrl);
  final tokenStorage = TokenStorageService();
  final authRepository = AuthRepository(
    apiClient: apiClient,
    tokenStorage: tokenStorage,
  );
  final aiChatRepository = AIChatRepository(apiClient: apiClient);

  final savedToken = await tokenStorage.readToken();
  if (savedToken != null && savedToken.isNotEmpty) {
    apiClient.setAuthToken(savedToken);
  }

  runApp(
    EcosenseApp(
      apiClient: apiClient,
      tokenStorage: tokenStorage,
      authRepository: authRepository,
      aiChatRepository: aiChatRepository,
    ),
  );
}

class EcosenseApp extends StatelessWidget {
  const EcosenseApp({
    required this.apiClient,
    required this.tokenStorage,
    required this.authRepository,
    required this.aiChatRepository,
    super.key,
  });

  final ApiClient apiClient;
  final TokenStorageService tokenStorage;
  final AuthRepository authRepository;
  final AIChatRepository aiChatRepository;

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<ApiClient>.value(value: apiClient),
        Provider<TokenStorageService>.value(value: tokenStorage),
        Provider<AuthRepository>.value(value: authRepository),
        Provider<AIChatRepository>.value(value: aiChatRepository),
        ChangeNotifierProvider(
          create: (_) => AuthProvider(
            authRepository: authRepository,
            tokenStorage: tokenStorage,
            apiClient: apiClient,
          ),
        ),
        ChangeNotifierProvider(create: (_) => ThemeProvider()),
        ChangeNotifierProvider(create: (_) => HomeProvider()),
        ChangeNotifierProvider(create: (_) => PlantProvider()),
        ChangeNotifierProvider(create: (_) => SensorProvider()),
        ChangeNotifierProvider(create: (_) => PredictionProvider()),
        ChangeNotifierProvider(create: (_) => DatasetProvider()),
        ChangeNotifierProvider(
          create: (_) => AIChatProvider(aiChatRepository: aiChatRepository),
        ),
        ChangeNotifierProvider(create: (_) => LanguageProvider()),
      ],
      child: Consumer2<ThemeProvider, LanguageProvider>(
        builder: (context, themeProvider, languageProvider, _) {
          return MaterialApp(
            title: 'Ecosense',
            debugShowCheckedModeBanner: false,
            locale: languageProvider.locale,
            localizationsDelegates: const [
              GlobalMaterialLocalizations.delegate,
              GlobalWidgetsLocalizations.delegate,
              GlobalCupertinoLocalizations.delegate,
            ],
            supportedLocales: const [
              Locale('en'),
              Locale('ar'),
            ],
            theme: AppTheme.lightTheme,
            darkTheme: AppTheme.darkTheme,
            themeMode: themeProvider.themeMode,
            home: const SplashScreen(),
          );
        },
      ),
    );
  }
}
