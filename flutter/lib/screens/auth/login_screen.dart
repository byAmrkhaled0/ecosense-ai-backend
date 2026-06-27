import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:animate_do/animate_do.dart';
import 'package:provider/provider.dart';
import '../../providers/auth_provider.dart';
import '../../providers/language_provider.dart';
import '../../core/localization/app_translations.dart';
import '../main/main_layout.dart';
import 'signup_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  late TextEditingController _emailController;
  late TextEditingController _passwordController;
  bool _obscurePassword = true;

  @override
  void initState() {
    super.initState();
    _emailController = TextEditingController();
    _passwordController = TextEditingController();
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 40),
              FadeInDown(
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: const Color(0xFF2E7D32).withValues(alpha: 0.1),
                  ),
                  child: const Icon(
                    Icons.eco,
                    size: 80,
                    color: Color(0xFF2E7D32),
                  ),
                ),
              ),
              const SizedBox(height: 40),
              FadeInLeft(
                child: Text(
                  AppTranslations.get(context, 'welcome_back'),
                  style: GoogleFonts.inter(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 8),
              FadeInLeft(
                delay: const Duration(milliseconds: 200),
                child: Text(
                  AppTranslations.get(context, 'sign_in_continue'),
                  style: GoogleFonts.inter(
                    fontSize: 14,
                    color: Colors.grey,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 40),
              FadeInUp(
                delay: const Duration(milliseconds: 300),
                child: TextField(
                  controller: _emailController,
                  decoration: InputDecoration(
                    labelText: AppTranslations.get(context, 'email'),
                    prefixIcon: const Icon(Icons.email_outlined),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              FadeInUp(
                delay: const Duration(milliseconds: 350),
                child: TextField(
                  controller: _passwordController,
                  obscureText: _obscurePassword,
                  decoration: InputDecoration(
                    labelText: AppTranslations.get(context, 'password'),
                    prefixIcon: const Icon(Icons.lock_outline),
                    suffixIcon: IconButton(
                      icon: Icon(
                        _obscurePassword
                            ? Icons.visibility_off
                            : Icons.visibility,
                      ),
                      onPressed: () {
                        setState(() {
                          _obscurePassword = !_obscurePassword;
                        });
                      },
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              Consumer<AuthProvider>(
                builder: (context, authProvider, _) {
                  if (authProvider.error == null) {
                    return const SizedBox.shrink();
                  }
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Text(
                      authProvider.error!,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.error,
                        fontSize: 13,
                      ),
                    ),
                  );
                },
              ),
              FadeInUp(
                delay: const Duration(milliseconds: 400),
                child: Consumer<AuthProvider>(
                  builder: (context, authProvider, _) {
                    return ElevatedButton(
                      onPressed: authProvider.isLoading
                          ? null
                          : () async {
                              await authProvider.login(
                                email: _emailController.text,
                                password: _passwordController.text,
                              );
                              if (!mounted) return;
                              if (authProvider.isAuthenticated) {
                                // ignore: use_build_context_synchronously
                                Navigator.of(context).pushReplacement(
                                  MaterialPageRoute(
                                    builder: (context) => const MainLayout(),
                                  ),
                                );
                              }
                            },
                      child: authProvider.isLoading
                          ? const SizedBox(
                              height: 20,
                              width: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                              ),
                            )
                          : Text(
                              AppTranslations.get(context, 'login'),
                              style: GoogleFonts.inter(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 24),
              FadeInUp(
                delay: const Duration(milliseconds: 500),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      AppTranslations.get(context, 'no_account'),
                      style: GoogleFonts.inter(),
                    ),
                    TextButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => const SignupScreen(),
                          ),
                        );
                      },
                      child: Text(
                        AppTranslations.get(context, 'sign_up'),
                        style: GoogleFonts.inter(
                          fontWeight: FontWeight.bold,
                          color: const Color(0xFF2E7D32),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              FadeInUp(
                delay: const Duration(milliseconds: 600),
                child: TextButton(
                  onPressed: () {
                    context.read<AuthProvider>().skipAuth();
                    Navigator.of(context).pushReplacement(
                      MaterialPageRoute(
                        builder: (context) => const MainLayout(),
                      ),
                    );
                  },
                  child: Text(
                    AppTranslations.get(context, 'skip_test'),
                    style: GoogleFonts.inter(
                      color: Colors.grey,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        Positioned(
          top: 16,
          right: 16,
          child: PopupMenuButton<String>(
            icon: const Icon(Icons.language),
            tooltip: 'Change Language',
            onSelected: (String value) {
              context.read<LanguageProvider>().changeLanguage(value);
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
              const PopupMenuItem<String>(
                value: 'ar',
                child: Text('العربية'),
              ),
              const PopupMenuItem<String>(
                value: 'en',
                child: Text('English'),
              ),
            ],
          ),
        ),
        ],
      ),
      ),
    );
  }
}
