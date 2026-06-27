import 'package:animate_do/animate_do.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/auth_provider.dart';
import 'auth/login_screen.dart';
import 'main/main_layout.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    await Future.delayed(const Duration(milliseconds: 800));
    if (!mounted) return;

    final auth = context.read<AuthProvider>();
    await auth.restoreSession();

    if (!mounted) return;

    final Widget next =
        auth.isAuthenticated ? const MainLayout() : const LoginScreen();

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => next),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF121212) : Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ZoomIn(
              duration: const Duration(seconds: 1),
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: const Color(0xFF2E7D32).withValues(alpha: 0.1),
                ),
                child: const Icon(
                  Icons.eco,
                  size: 100,
                  color: Color(0xFF2E7D32),
                ),
              ),
            ),
            const SizedBox(height: 40),
            FadeInUp(
              delay: const Duration(seconds: 1),
              child: const CircularProgressIndicator(
                color: Color(0xFF2E7D32),
              ),
            ),
            const SizedBox(height: 20),
            FadeInUp(
              delay: const Duration(seconds: 2),
              child: Text(
                'Ecosense',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: isDark ? Colors.white : Colors.black,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
