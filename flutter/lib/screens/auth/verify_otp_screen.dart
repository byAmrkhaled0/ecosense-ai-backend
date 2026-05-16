import 'package:animate_do/animate_do.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../main/main_layout.dart';
import 'login_screen.dart';

class VerifyOtpScreen extends StatefulWidget {
  const VerifyOtpScreen({super.key, this.email});

  final String? email;

  @override
  State<VerifyOtpScreen> createState() => _VerifyOtpScreenState();
}

class _VerifyOtpScreenState extends State<VerifyOtpScreen> {
  late final TextEditingController _codeController;

  @override
  void initState() {
    super.initState();
    _codeController = TextEditingController();
  }

  @override
  void dispose() {
    _codeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Verify account',
          style: GoogleFonts.inter(fontWeight: FontWeight.w600),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              FadeInDown(
                child: Text(
                  'Enter OTP',
                  style: GoogleFonts.inter(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              FadeInDown(
                delay: const Duration(milliseconds: 150),
                child: Text(
                  widget.email != null
                      ? 'We sent a code to ${widget.email}'
                      : 'Enter the verification code from your email',
                  style: GoogleFonts.inter(fontSize: 14, color: Colors.grey),
                ),
              ),
              const SizedBox(height: 28),
              FadeInUp(
                delay: const Duration(milliseconds: 200),
                child: TextField(
                  controller: _codeController,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    labelText: 'Code',
                    prefixIcon: Icon(Icons.pin_outlined),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Consumer<AuthProvider>(
                builder: (context, auth, _) {
                  if (auth.error != null) {
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Text(
                        auth.error!,
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.error,
                        ),
                      ),
                    );
                  }
                  return const SizedBox.shrink();
                },
              ),
              FadeInUp(
                delay: const Duration(milliseconds: 280),
                child: Consumer<AuthProvider>(
                  builder: (context, auth, _) {
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        ElevatedButton(
                          onPressed: auth.isLoading
                              ? null
                              : () async {
                                  final ok = await auth.verifyOtp(
                                    _codeController.text,
                                  );
                                  if (!context.mounted) return;
                                  if (ok) {
                                    Navigator.of(context).pushAndRemoveUntil(
                                      MaterialPageRoute(
                                        builder: (_) => const LoginScreen(),
                                      ),
                                      (route) => false,
                                    );
                                  }
                                },
                          child: auth.isLoading
                              ? const SizedBox(
                                  height: 20,
                                  width: 20,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              : Text(
                                  'Verify',
                                  style: GoogleFonts.inter(
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                        ),
                        const SizedBox(height: 16),
                        TextButton(
                          onPressed: () {
                            auth.skipAuth();
                            Navigator.of(context).pushAndRemoveUntil(
                              MaterialPageRoute(
                                builder: (_) => const MainLayout(),
                              ),
                              (route) => false,
                            );
                          },
                          child: Text(
                            'Skip (Test Mode)',
                            style: GoogleFonts.inter(
                              color: Colors.grey,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
