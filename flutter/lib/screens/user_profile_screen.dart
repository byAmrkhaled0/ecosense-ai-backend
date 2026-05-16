import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../providers/auth_provider.dart';

class UserProfileScreen extends StatelessWidget {
  const UserProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final user = auth.user;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'User Profile',
          style: GoogleFonts.inter(fontWeight: FontWeight.w700),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: user == null
            ? const Center(child: Text('No user data'))
            : ListView(
                children: [
                  Text(
                    user.name,
                    style: GoogleFonts.inter(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    user.email,
                    style: GoogleFonts.inter(fontSize: 14, color: Colors.grey),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Type: ${user.userType}',
                    style: GoogleFonts.inter(fontSize: 14),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Created: ${user.createdAt.toLocal()}',
                    style: GoogleFonts.inter(fontSize: 14),
                  ),
                ],
              ),
      ),
    );
  }
}

