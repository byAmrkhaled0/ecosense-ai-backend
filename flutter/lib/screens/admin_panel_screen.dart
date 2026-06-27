import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../providers/auth_provider.dart';

class AdminPanelScreen extends StatelessWidget {
  const AdminPanelScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final user = auth.user;

    final canAccess =
        user?.userType == 'owner' || user?.userType == 'admin';

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Admin Panel',
          style: GoogleFonts.inter(fontWeight: FontWeight.w700),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: canAccess
            ? const Center(child: Text('Admin content (placeholder)'))
            : Center(
                child: Text(
                  'You don\'t have access to Admin Panel',
                  style: GoogleFonts.inter(fontSize: 16, color: Colors.grey),
                ),
              ),
      ),
    );
  }
}

