import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/auth_provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../core/localization/app_translations.dart';
// ignore: unused_import
import '../dialogs/logout_dialog.dart';
// ignore: unused_import
import '../dialogs/menu_navigation.dart';
import '../../screens/auth/login_screen.dart';
import '../../screens/ai_chat_screen.dart';
import '../../screens/sensor_details_screen.dart';
import '../../screens/plant_report_screen.dart';
import '../../screens/user_profile_screen.dart';
import '../../screens/admin_panel_screen.dart';

class Sidebar extends StatelessWidget {
  final Function(int)? onItemTap;
  
  const Sidebar({this.onItemTap, super.key});

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          DrawerHeader(
            decoration: BoxDecoration(
              color: Theme.of(context).primaryColor,
            ),
            child: Consumer<AuthProvider>(
              builder: (context, auth, _) {
                final u = auth.user;
                final name = u?.name ?? 'EcoSense';
                final email = (u?.email.isNotEmpty ?? false)
                    ? u!.email
                    : 'Signed in user';
                return Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const CircleAvatar(
                      radius: 40,
                      child: Icon(Icons.person, size: 40),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      name,
                      style: GoogleFonts.inter(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                      textAlign: TextAlign.center,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    Text(
                      email,
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        color: Colors.white70,
                      ),
                      textAlign: TextAlign.center,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                );
              },
            ),
          ),
          _buildMenuItem(
            icon: Icons.home,
            title: AppTranslations.get(context, 'menu_home'),
            onTap: () {
              onItemTap?.call(0);
              Navigator.pop(context);
            },
          ),
          _buildMenuItem(
            icon: Icons.person,
            title: AppTranslations.get(context, 'menu_profile'),
            onTap: () {
              final rootNavigator =
                  Navigator.of(context, rootNavigator: true);
              Navigator.pop(context); // close drawer
              rootNavigator.push(
                MaterialPageRoute(
                  builder: (_) => const UserProfileScreen(),
                ),
              );
            },
          ),
          _buildMenuItem(
            icon: Icons.sensors,
            title: AppTranslations.get(context, 'menu_sensor'),
            onTap: () {
              final rootNavigator =
                  Navigator.of(context, rootNavigator: true);
              Navigator.pop(context); // close drawer
              rootNavigator.push(
                MaterialPageRoute(
                  builder: (_) => const SensorDetailsScreen(),
                ),
              );
            },
          ),
          _buildMenuItem(
            icon: Icons.history,
            title: AppTranslations.get(context, 'menu_history'),
            onTap: () {
              onItemTap?.call(3);
              Navigator.pop(context);
            },
          ),
          _buildMenuItem(
            icon: Icons.chat,
            title: AppTranslations.get(context, 'menu_chat'),
            onTap: () {
              final rootNavigator = Navigator.of(context, rootNavigator: true);
              Navigator.pop(context); // close drawer
              rootNavigator.push(
                MaterialPageRoute(builder: (_) => const AIChatScreen()),
              );
            },
          ),
          _buildMenuItem(
            icon: Icons.upload_file,
            title: AppTranslations.get(context, 'menu_upload'),
            onTap: () {
              onItemTap?.call(1);
              Navigator.pop(context);
            },
          ),
          _buildMenuItem(
            icon: Icons.description,
            title: AppTranslations.get(context, 'menu_report'),
            onTap: () {
              final rootNavigator =
                  Navigator.of(context, rootNavigator: true);
              Navigator.pop(context); // close drawer
              rootNavigator.push(
                MaterialPageRoute(
                  builder: (_) => const PlantReportScreen(),
                ),
              );
            },
          ),
          _buildMenuItem(
            icon: Icons.admin_panel_settings,
            title: AppTranslations.get(context, 'menu_admin'),
            onTap: () {
              final rootNavigator =
                  Navigator.of(context, rootNavigator: true);
              Navigator.pop(context); // close drawer
              rootNavigator.push(
                MaterialPageRoute(
                  builder: (_) => const AdminPanelScreen(),
                ),
              );
            },
          ),
          const Divider(),
          _buildMenuItem(
            icon: Icons.logout,
            title: AppTranslations.get(context, 'menu_logout'),
            onTap: () {
              final rootNavigator =
                  Navigator.of(context, rootNavigator: true);
              Navigator.pop(context);
              showDialog(
                context: context,
                builder: (_) => LogoutDialog(
                  onConfirm: () {
                    Provider.of<AuthProvider>(context, listen: false)
                        .logout()
                        .then((_) {
                      rootNavigator.pushAndRemoveUntil(
                        MaterialPageRoute(
                          builder: (_) => const LoginScreen(),
                        ),
                        (route) => false,
                      );
                    });
                  },
                ),
              );
            },
            isLogout: true,
          ),
        ],
      ),
    );
  }

  Widget _buildMenuItem({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
    bool isLogout = false,
  }) {
    return ListTile(
      leading: Icon(
        icon,
        color: isLogout ? Colors.red : null,
      ),
      title: Text(
        title,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w500,
          color: isLogout ? Colors.red : null,
        ),
      ),
      onTap: onTap,
    );
  }
  
}
