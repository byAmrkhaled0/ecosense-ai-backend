// Menu navigation constants and helpers
const List<MenuItemData> sidebarItems = [
  MenuItemData(
    id: 0,
    icon: 'home',
    label: 'Home',
    route: '/home',
  ),
  MenuItemData(
    id: 1,
    icon: 'person',
    label: 'User Profile',
    route: '/profile',
  ),
  MenuItemData(
    id: 2,
    icon: 'sensors',
    label: 'Sensor Details',
    route: '/sensors',
  ),
  MenuItemData(
    id: 3,
    icon: 'history',
    label: 'Prediction History',
    route: '/predictions',
  ),
  MenuItemData(
    id: 4,
    icon: 'chat',
    label: 'AI Chat',
    route: '/chat',
  ),
  MenuItemData(
    id: 5,
    icon: 'upload',
    label: 'Upload Dataset',
    route: '/upload',
  ),
  MenuItemData(
    id: 6,
    icon: 'report',
    label: 'Plant Report',
    route: '/report',
  ),
  MenuItemData(
    id: 7,
    icon: 'admin',
    label: 'Admin Panel',
    route: '/admin',
  ),
];

class MenuItemData {
  final int id;
  final String icon;
  final String label;
  final String route;

  const MenuItemData({
    required this.id,
    required this.icon,
    required this.label,
    required this.route,
  });
}
