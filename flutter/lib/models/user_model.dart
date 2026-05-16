class UserModel {
  final String id;
  final String name;
  final String email;
  final String? firstName;
  final String? lastName;
  final String? profileImage;
  final String? phone;
  final String? location;
  final String? ownerId;
  final String userType;
  final bool isDarkMode;
  final String language;
  final DateTime createdAt;
  final DateTime? updatedAt;

  UserModel({
    required this.id,
    required this.name,
    required this.email,
    this.firstName,
    this.lastName,
    this.profileImage,
    this.phone,
    this.location,
    this.ownerId,
    required this.userType,
    this.isDarkMode = false,
    this.language = 'en',
    required this.createdAt,
    this.updatedAt,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) {
    final id = json['id']?.toString() ?? '';
    final firstName = json['firstName'] as String?;
    final lastName = json['lastName'] as String?;
    final legacyName = json['name'] as String?;
    var displayName = legacyName ?? '';
    if (displayName.isEmpty) {
      displayName = [firstName, lastName]
          .map((e) => e?.trim() ?? '')
          .where((e) => e.isNotEmpty)
          .join(' ');
    }
    if (displayName.isEmpty) {
      displayName = 'User';
    }

    final email = json['email'] as String? ?? '';
    final role = json['role'] as String? ??
        json['userType'] as String? ??
        'owner';

    DateTime parseDate(dynamic v) {
      if (v == null) return DateTime.now();
      if (v is DateTime) return v;
      return DateTime.tryParse(v.toString()) ?? DateTime.now();
    }

    return UserModel(
      id: id,
      name: displayName,
      email: email,
      firstName: firstName,
      lastName: lastName,
      profileImage: json['profileImage'] as String?,
      phone: (json['phone'] ?? json['phoneNumber']) as String?,
      location: (json['location'] ?? json['address']) as String?,
      ownerId: json['ownerId']?.toString(),
      userType: role,
      isDarkMode: json['isDarkMode'] as bool? ?? false,
      language: json['language'] as String? ?? 'en',
      createdAt: parseDate(json['createdAt']),
      updatedAt: json['updatedAt'] != null
          ? DateTime.tryParse(json['updatedAt'].toString())
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'email': email,
      'firstName': firstName,
      'lastName': lastName,
      'profileImage': profileImage,
      'phone': phone,
      'location': location,
      'ownerId': ownerId,
      'userType': userType,
      'isDarkMode': isDarkMode,
      'language': language,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt?.toIso8601String(),
    };
  }

  UserModel copyWith({
    String? id,
    String? name,
    String? email,
    String? firstName,
    String? lastName,
    String? profileImage,
    String? phone,
    String? location,
    String? ownerId,
    String? userType,
    bool? isDarkMode,
    String? language,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return UserModel(
      id: id ?? this.id,
      name: name ?? this.name,
      email: email ?? this.email,
      firstName: firstName ?? this.firstName,
      lastName: lastName ?? this.lastName,
      profileImage: profileImage ?? this.profileImage,
      phone: phone ?? this.phone,
      location: location ?? this.location,
      ownerId: ownerId ?? this.ownerId,
      userType: userType ?? this.userType,
      isDarkMode: isDarkMode ?? this.isDarkMode,
      language: language ?? this.language,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }
}
