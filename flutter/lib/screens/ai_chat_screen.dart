import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../providers/ai_chat_provider.dart';
import '../core/localization/app_translations.dart';

class AIChatScreen extends StatefulWidget {
  const AIChatScreen({super.key});

  @override
  State<AIChatScreen> createState() => _AIChatScreenState();
}

class _AIChatScreenState extends State<AIChatScreen> {
  late final TextEditingController _controller;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _onSend(BuildContext context) async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;

    _controller.clear();
    await context.read<AIChatProvider>().sendMessage(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              AppTranslations.get(context, 'ai_chat_title'),
              style: GoogleFonts.inter(fontWeight: FontWeight.w700, fontSize: 18),
            ),
            Consumer<AIChatProvider>(
              builder: (context, provider, _) {
                return Row(
                  children: List.generate(5, (index) {
                    return Icon(
                      index < provider.remainingAttempts
                          ? Icons.bolt
                          : Icons.bolt_outlined,
                      color: index < provider.remainingAttempts
                          ? Colors.amber
                          : Colors.grey,
                      size: 16,
                    );
                  }),
                );
              },
            ),
          ],
        ),
      ),
      body: Consumer<AIChatProvider>(
        builder: (context, chatProvider, _) {
          return Column(
            children: [
              Expanded(
                child: ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: chatProvider.messages.length,
                  itemBuilder: (context, index) {
                    final msg = chatProvider.messages[index];
                    final isUser = msg.isUser;
                    final isDark = Theme.of(context).brightness == Brightness.dark;
                    final bg = isUser
                        ? const Color(0xFF2E7D32) // Primary green for user
                        : (isDark ? const Color(0xFF2A2A2A) : Colors.white);
                    final textColor = isUser
                        ? Colors.white
                        : (isDark ? Colors.white : Colors.black87);

                    final align =
                        isUser ? Alignment.centerRight : Alignment.centerLeft;

                    return Align(
                      alignment: align,
                      child: Padding(
                        padding: const EdgeInsets.only(bottom: 16),
                        child: Row(
                          mainAxisAlignment: isUser
                              ? MainAxisAlignment.end
                              : MainAxisAlignment.start,
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            if (!isUser) ...[
                              const CircleAvatar(
                                radius: 16,
                                backgroundColor: Color(0xFFE8F5E9),
                                child: Icon(Icons.eco, size: 18, color: Color(0xFF2E7D32)),
                              ),
                              const SizedBox(width: 8),
                            ],
                            Flexible(
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 16, vertical: 12),
                                decoration: BoxDecoration(
                                  color: bg,
                                  boxShadow: isUser
                                      ? []
                                      : [
                                          BoxShadow(
                                            color: Colors.black.withValues(alpha: 0.05),
                                            blurRadius: 5,
                                            offset: const Offset(0, 2),
                                          )
                                        ],
                                  borderRadius: BorderRadius.only(
                                    topLeft: const Radius.circular(16),
                                    topRight: const Radius.circular(16),
                                    bottomLeft: isUser
                                        ? const Radius.circular(16)
                                        : const Radius.circular(4),
                                    bottomRight: isUser
                                        ? const Radius.circular(4)
                                        : const Radius.circular(16),
                                  ),
                                ),
                                child: Text(
                                  msg.text,
                                  style: GoogleFonts.inter(
                                    fontSize: 15,
                                    color: textColor,
                                    height: 1.4,
                                  ),
                                ),
                              ),
                            ),
                            if (isUser) ...[
                              const SizedBox(width: 8),
                              CircleAvatar(
                                radius: 16,
                                backgroundColor: Theme.of(context).colorScheme.primary.withValues(alpha: 0.2),
                                child: const Icon(Icons.person, size: 18),
                              ),
                            ],
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
              if (chatProvider.error != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Text(
                    chatProvider.error!,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                ),
              SafeArea(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _controller,
                          minLines: 1,
                          maxLines: 4,
                          decoration: InputDecoration(
                            hintText: AppTranslations.get(context, 'type_message'),
                            hintStyle: TextStyle(color: Colors.grey.shade400),
                            filled: true,
                            fillColor: Theme.of(context).cardColor,
                            contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(24),
                              borderSide: BorderSide(color: Colors.grey.shade200),
                            ),
                            enabledBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(24),
                              borderSide: BorderSide(color: Colors.grey.shade200),
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(24),
                              borderSide: BorderSide(color: Theme.of(context).primaryColor),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      chatProvider.isLoading
                          ? const SizedBox(
                              width: 44,
                              height: 44,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : Container(
                              decoration: const BoxDecoration(
                                color: Color(0xFF2E7D32),
                                shape: BoxShape.circle,
                              ),
                              child: IconButton(
                                onPressed: chatProvider.remainingAttempts > 0
                                    ? () => _onSend(context)
                                    : null,
                                icon: const Icon(Icons.send_rounded),
                                color: Colors.white,
                              ),
                            ),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

