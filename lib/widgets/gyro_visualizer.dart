import 'package:flutter/material.dart';
import '../models/types.dart';

class GyroVisualizer extends StatelessWidget {
  final GyroData data;

  const GyroVisualizer({
    super.key,
    required this.data,
  });

  double _normalize(double? val) {
    if (val == null) return 50.0;
    // Map -90 to 90 into 0 to 100
    final clamped = val.clamp(-90.0, 90.0);
    return ((clamped + 90.0) / 180.0) * 100.0;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.4),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.white.withOpacity(0.1),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'DEVICE SENSORS',
                style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF60A5FA),
                  letterSpacing: 1.5,
                ),
              ),
              const Icon(
                Icons.memory,
                color: Color(0xFF3B82F6),
                size: 16,
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildSensorBar(
            label: 'TILT (X)',
            value: data.beta,
            color: const Color(0xFF3B82F6),
          ),
          const SizedBox(height: 16),
          _buildSensorBar(
            label: 'ROLL (Y)',
            value: data.gamma,
            color: const Color(0xFF10B981),
          ),
          const SizedBox(height: 16),
          _buildSensorBar(
            label: 'YAW (Z)',
            value: data.alpha,
            color: const Color(0xFFA855F7),
          ),
        ],
      ),
    );
  }

  Widget _buildSensorBar({
    required String label,
    required double? value,
    required Color color,
  }) {
    final normalized = _normalize(value);
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 10,
                color: Colors.white70,
              ),
            ),
            Text(
              value?.toStringAsFixed(1) ?? '0.0',
              style: const TextStyle(
                fontSize: 10,
                color: Colors.white70,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: Container(
            height: 6,
            width: double.infinity,
            color: Colors.white.withOpacity(0.05),
            child: FractionallySizedBox(
              alignment: Alignment.centerLeft,
              widthFactor: normalized / 100,
              child: Container(
                color: color,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

