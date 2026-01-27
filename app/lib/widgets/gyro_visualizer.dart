import 'package:flutter/material.dart';
import '../models/types.dart';

class GyroVisualizer extends StatefulWidget {
  final GyroData data;
  const GyroVisualizer({super.key, required this.data});

  @override
  State<GyroVisualizer> createState() => _GyroVisualizerState();
}

class _GyroVisualizerState extends State<GyroVisualizer> {
  final List<double> _historyX = [];
  final List<double> _historyY = [];
  final List<double> _historyZ = [];
  final int _limit = 100;

  @override
  void didUpdateWidget(covariant GyroVisualizer oldWidget) {
    super.didUpdateWidget(oldWidget);
    _add(_historyX, widget.data.x);
    _add(_historyY, widget.data.y);
    _add(_historyZ, widget.data.z);
  }

  void _add(List<double> list, double val) {
    list.add(val);
    if (list.length > _limit) list.removeAt(0);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: CustomPaint(
        painter: _GraphPainter(_historyX, _historyY, _historyZ),
        child: Container(),
      ),
    );
  }
}

class _GraphPainter extends CustomPainter {
  final List<double> x, y, z;
  _GraphPainter(this.x, this.y, this.z);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..strokeWidth = 1.0..style = PaintingStyle.stroke;
    
    // Draw Axis
    canvas.drawLine(Offset(0, size.height/2), Offset(size.width, size.height/2), Paint()..color=Colors.white24);
    
    _draw(canvas, size, x, Colors.redAccent, paint);
    _draw(canvas, size, y, Colors.greenAccent, paint);
    _draw(canvas, size, z, Colors.blueAccent, paint);
  }

  void _draw(Canvas canvas, Size size, List<double> data, Color color, Paint paint) {
    if (data.isEmpty) return;
    paint.color = color;
    final path = Path();
    final stepX = size.width / (data.length > 1 ? data.length - 1 : 1);
    const scale = 12.0; // ปรับความสูงกราฟตรงนี้

    for (int i = 0; i < data.length; i++) {
      double py = (size.height / 2) - (data[i] * scale);
      double px = i * stepX;
      if (i == 0) path.moveTo(px, py); else path.lineTo(px, py);
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}