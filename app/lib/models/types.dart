enum LivenessStatus { idle, pending, verifying, success, failed }

class GyroData {
  final double x;
  final double y;
  final double z;

  GyroData({this.x = 0, this.y = 0, this.z = 0});
}

class VerificationResult {
  final bool isReal;
  final double confidence;
  final String reason;

  VerificationResult({
    required this.isReal,
    required this.confidence,
    required this.reason,
  });
}