enum LivenessStatus {
  idle,
  initializing,
  challengeRequested,
  recording,
  verifying,
  success,
  failed,
}

class GyroData {
  final double? alpha; // Rotation around z-axis
  final double? beta;  // Rotation around x-axis
  final double? gamma; // Rotation around y-axis

  GyroData({
    this.alpha,
    this.beta,
    this.gamma,
  });

  GyroData copyWith({
    double? alpha,
    double? beta,
    double? gamma,
  }) {
    return GyroData(
      alpha: alpha ?? this.alpha,
      beta: beta ?? this.beta,
      gamma: gamma ?? this.gamma,
    );
  }
}

class Challenge {
  final String id;
  final String instruction;
  final String expectedMovement; // 'tilt_up' | 'tilt_down' | 'rotate_left' | 'rotate_right' | 'steady'

  Challenge({
    required this.id,
    required this.instruction,
    required this.expectedMovement,
  });

  static final List<Challenge> challenges = [
    Challenge(
      id: '1',
      instruction: 'Please tilt your phone UP slowly',
      expectedMovement: 'tilt_up',
    ),
    Challenge(
      id: '2',
      instruction: 'Please tilt your phone DOWN slowly',
      expectedMovement: 'tilt_down',
    ),
    Challenge(
      id: '3',
      instruction: 'Slowly rotate your phone to the LEFT',
      expectedMovement: 'rotate_left',
    ),
    Challenge(
      id: '4',
      instruction: 'Slowly rotate your phone to the RIGHT',
      expectedMovement: 'rotate_right',
    ),
  ];
}

class VerificationResult {
  final bool isLive;
  final double confidence;
  final String reasoning;

  VerificationResult({
    required this.isLive,
    required this.confidence,
    required this.reasoning,
  });

  factory VerificationResult.fromJson(Map<String, dynamic> json) {
    return VerificationResult(
      isLive: json['isLive'] as bool,
      confidence: (json['confidence'] as num).toDouble(),
      reasoning: json['reasoning'] as String,
    );
  }
}

