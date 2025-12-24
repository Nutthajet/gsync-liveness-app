import 'dart:async';
import 'dart:math' as math;
import 'package:sensors_plus/sensors_plus.dart';
import '../models/types.dart';

class SensorService {
  StreamSubscription<AccelerometerEvent>? _accelSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroSubscription;
  StreamSubscription<MagnetometerEvent>? _magSubscription;
  
  GyroData _gyroData = GyroData();
  
  // Store raw sensor values for fusion
  double _accelX = 0, _accelY = 0, _accelZ = 0;
  double _gyroX = 0, _gyroY = 0, _gyroZ = 0;
  double _magX = 0, _magY = 0, _magZ = 0;
  
  GyroData get gyroData => _gyroData;

  void startListening() {
    // Listen to accelerometer for tilt (beta) and roll (gamma)
    _accelSubscription = accelerometerEvents.listen((AccelerometerEvent event) {
      _accelX = event.x;
      _accelY = event.y;
      _accelZ = event.z;
      _updateOrientation();
    });

    // Listen to gyroscope for rotation rates
    _gyroSubscription = gyroscopeEvents.listen((GyroscopeEvent event) {
      _gyroX = event.x;
      _gyroY = event.y;
      _gyroZ = event.z;
    });

    // Listen to magnetometer for heading (alpha)
    _magSubscription = magnetometerEvents.listen((MagnetometerEvent event) {
      _magX = event.x;
      _magY = event.y;
      _magZ = event.z;
      _updateOrientation();
    });
  }

  void _updateOrientation() {
    // Calculate beta (tilt around X-axis) from accelerometer
    // Beta ranges from -180 to 180, but we'll clamp to -90 to 90 for display
    final beta = math.atan2(_accelY, math.sqrt(_accelX * _accelX + _accelZ * _accelZ)) * 180 / math.pi;
    
    // Calculate gamma (roll around Y-axis) from accelerometer
    final gamma = math.atan2(-_accelX, _accelZ) * 180 / math.pi;
    
    // Calculate alpha (yaw/heading around Z-axis) from magnetometer
    // This is a simplified calculation - proper fusion would use both accel and mag
    double? alpha;
    if (_magX != 0 || _magY != 0) {
      alpha = math.atan2(_magY, _magX) * 180 / math.pi;
      // Normalize to 0-360
      if (alpha < 0) alpha += 360;
    }
    
    _gyroData = GyroData(
      alpha: alpha,
      beta: beta.clamp(-90.0, 90.0),
      gamma: gamma.clamp(-90.0, 90.0),
    );
  }

  void stopListening() {
    _accelSubscription?.cancel();
    _gyroSubscription?.cancel();
    _magSubscription?.cancel();
    _accelSubscription = null;
    _gyroSubscription = null;
    _magSubscription = null;
  }

  void dispose() {
    stopListening();
  }
}

