import 'dart:async';
import 'dart:io';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:path_provider/path_provider.dart';

class SensorService {
  // Stream สำหรับกราฟ (Real-time)
  Stream<GyroscopeEvent> get gyroStream => gyroscopeEvents;

  // สำหรับอัดข้อมูล (Recording)
  List<List<dynamic>> _gyroData = [];
  List<List<dynamic>> _accelData = [];
  DateTime? _startTime;
  StreamSubscription? _gyroSub;
  StreamSubscription? _accelSub;

  void startRecording() {
    _gyroData.clear();
    _accelData.clear();
    _startTime = DateTime.now();

    // ฟังและเก็บลง List
    _gyroSub = gyroscopeEvents.listen((event) {
      if (_startTime != null) {
        double elapsed = DateTime.now().difference(_startTime!).inMicroseconds / 1000000.0;
        _gyroData.add([elapsed, event.x, event.y, event.z]);
      }
    });

    _accelSub = accelerometerEvents.listen((event) {
      if (_startTime != null) {
        double elapsed = DateTime.now().difference(_startTime!).inMicroseconds / 1000000.0;
        _accelData.add([elapsed, event.x, event.y, event.z]);
      }
    });
  }

  Future<Map<String, File>> stopRecordingAndGetFiles() async {
    _gyroSub?.cancel();
    _accelSub?.cancel();

    final directory = await getApplicationDocumentsDirectory();
    
    // Write Gyro CSV
    File gyroFile = File('${directory.path}/Gyroscope.csv');
    String gyroCsv = "seconds_elapsed,x,y,z\n";
    for (var row in _gyroData) {
      gyroCsv += "${row[0]},${row[1]},${row[2]},${row[3]}\n";
    }
    await gyroFile.writeAsString(gyroCsv);

    // Write Accel CSV
    File accelFile = File('${directory.path}/Accelerometer.csv');
    String accelCsv = "seconds_elapsed,x,y,z\n";
    for (var row in _accelData) {
      accelCsv += "${row[0]},${row[1]},${row[2]},${row[3]}\n";
    }
    await accelFile.writeAsString(accelCsv);

    return {"gyro": gyroFile, "accel": accelFile};
  }
}