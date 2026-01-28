import React, { useState, useRef, useCallback, useEffect } from 'react';

// --- ENUMS & TYPES ---

enum Status {
  IDLE,
  REQUESTING_PERMISSIONS,
  READY,
  SCANNING,
  PROCESSING,
  SUCCESS,
  FAILED,
}

type SensorDataPoint = {
  t: number;
  x: number | null;
  y: number | null;
  z: number | null;
};

// --- ICON COMPONENTS (self-contained SVGs, inspired by Lucide) ---

const IconWrapper: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    {children}
  </svg>
);

const VideoIcon = ({ className }: { className?: string }) => (
  <IconWrapper className={className}>
    <path d="m22 8-6 4 6 4V8Z" />
    <rect width="14" height="12" x="2" y="6" rx="2" ry="2" />
  </IconWrapper>
);

const CameraOffIcon = ({ className }: { className?: string }) => (
  <IconWrapper className={className}>
    <line x1="2" x2="22" y1="2" y2="22" />
    <path d="M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34" />
    <path d="m15.73 15.73-4.4-4.4" />
    <path d="M11.5 11.5 8 15l-4-4" />
  </IconWrapper>
);

const LoaderIcon = ({ className }: { className?: string }) => (
  <IconWrapper className={`animate-spin ${className}`}>
    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
  </IconWrapper>
);

const CheckCircleIcon = ({ className }: { className?: string }) => (
  <IconWrapper className={className}>
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </IconWrapper>
);

const XCircleIcon = ({ className }: { className?: string }) => (
  <IconWrapper className={className}>
    <circle cx="12" cy="12" r="10" />
    <line x1="15" y1="9" x2="9" y2="15" />
    <line x1="9" y1="9" x2="15" y2="15" />
  </IconWrapper>
);

const ShieldCheckIcon = ({ className }: { className?: string }) => (
    <IconWrapper className={className}>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
        <path d="m9 12 2 2 4-4" />
    </IconWrapper>
);

// --- MAIN APP COMPONENT ---

const App: React.FC = () => {
  const [status, setStatus] = useState<Status>(Status.IDLE);
  const [error, setError] = useState<string | null>(null);
  const [apiResult, setApiResult] = useState<any | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const gyroDataRef = useRef<SensorDataPoint[]>([]);
  const accelDataRef = useRef<SensorDataPoint[]>([]);
  const recordingStartTimeRef = useRef<number | null>(null);
  
  const RECORDING_DURATION_MS = 5000;

  const isIOS = typeof (DeviceMotionEvent as any)?.requestPermission === 'function';

  const createCSV = (data: SensorDataPoint[]): Blob => {
    const header = "seconds_elapsed,x,y,z\n";
    const rows = data
      .map(row => 
        `${row.t.toFixed(4)},${row.x ?? ''},${row.y ?? ''},${row.z ?? ''}`
      )
      .join("\n");
    return new Blob([header + rows], { type: 'text/csv' });
  };

  const cleanup = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    window.removeEventListener('devicemotion', handleDeviceMotion);
    window.removeEventListener('deviceorientation', handleDeviceOrientation);
  }, []);

  const handleReset = () => {
    cleanup();
    setStatus(Status.IDLE);
    setError(null);
    setApiResult(null);
    recordedChunksRef.current = [];
    gyroDataRef.current = [];
    accelDataRef.current = [];
    recordingStartTimeRef.current = null;
  };

  const handleDeviceMotion = useCallback((event: DeviceMotionEvent) => {
    if (recordingStartTimeRef.current === null) return;
    const { accelerationIncludingGravity } = event;
    const timestamp = (performance.now() - recordingStartTimeRef.current) / 1000;
    accelDataRef.current.push({
      t: timestamp,
      x: accelerationIncludingGravity?.x ?? null,
      y: accelerationIncludingGravity?.y ?? null,
      z: accelerationIncludingGravity?.z ?? null,
    });
  }, []);

  const handleDeviceOrientation = useCallback((event: DeviceOrientationEvent) => {
    if (recordingStartTimeRef.current === null) return;
    const { alpha, beta, gamma } = event;
    const timestamp = (performance.now() - recordingStartTimeRef.current) / 1000;
    gyroDataRef.current.push({
      t: timestamp,
      x: alpha,
      y: beta,
      z: gamma,
    });
  }, []);
  
  // รวม StartCamera + Permission ไว้ที่เดียวกัน
  const handleRequestPermissions = async () => {
    setStatus(Status.REQUESTING_PERMISSIONS);
    setError(null);

    try {
      // 1. ขอ Permission พิเศษสำหรับ iOS (ถ้ามี)
      if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
        const motionPermission = await (DeviceMotionEvent as any).requestPermission();
        if (motionPermission !== 'granted') throw new Error('Device motion permission denied.');
        
        const orientationPermission = await (DeviceOrientationEvent as any).requestPermission();
        if (orientationPermission !== 'granted') throw new Error('Device orientation permission denied.');
      }
      
      // 2. ขอเข้าถึงกล้อง
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
        }, 
        audio: false 
      });

      console.log("✅ Camera Stream Acquired");

      // 3. เปลี่ยนสถานะเป็น READY เพื่อให้ React วาด <video> ออกมา
      setStatus(Status.READY);

      // 4. รอเล็กน้อยให้ <video> โผล่มา แล้วค่อยยัด Stream ใส่ (แก้จอดำ)
      setTimeout(() => {
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
            // บังคับเล่น
            videoRef.current.play().catch(e => {
                console.warn("Autoplay blocked, muting...", e);
                videoRef.current!.muted = true;
                videoRef.current!.play();
            });
        }
        
        // เก็บ Stream ไว้ใช้ตอนปิดกล้อง (ถ้ามี ref นี้)
        if (typeof streamRef !== 'undefined') {
            streamRef.current = stream; 
        }
      }, 100); // รอ 100ms ให้ UI พร้อม

    } catch (err: any) {
      console.error(err);
      let friendlyError = 'An unknown error occurred.';
      
      if (err.name === 'NotAllowedError') friendlyError = 'Camera permission denied.';
      else if (err.name === 'NotFoundError') friendlyError = 'No camera found.';
      else if (err.message) friendlyError = err.message;

      setError(friendlyError);
      setStatus(Status.FAILED);
    }
  };
  
  const uploadData = async () => {
    // 🔴 แก้จุดที่ 1: สร้าง Video Blob จาก chunksRef โดยตรง (แก้ ReferenceError)
    const videoBlob = new Blob(recordedChunksRef.current, { type: 'video/webm' });

    // --- Helper: สร้างไฟล์ CSV จากข้อมูลจริง ---
    const createCSV = (data: SensorDataPoint[]) => {
      const header = "seconds_elapsed,x,y,z\n";
      const rows = data.map(d => `${d.t.toFixed(4)},${d.x},${d.y},${d.z}`).join("\n");
      return new Blob([header + rows], { type: 'text/csv' });
    };

    // --- Helper: สร้างข้อมูลหลอก (Dummy) ถ้าเทสบนคอม ---
    const generateDummyCsv = () => {
        let csvContent = "seconds_elapsed,x,y,z\n";
        // สร้างข้อมูล 30 แถว (ประมาณ 3 วินาที)
        for (let i = 0; i < 30; i++) {
            csvContent += `${(i * 0.1).toFixed(2)},0.0,0.0,9.8\n`;
        }
        return new Blob([csvContent], { type: 'text/csv' });
    };

    // 🔴 แก้จุดที่ 2: เช็คว่ามีข้อมูลเซนเซอร์ไหม ถ้าไม่มี (เล่นบนคอม) ให้ใช้ Dummy
    const gyroBlob = gyroDataRef.current.length > 5 
        ? createCSV(gyroDataRef.current) 
        : generateDummyCsv();

    const accelBlob = accelDataRef.current.length > 5 
        ? createCSV(accelDataRef.current) 
        : generateDummyCsv();

    // 3. แพ็คลง FormData
    const formData = new FormData();
    formData.append('video', videoBlob, 'video.webm'); // ใช้ videoBlob ที่สร้างบรรทัดแรก
    formData.append('gyroscope', gyroBlob, 'gyro.csv');
    formData.append('accelerometer', accelBlob, 'accel.csv');

    // 4. ส่งไป Server
    try {
      setStatus(Status.PROCESSING);
      
      // 🔴 ตรวจสอบ URL: ถ้าเทสบนคอมใช้ localhost:8000 / ถ้าเทสมือถือใช้ Ngrok URL
      const response = await fetch('https://malika-shedable-recollectively.ngrok-free.dev', { 
        method: 'POST',
        headers: {
            'ngrok-skip-browser-warning': 'true',
        },
        body: formData,
      });

      if (!response.ok) {
         const errorText = await response.text();
         console.error("Server Error Details:", errorText);
         throw new Error(`Server Error ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log("Verification Result:", result);
      
      // ปรับ Logic การแสดงผลตาม Response ของ Server
      // เช็คทั้ง boolean (true/false) และ string ("REAL"/"FAKE")
      if (result.result === true || result.result === "REAL" || result.classification === "REAL") {
          setApiResult(result);
          setStatus(Status.SUCCESS);
      } else {
          setApiResult(result);
          setStatus(Status.FAILED);
          // ดึงคะแนนความมั่นใจมาโชว์ (ถ้ามี)
          const score = result.score || result.confidence || 0;
          setError(`Fake Detected (Score: ${Number(score).toFixed(4)})`);
      }

    } catch (error: any) {
      console.error("Upload failed:", error);
      setStatus(Status.FAILED);
      setError(error.message || "Upload failed");
    }
  };

  const handleStartScanning = () => {
    if (!streamRef.current) {
      setError('Camera stream not available.');
      setStatus(Status.FAILED);
      return;
    }
    
    setStatus(Status.SCANNING);
    recordedChunksRef.current = [];
    gyroDataRef.current = [];
    accelDataRef.current = [];

    mediaRecorderRef.current = new MediaRecorder(streamRef.current, { mimeType: 'video/webm' });
    
    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };
    
    mediaRecorderRef.current.onstop = () => {
        setStatus(Status.PROCESSING);
        const videoBlob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        uploadData(videoBlob);
    };
    
    window.addEventListener('devicemotion', handleDeviceMotion);
    window.addEventListener('deviceorientation', handleDeviceOrientation);
    
    recordingStartTimeRef.current = performance.now();
    mediaRecorderRef.current.start();

    setTimeout(() => {
        if(mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, RECORDING_DURATION_MS);
    
  };

// 🟢 สั่งเริ่มระบบทันทีที่เปิดหน้าเว็บ
  useEffect(() => {
    // เรียกใช้ฟังก์ชันที่เราเพิ่งแก้ด้านบน
    handleRequestPermissions();

    // Cleanup: ปิดกล้องเมื่อปิดหน้าเว็บ
    return () => {
       // โค้ดปิดกล้อง (ถ้ามี streamRef ให้ใช้ streamRef)
       if (videoRef.current && videoRef.current.srcObject) {
         const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
         tracks.forEach(track => track.stop());
       }
    };
  }, []); // ทำงานครั้งเดียวตอนโหลด
  
  const renderContent = () => {
    switch (status) {
      case Status.IDLE:
        return (
          <div className="text-center">
            <ShieldCheckIcon className="w-24 h-24 mx-auto text-rose-500" />
            <h1 className="text-3xl font-bold mt-4">Gsync Liveness</h1>
            <p className="text-gray-400 mt-2 mb-8">Secure identity verification</p>
            <button onClick={handleRequestPermissions} className="bg-rose-600 hover:bg-rose-700 text-white font-bold py-3 px-6 rounded-full flex items-center gap-2 transition-all duration-300 mx-auto transform hover:scale-105">
              <VideoIcon />
              Start Verification
            </button>
          </div>
        );
      case Status.REQUESTING_PERMISSIONS:
      case Status.PROCESSING:
          const message = status === Status.PROCESSING ? 'Analyzing biometric data...' : 'Accessing secure devices...';
          return (
            <div className="text-center">
                <LoaderIcon className="w-16 h-16 text-rose-500 mx-auto" />
                <p className="mt-4 text-lg">{message}</p>
            </div>
          );
      case Status.READY:
      case Status.SCANNING:
        return (
          <div className="flex flex-col items-center gap-4 w-full max-w-md">
            <div className="relative w-full aspect-[3/4] rounded-2xl overflow-hidden border-2 border-rose-500/50 shadow-lg shadow-rose-900/50">
              <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted 
                  className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]">
              </video>
              {status === Status.SCANNING && (
                  <div className="absolute top-4 right-4 flex items-center gap-2 bg-black/50 p-2 rounded-lg">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse-strong"></div>
                      <span className="font-mono text-sm text-white">REC</span>
                  </div>
              )}
            </div>
            {status === Status.READY && (
                 <button onClick={handleStartScanning} className="bg-rose-600 hover:bg-rose-700 text-white font-bold py-3 px-6 rounded-full flex items-center gap-2 transition-all duration-300 transform hover:scale-105">
                     Start Scan
                 </button>
            )}
             {status === Status.SCANNING && (
                 <p className="text-rose-400">Recording in progress...</p>
            )}
          </div>
        );
      case Status.SUCCESS:
        // 🔍 DEBUG: ดูค่าใน Console ว่าข้อมูลมาจริงไหม
        console.log("👉 API Result Debug:", apiResult);

        // ดึงค่าแบบปลอดภัย (ใช้ as any เพื่อกัน Error เรื่อง Type)
        const rawScore = (apiResult as any)?.pass_rate;
        const rawConf = (apiResult as any)?.confidence;

        // แปลงค่า (ถ้าไม่มีข้อมูล ให้เป็น 0)
        const successScore = rawScore ? (rawScore).toFixed(2) : "0.00";
        const successConf = rawConf || "N/A";
        return (
            <div className="text-center bg-gray-900/80 p-8 rounded-2xl border border-green-500/30 backdrop-blur-xl w-full max-w-sm">
                <div className="inline-block p-4 rounded-full bg-green-500/10 mb-4">
                    <CheckCircleIcon className="w-16 h-16 text-green-400" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-1">Human Verified</h2>
                <p className="text-green-400 text-sm mb-6">Liveness check passed successfully</p>
                
                {/* Grid แสดงคะแนน */}
                <div className="grid grid-cols-2 gap-3 mb-6">
                    <div className="bg-gray-800/50 p-3 rounded-xl border border-gray-700">
                        <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Liveness pass_rate</p>
                        <p className="text-2xl font-mono font-bold text-white">{successScore}%</p>
                    </div>
                    <div className="bg-gray-800/50 p-3 rounded-xl border border-gray-700">
                        <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Confidence</p>
                        <p className="text-2xl font-mono font-bold text-blue-400">{successConf}</p>
                    </div>
                </div>

                <button onClick={handleReset} className="w-full bg-gray-700 hover:bg-gray-600 text-white font-bold py-3 px-4 rounded-xl transition-all">
                    Scan Again
                </button>
            </div>
        );

      // 🔴🔴🔴 จุดที่แก้: แสดงผล Failed พร้อม Score (ถ้ามี) 🔴🔴🔴
      case Status.FAILED:
        return (
            <div className="text-center bg-gray-900/80 p-8 rounded-2xl border border-red-500/30 backdrop-blur-xl w-full max-w-sm">
                <div className="inline-block p-4 rounded-full bg-red-500/10 mb-4">
                    <XCircleIcon className="w-16 h-16 text-red-400" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-1">Verification Failed</h2>
                <p className="text-red-400 text-sm mb-6 font-mono bg-red-950/30 py-2 px-3 rounded-lg inline-block">
                    {error || "Spoof attempt detected"}
                </p>

                <p className="text-gray-400 text-xs mb-6 px-4">
                    Ensure you are moving the phone slightly while keeping your face in frame.
                </p>

                <button onClick={handleReset} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-xl transition-all shadow-lg shadow-red-900/20">
                    Try Again
                </button>
            </div>
        );

      default:
        return null;
    }
  };
  return (
    <div className="bg-dark-bg text-gray-200 min-h-screen w-full flex flex-col items-center justify-center p-4 font-mono antialiased">
      <main className="w-full max-w-lg flex items-center justify-center">
        {renderContent()}
      </main>
      <footer className="text-gray-600 text-xs text-center mt-8 absolute bottom-4">
      </footer>
    </div>
  );
};

export default App;