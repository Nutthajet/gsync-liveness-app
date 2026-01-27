import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
import joblib
import mediapipe as mp
import os
import shutil
import sys
import traceback
from scipy.interpolate import interp1d

# ================= 1. MODEL CONFIG =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตหมด
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก Method (GET, POST, OPTIONS)
    allow_headers=["*"],  # อนุญาตทุก Header (รวมถึง ngrok-skip-browser-warning)
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path Configuration with validation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'detection_model', 'kyc_model_best.pth')
SCALER_PATH = os.path.join(BASE_DIR, 'detection_model', 'scaler.pkl')

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'kyc_model_best.pth')
    SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

FLOW_RESIZE_WIDTH = 480
SEQ_LEN = 30

# Validate files exist
print("="*60)
print("🔍 VALIDATING MODEL FILES")
print("="*60)
print(f"📂 Current directory: {os.getcwd()}")
print(f"📂 Script directory: {BASE_DIR}")
print(f"📍 Model path: {MODEL_PATH}")
print(f"📍 Scaler path: {SCALER_PATH}")
print()

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found!")
    print(f"   Expected at: {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(SCALER_PATH):
    print(f"❌ ERROR: Scaler file not found!")
    print(f"   Expected at: {SCALER_PATH}")
    sys.exit(1)

print("✅ Model file found")
print("✅ Scaler file found")
print()

class AntiDeepfakeModelPro(nn.Module):
    def __init__(self, vision_dim=19, sensor_dim=6, hidden_dim=48):
        super().__init__()
        self.vision_dim = vision_dim
        self.sensor_dim = sensor_dim
        
        # GRU Layers
        self.gru_vision = nn.GRU(vision_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_sensor = nn.GRU(sensor_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden_dim * 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128), # Layer 0
            nn.ReLU(),                      # Layer 1
            nn.Dropout(0.5),                # Layer 2
            nn.Linear(128, 64),             # Layer 3
            nn.ReLU(),                      # Layer 4 
            nn.Dropout(0.5),                # Layer 5 
            nn.Linear(64, 1),               # Layer 6 
            nn.Sigmoid()                    # Layer 7
        )
    
    def forward(self, x):
        vis, sen = x[:, :, :self.vision_dim], x[:, :, self.vision_dim:]
        _, h_v = self.gru_vision(vis)
        _, h_s = self.gru_sensor(sen)
        feat_v = torch.cat((h_v[-2], h_v[-1]), dim=1)
        feat_s = torch.cat((h_s[-2], h_s[-1]), dim=1)
        return self.classifier(self.bn(torch.cat((feat_v, feat_s), dim=1)))

# Load Model & Tools with error handling
print("⏳ Loading Model & Scaler...")
try:
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("📦 Loaded model from checkpoint['model']")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("📦 Loaded model from checkpoint['state_dict']")
        else:
            state_dict = checkpoint
            print("📦 Loaded model from checkpoint")
    else:
        state_dict = checkpoint
        print("📦 Loaded model directly")
    
    # Auto-detect dimensions from state_dict
    try:
        gru_vision_weight = state_dict['gru_vision.weight_ih_l0']
        gru_sensor_weight = state_dict['gru_sensor.weight_ih_l0']
        
        vision_dim = gru_vision_weight.shape[1]     
        sensor_dim = gru_sensor_weight.shape[1]      
        hidden_dim = gru_vision_weight.shape[0] // 3 
        
        print(f"📊 Auto-detected dimensions:")
        print(f"   Vision features: {vision_dim}")
        print(f"   Sensor features: {sensor_dim}")
        print(f"   Hidden dimension: {hidden_dim}")
        print(f"   Total features: {vision_dim + sensor_dim}")
    except Exception as e:
        print(f"⚠️  Auto-detect failed: {e}")
        # Fallback to corrected defaults
        vision_dim, sensor_dim, hidden_dim = 19, 6, 48
        print(f"⚠️  Using default dimensions: vision={vision_dim}, sensor={sensor_dim}, hidden={hidden_dim}")
    
    # Create and load model
    model = AntiDeepfakeModelPro(vision_dim=vision_dim, sensor_dim=sensor_dim, hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"✅ Model loaded successfully on {DEVICE}")
    
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully")
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe Face Mesh initialized")
    
    print()
    print("="*60)
    print("🚀 SERVER READY!")
    print("="*60)
    print()
    
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    sys.exit(1)
except RuntimeError as e:
    print(f"❌ Model loading error: {e}")
    print()
    print("💡 This usually means the model architecture doesn't match the saved weights.")
    print("   Please check if you're using the correct model file.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error loading model: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Store dimensions globally for later use
TOTAL_FEATURES = vision_dim + sensor_dim

# ================= 2. HELPER FUNCTIONS =================
def get_background_flow(prev_gray, curr_gray, face_landmarks, img_w, img_h):
    """Calculate background optical flow to detect screen replay attacks"""
    if prev_gray is None: 
        return 0.0, 0.0
    
    try:
        scale = FLOW_RESIZE_WIDTH / img_w
        target_h = int(img_h * scale)
        prev_small = cv2.resize(prev_gray, (FLOW_RESIZE_WIDTH, target_h))
        curr_small = cv2.resize(curr_gray, (FLOW_RESIZE_WIDTH, target_h))
        
        # Create mask to exclude face region
        mask = np.ones_like(prev_small, dtype=np.uint8) * 255
        face_points = [(int(lm.x * FLOW_RESIZE_WIDTH), int(lm.y * target_h)) for lm in face_landmarks]
        if face_points and len(face_points) > 3:
            hull = cv2.convexHull(np.array(face_points))
            cv2.fillConvexPoly(mask, hull, 0)

        # Detect features in background
        p0 = cv2.goodFeaturesToTrack(
            prev_small, 
            mask=mask, 
            maxCorners=20, 
            qualityLevel=0.2, 
            minDistance=5,
            blockSize=3
        )
        
        if p0 is None or len(p0) < 5: 
            return 0.0, 0.0
        
        # Calculate optical flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_small, 
            curr_small, 
            p0, 
            None, 
            winSize=(15, 15), 
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) > 0:
                d = good_new - good_old
                return float(np.mean(d[:, 0])), float(np.mean(d[:, 1]))
                
    except Exception as e:
        print(f"⚠️  Optical flow calculation error: {str(e)}")
    
    return 0.0, 0.0

def calculate_ear(landmarks, indices):
    """Calculate Eye Aspect Ratio"""
    # EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    try:
        p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
        p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
        p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
        p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
        p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
        p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])

        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        dist = np.linalg.norm(p1 - p4)
        
        return (v1 + v2) / (2.0 * dist) if dist > 0 else 0.0
    except:
        return 0.0

def calculate_mar(landmarks, indices):
    """Calculate Mouth Aspect Ratio"""
    try:
        p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) # มุมปากซ้าย
        p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]) # มุมปากขวา
        
        p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
        p8 = np.array([landmarks[indices[7]].x, landmarks[indices[7]].y])
        
        p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
        p7 = np.array([landmarks[indices[6]].x, landmarks[indices[6]].y])
        
        p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
        p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])

        v1 = np.linalg.norm(p2 - p8)
        v2 = np.linalg.norm(p3 - p7)
        v3 = np.linalg.norm(p4 - p6)
        dist = np.linalg.norm(p1 - p5)

        return (v1 + v2 + v3) / (2.0 * dist) if dist > 0 else 0.0
    except:
        return 0.0

def process_data(video_path, gyro_path, accel_path):
    """Extract features from video and sensor data"""
    print(f"📹 Processing video: {os.path.basename(video_path)}")
    print(f"📱 Processing sensors: {os.path.basename(gyro_path)}, {os.path.basename(accel_path)}")
    
    # Load sensor data
    try:
        df_gyro = pd.read_csv(gyro_path)
        df_accel = pd.read_csv(accel_path)
        print(f"✅ Loaded gyro: {len(df_gyro)} rows, accel: {len(df_accel)} rows")
        
        # Validate columns
        required_cols = ['seconds_elapsed', 'x', 'y', 'z']
        if not all(col in df_gyro.columns for col in required_cols):
            print(f"❌ Gyroscope CSV missing required columns: {required_cols}")
            return None
        if not all(col in df_accel.columns for col in required_cols):
            print(f"❌ Accelerometer CSV missing required columns: {required_cols}")
            return None
            
    except Exception as e:
        print(f"❌ Error reading sensor CSV files: {str(e)}")
        return None

    # Create interpolation functions for sensor data
    try:
        t_g = df_gyro['seconds_elapsed'].values
        t_a = df_accel['seconds_elapsed'].values
        
        fg = [interp1d(t_g, df_gyro[c].values, fill_value="extrapolate", bounds_error=False) 
              for c in ['x', 'y', 'z']]
        fa = [interp1d(t_a, df_accel[c].values, fill_value="extrapolate", bounds_error=False) 
              for c in ['x', 'y', 'z']]
        
        print(f"✅ Sensor interpolation ready (time range: {t_g.min():.2f}s - {t_g.max():.2f}s)")
        
    except Exception as e:
        print(f"❌ Error creating sensor interpolations: {str(e)}")
        return None

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS")
    
    # Create timestamps for video frames
    timestamps = np.linspace(t_g.min(), t_g.max(), total_frames)
    
    extracted = []
    prev_gray = None
    idx = 0
    frames_with_face = 0
    frames_without_face = 0
    
    print("⏳ Processing frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        ts = timestamps[min(idx, len(timestamps) - 1)]
        
        # Convert to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape
        
        # Process with MediaPipe
        results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Calculate background optical flow
            fx, fy = get_background_flow(prev_gray, curr_gray, lm, w, h)
            
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            MOUTH = [61, 185, 40, 39, 291, 146, 91, 181] # 8 จุดรอบปากใน

            ear_left = calculate_ear(lm, LEFT_EYE)
            ear_right = calculate_ear(lm, RIGHT_EYE)
            avg_ear = (ear_left + ear_right) / 2.0
            mar = calculate_mar(lm, MOUTH)
            
            # Extract facial landmarks (5 key points: nose, left eye, right eye, left mouth, right mouth)
            # และ optical flow + sensor data
            row = [
                lm[1].x, lm[1].y, lm[1].z,          # nose tip
                lm[33].x, lm[33].y, lm[33].z,       # left eye inner corner
                lm[263].x, lm[263].y, lm[263].z,    # right eye inner corner
                lm[234].x, lm[234].y, lm[234].z,    # left mouth corner
                lm[454].x, lm[454].y, lm[454].z,    # right mouth corner
                fx, fy,                              # background optical flow (2 features)
                
                avg_ear, mar,

                float(fg[0](ts)), float(fg[1](ts)), float(fg[2](ts)),   # gyroscope (3 features)
                float(fa[0](ts)), float(fa[1](ts)), float(fa[2](ts))    # accelerometer (3 features)
            ]
            
            extracted.append(row)
            frames_with_face += 1
        else:
            frames_without_face += 1
        
        prev_gray = curr_gray.copy()
        idx += 1
        
        # Progress indicator
        if idx % 30 == 0:
            print(f"   Processed {idx}/{total_frames} frames ({frames_with_face} with face)...", end='\r')
    
    cap.release()
    
    print()  # New line after progress
    print(f"✅ Processing complete: {frames_with_face} frames with face, {frames_without_face} without")
    
    if len(extracted) == 0:
        print("❌ No faces detected in video")
        return None
    
    return np.array(extracted)

# ================= 3. API ENDPOINTS =================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "AntiDeepfake Pro",
        "device": str(DEVICE),
        "features": TOTAL_FEATURES,
        "sequence_length": SEQ_LEN
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "mediapipe_ready": mp_face_mesh is not None,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/verify")
async def verify(
    video: UploadFile = File(...), 
    gyroscope: UploadFile = File(...), 
    accelerometer: UploadFile = File(...)
):
    """Main liveness verification endpoint"""
    print()
    print("="*60)
    print(f"🔍 NEW VERIFICATION REQUEST")
    print("="*60)
    print(f"📹 Video: {video.filename} ({video.content_type})")
    print(f"📱 Gyroscope: {gyroscope.filename}")
    print(f"📱 Accelerometer: {accelerometer.filename}")
    print()
    
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    v_path = f"temp/{video.filename}"
    g_path = f"temp/{gyroscope.filename}"
    a_path = f"temp/{accelerometer.filename}"
    
    try:
        # Save uploaded files
        print("💾 Saving uploaded files...")
        with open(v_path, "wb") as f: 
            shutil.copyfileobj(video.file, f)
        with open(g_path, "wb") as f: 
            shutil.copyfileobj(gyroscope.file, f)
        with open(a_path, "wb") as f: 
            shutil.copyfileobj(accelerometer.file, f)
        
        v_size = os.path.getsize(v_path) / 1024 / 1024  # MB
        print(f"✅ Files saved (video: {v_size:.2f} MB)")
        print()
        
        # Process data
        data = process_data(v_path, g_path, a_path)
        
        if data is None:
            raise HTTPException(
                status_code=400, 
                detail="Failed to process video. Possible issues: no face detected, invalid sensor data, or corrupted files."
            )
        
        print()
        print(f"📊 Extracted {len(data)} frames with features")
        
        if len(data) < SEQ_LEN:
            raise HTTPException(
                status_code=400, 
                detail=f"Video too short. Need at least {SEQ_LEN} frames with detected face, got {len(data)}."
            )
        
        # Create sequences
        print(f"🔄 Creating sequences (length={SEQ_LEN})...")
        sequences = []
        for i in range(len(data) - SEQ_LEN + 1):
            sequences.append(data[i:i + SEQ_LEN])
        X = np.array(sequences)
        
        print(f"✅ Created {len(X)} sequences of shape {X.shape}")
        
        current_features = X.shape[-1]
        if current_features != TOTAL_FEATURES:
            print(f"⚠️ Feature mismatch: Data has {current_features}, Model needs {TOTAL_FEATURES}")
            if current_features < TOTAL_FEATURES:
                diff = TOTAL_FEATURES - current_features
                print(f"🔧 Auto-padding with {diff} zeros per frame to match model...")
                X_vision = X[:, :, :17]
                X_sensor = X[:, :, 17:]
                
                # Create zeros for missing vision features
                zeros = np.zeros((X.shape[0], X.shape[1], diff))
                
                # Combine: [Vision(17) + Zeros(2) | Sensor(6)]
                X = np.concatenate([X_vision, zeros, X_sensor], axis=2)
                print(f"✅ New shape after padding: {X.shape}")

        
        # Normalize data
        print("🔄 Normalizing features...")
        X_flat = X.reshape(-1, X.shape[-1])
        
        # Handle Scaler Mismatch if necessary
        try:
            X_norm = scaler.transform(X_flat).reshape(X.shape)
        except ValueError as e:
            print(f"⚠️ Scaler mismatch: {e}")
            print("🔄 Attempting to fit new scaler (Just for testing, results might be inaccurate)...")
            from sklearn.preprocessing import StandardScaler
            temp_scaler = StandardScaler()
            X_norm = temp_scaler.fit_transform(X_flat).reshape(X.shape)

        print(f"✅ Normalized to range [{X_norm.min():.3f}, {X_norm.max():.3f}]")
        
        # Run inference
        print("🤖 Running model inference...")
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(DEVICE)
            predictions = model(X_tensor)
            scores = predictions.cpu().numpy().flatten()
        
        print(f"✅ Inference complete")
        print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"   Mean score: {scores.mean():.3f}")
        print(f"   Median score: {np.median(scores):.3f}")
        
        # Calculate metrics
        threshold = 0.80
        pass_count = np.sum(scores > threshold)
        pass_rate = float((pass_count / len(scores)) * 100)
        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        std_score = float(np.std(scores))
        
        # Determine result
        if pass_rate >= 80:
            result = "REAL"
            confidence = "HIGH"
        elif pass_rate >= 60:
            result = "REAL"
            confidence = "MEDIUM"
        elif pass_rate >= 40:
            result = "UNCERTAIN"
            confidence = "LOW"
        else:
            result = "FAKE"
            confidence = "HIGH" if pass_rate < 10 else "MEDIUM"
        
        response = {
            "status": "success",
            "result": result,
            "confidence": confidence,
            
           
            "pass_rate": round(pass_rate, 2),    
            "score": round(mean_score, 4),       
            
            "metrics": {
                "pass_rate": round(pass_rate, 2),
                "mean_score": round(mean_score, 4),
                "median_score": round(median_score, 4),
                "std_score": round(std_score, 4),
                "threshold": threshold
            },
            "analysis": {
                "total_frames": len(data),
                "sequences_analyzed": len(X),
                "high_confidence_frames": int(pass_count),
                "low_confidence_frames": int(len(scores) - pass_count)
            }
        }
        
        print()
        print("="*60)
        print(f"🎯 VERIFICATION RESULT")
        print("="*60)
        print(f"Result: {result} (Confidence: {confidence})")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Mean Score: {mean_score:.4f}")
        print(f"Sequences Analyzed: {len(X)}")
        print("="*60)
        print()
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print()
        print(f"❌ ERROR DURING VERIFICATION:")
        print(f"   {str(e)}")
        print()
        print("📋 Traceback:")
        print(traceback.format_exc())
        print()
        
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists("temp"):
                shutil.rmtree("temp", ignore_errors=True)
                print("🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")

# ================= 4. STARTUP =================
if __name__ == "__main__":
    print()
    print("="*60)
    print("🚀 STARTING ANTI-DEEPFAKE SERVER")
    print("="*60)
    print(f"📍 Host: 0.0.0.0")
    print(f"📍 Port: 8000")
    print(f"🌐 Access at: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print("="*60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")