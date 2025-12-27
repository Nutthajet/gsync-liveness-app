import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
import os
import glob
import mediapipe as mp
import joblib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt # 🔥 ใช้สำหรับวาดกราฟ

# ================= ⚙️ CONFIGURATION =================
# 👇 แก้ Path ตรงนี้ให้เป็นโฟลเดอร์ที่คุณต้องการทดสอบ
TEST_SESSION_PATH = 'old_test/4'
MODEL_PATH = 'kyc_model_best.pth'
SCALER_PATH = 'scaler.pkl'
SEQ_LEN = 30
FLOW_RESIZE_WIDTH = 240 # ต้องตรงกับตอน Train (ห้ามเปลี่ยน)
# ====================================================

# 1. Helper Function: คำนวณ Optical Flow
def get_background_flow(prev_gray, curr_gray, face_landmarks, img_w, img_h):
    if prev_gray is None: return 0.0, 0.0

    # ย่อภาพเพื่อความเร็ว (ให้เหมือนตอน Train)
    scale = FLOW_RESIZE_WIDTH / img_w
    target_h = int(img_h * scale)
    
    prev_small = cv2.resize(prev_gray, (FLOW_RESIZE_WIDTH, target_h))
    curr_small = cv2.resize(curr_gray, (FLOW_RESIZE_WIDTH, target_h))

    # Masking Face
    mask = np.ones_like(prev_small, dtype=np.uint8) * 255
    face_points = []
    for lm in face_landmarks:
        face_points.append((int(lm.x * FLOW_RESIZE_WIDTH), int(lm.y * target_h)))
    
    if face_points:
        hull = cv2.convexHull(np.array(face_points))
        cv2.fillConvexPoly(mask, hull, 0) 

    # หาจุด Tracking
    p0 = cv2.goodFeaturesToTrack(prev_small, mask=mask, maxCorners=20, qualityLevel=0.2, minDistance=5)

    if p0 is None or len(p0) < 5: return 0.0, 0.0

    # Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, p0, None, winSize=(15, 15), maxLevel=2)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) == 0: return 0.0, 0.0

        dxs = good_new[:, 0] - good_old[:, 0]
        dys = good_new[:, 1] - good_old[:, 1]
        
        return np.mean(dxs), np.mean(dys)
    
    return 0.0, 0.0

# 2. Model Structure (ต้องรับ 23 Features)
class AntiDeepfakeModelPro(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision Branch: 15 (Face) + 2 (Flow) = 17
        self.gru_vision = nn.GRU(17, 64, batch_first=True, bidirectional=True)
        # Sensor Branch: 6
        self.gru_sensor = nn.GRU(6, 64, batch_first=True, bidirectional=True)
        
        self.bn = nn.BatchNorm1d(256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        vis, sen = x[:, :, :17], x[:, :, 17:]
        
        _, h_v = self.gru_vision(vis)
        _, h_s = self.gru_sensor(sen)
        
        feat_v = torch.cat((h_v[-2], h_v[-1]), dim=1)
        feat_s = torch.cat((h_s[-2], h_s[-1]), dim=1)
        
        combined = torch.cat((feat_v, feat_s), dim=1)
        return self.classifier(self.bn(combined))

# 3. Feature Extraction
def extract_features_from_session(session_path):
    print(f"📂 Reading session: {session_path}")
    img_folder = os.path.join(session_path, 'Camera')
    gyro_path = os.path.join(session_path, 'Gyroscope.csv')
    accel_path = os.path.join(session_path, 'Accelerometer.csv')

    if not (os.path.exists(img_folder) and os.path.exists(gyro_path) and os.path.exists(accel_path)):
        print(f"❌ Error: Files missing in {session_path}")
        return None

    try:
        df_gyro = pd.read_csv(gyro_path)
        df_accel = pd.read_csv(accel_path)
    except Exception as e:
        print(f"❌ CSV Read Error: {e}")
        return None

    # Interpolate Sensors
    start_t = df_gyro['seconds_elapsed'].min()
    end_t = df_gyro['seconds_elapsed'].max()
    
    fg_x = interp1d(df_gyro['seconds_elapsed'], df_gyro['x'], fill_value="extrapolate")
    fg_y = interp1d(df_gyro['seconds_elapsed'], df_gyro['y'], fill_value="extrapolate")
    fg_z = interp1d(df_gyro['seconds_elapsed'], df_gyro['z'], fill_value="extrapolate")
    fa_x = interp1d(df_accel['seconds_elapsed'], df_accel['x'], fill_value="extrapolate")
    fa_y = interp1d(df_accel['seconds_elapsed'], df_accel['y'], fill_value="extrapolate")
    fa_z = interp1d(df_accel['seconds_elapsed'], df_accel['z'], fill_value="extrapolate")

    image_files = sorted(glob.glob(os.path.join(img_folder, '*.*')))
    if len(image_files) < 10:
        print("❌ Not enough images.")
        return None

    timestamps = np.linspace(start_t, end_t, len(image_files))
    extracted_rows = []

    print("⏳ Extracting features (Face + Flow + Sensors)...")
    mp_face_mesh = mp.solutions.face_mesh
    
    prev_gray = None 

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        
        for i, img_path in enumerate(image_files):
            ts = timestamps[i]
            image = cv2.imread(img_path)
            if image is None: continue
            
            # Prepare Flow
            curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = curr_gray.shape

            # Prepare MediaPipe
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)
            
            if not results.multi_face_landmarks:
                prev_gray = curr_gray.copy()
                continue

            lm = results.multi_face_landmarks[0].landmark
            
            # Calculate Flow
            flow_x, flow_y = 0.0, 0.0
            if prev_gray is not None:
                flow_x, flow_y = get_background_flow(prev_gray, curr_gray, lm, w, h)
            
            prev_gray = curr_gray.copy()

            # Assemble Vector (23 values)
            row = [
                lm[1].x, lm[1].y, lm[1].z,       # Nose
                lm[33].x, lm[33].y, lm[33].z,    # Left Eye
                lm[263].x, lm[263].y, lm[263].z, # Right Eye
                lm[234].x, lm[234].y, lm[234].z, # Left Ear
                lm[454].x, lm[454].y, lm[454].z, # Right Ear
                flow_x, flow_y,                  # Flow
                fg_x(ts), fg_y(ts), fg_z(ts),    # Gyro
                fa_x(ts), fa_y(ts), fa_z(ts)     # Accel
            ]
            extracted_rows.append(row)

    return np.array(extracted_rows)

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Model/Scaler not found! Run train_model.py first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Loading model on {device}...")
    
    model = AntiDeepfakeModelPro().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"❌ Model Load Error: {e}"); return
    
    model.eval()
    scaler = joblib.load(SCALER_PATH)

    # Extract Features
    raw_data = extract_features_from_session(TEST_SESSION_PATH)
    
    if raw_data is None or len(raw_data) < SEQ_LEN:
        print("❌ Video too short or failed to process.")
        return

    # Create Sequences (Stride=1 for high resolution analysis)
    sequences = []
    for i in range(0, len(raw_data) - SEQ_LEN, 1):
        sequences.append(raw_data[i : i+SEQ_LEN])
    
    if len(sequences) == 0: return

    X = np.array(sequences) 
    
    # Normalize
    N, L, F = X.shape
    try:
        X_norm = scaler.transform(X.reshape(-1, F)).reshape(N, L, F)
    except ValueError:
        print(f"❌ Scaler Mismatch: Expected {scaler.n_features_in_} features, got {F}")
        return

    # Predict
    inputs = torch.FloatTensor(X_norm).to(device)
    print("🔮 Predicting...")
    with torch.no_grad():
        outputs = model(inputs)
        scores = outputs.cpu().numpy().flatten()
    
    # ==========================================
    # 🧠 SMART DECISION LOGIC (Pass Rate)
    # ==========================================
    
    # เกณฑ์คะแนนขั้นต่ำที่จะนับว่า "เสี้ยววินาทีนี้คือ Real"
    FRAME_THRESHOLD = 0.85 
    
    # นับจำนวนช่วงเวลาที่ผ่านเกณฑ์
    pass_count = np.sum(scores > FRAME_THRESHOLD)
    total_frames = len(scores)
    pass_rate = (pass_count / total_frames) * 100
    
    avg_score = np.mean(scores) * 100
    min_score = np.min(scores) * 100
    max_score = np.max(scores) * 100
    
    print("\n" + "="*50)
    print(f"📊 REPORT FOR: {os.path.basename(TEST_SESSION_PATH)}")
    print("="*50)
    print(f"🔹 Pass Rate         : {pass_rate:.2f}%  (Confidence Duration)")
    print(f"🔹 Average Score     : {avg_score:.2f}%")
    print(f"🔹 Min / Max Score   : {min_score:.2f}% - {max_score:.2f}%")
    print("-" * 50)
    
    # ตัดสินผลลัพธ์จาก Pass Rate (ถ้าเกิน 60% ของเวลาทั้งหมดถือว่าผ่าน)
    if pass_rate > 60.0:
        print("✅ RESULT: REAL PERSON (มนุษย์จริง)")
        print("   (ระบบพบความสัมพันธ์ของภาพและเซนเซอร์อย่างต่อเนื่อง)")
    elif pass_rate < 20.0:
        print("❌ RESULT: FAKE / REPLAY ATTACK (ปลอม)")
        print("   (ไม่พบความสัมพันธ์ที่สอดคล้องกัน)")
    else:
        print("⚠️ RESULT: UNCERTAIN (ไม่แน่ใจ)")
        print(f"   (คะแนนก้ำกึ่ง Pass Rate={pass_rate:.2f}%)")
    print("="*50 + "\n")

    # 📈 Plot Graph
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(scores, label='Confidence Score', color='#1f77b4', linewidth=1.5)
        plt.axhline(y=FRAME_THRESHOLD, color='#2ca02c', linestyle='--', label='Real Threshold (0.85)')
        
        # ถมสีพื้นที่ใต้กราฟ (สีเขียวถ้าผ่าน, สีแดงถ้าไม่ผ่าน)
        plt.fill_between(range(len(scores)), scores, where=(scores > FRAME_THRESHOLD), color='green', alpha=0.1)
        plt.fill_between(range(len(scores)), scores, where=(scores <= FRAME_THRESHOLD), color='red', alpha=0.1)
        
        plt.title(f"AI Confidence Analysis: {os.path.basename(TEST_SESSION_PATH)}")
        plt.xlabel("Time Sequence (Frames)")
        plt.ylabel("Probability (0=Fake, 1=Real)")
        plt.legend()
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not plot graph: {e}")

if __name__ == "__main__":
    main()