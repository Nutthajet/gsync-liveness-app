import uvicorn
from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
import joblib
import mediapipe as mp
import os
import shutil
from scipy.interpolate import interp1d

# ================= 1. MODEL CONFIG =================
app = FastAPI()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'kyc_model_best.pth'
SCALER_PATH = 'scaler.pkl'
FLOW_RESIZE_WIDTH = 480
SEQ_LEN = 30

# Model Structure (เหมือนตอน Train เป๊ะๆ)
class AntiDeepfakeModelPro(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru_vision = nn.GRU(17, 64, batch_first=True, bidirectional=True)
        self.gru_sensor = nn.GRU(6, 64, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        vis, sen = x[:, :, :17], x[:, :, 17:]
        _, h_v = self.gru_vision(vis)
        _, h_s = self.gru_sensor(sen)
        feat_v = torch.cat((h_v[-2], h_v[-1]), dim=1)
        feat_s = torch.cat((h_s[-2], h_s[-1]), dim=1)
        return self.classifier(self.bn(torch.cat((feat_v, feat_s), dim=1)))

# Load Model & Tools
print("⏳ Loading Model & Scaler...")
model = AntiDeepfakeModelPro().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
scaler = joblib.load(SCALER_PATH)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
print("✅ Server Ready!")

# ================= 2. HELPER FUNCTIONS =================
def get_background_flow(prev_gray, curr_gray, face_landmarks, img_w, img_h):
    if prev_gray is None: return 0.0, 0.0
    scale = FLOW_RESIZE_WIDTH / img_w
    target_h = int(img_h * scale)
    prev_small = cv2.resize(prev_gray, (FLOW_RESIZE_WIDTH, target_h))
    curr_small = cv2.resize(curr_gray, (FLOW_RESIZE_WIDTH, target_h))
    
    mask = np.ones_like(prev_small, dtype=np.uint8) * 255
    face_points = [(int(lm.x * FLOW_RESIZE_WIDTH), int(lm.y * target_h)) for lm in face_landmarks]
    if face_points:
        cv2.fillConvexPoly(mask, cv2.convexHull(np.array(face_points)), 0)

    p0 = cv2.goodFeaturesToTrack(prev_small, mask=mask, maxCorners=20, qualityLevel=0.2, minDistance=5)
    if p0 is None or len(p0) < 5: return 0.0, 0.0
    
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, p0, None, winSize=(15,15), maxLevel=2)
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        if len(good_new) > 0:
            d = good_new - good_old
            return np.mean(d[:, 0]), np.mean(d[:, 1])
    return 0.0, 0.0

def process_data(video_path, gyro_path, accel_path):
    try:
        df_gyro = pd.read_csv(gyro_path)
        df_accel = pd.read_csv(accel_path)
    except: return None

    # Interpolation Functions
    t_g, t_a = df_gyro['seconds_elapsed'], df_accel['seconds_elapsed']
    fg = [interp1d(t_g, df_gyro[c], fill_value="extrapolate") for c in ['x','y','z']]
    fa = [interp1d(t_a, df_accel[c], fill_value="extrapolate") for c in ['x','y','z']]

    cap = cv2.VideoCapture(video_path)
    timestamps = np.linspace(t_g.min(), t_g.max(), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    extracted, prev_gray, idx = [], None, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        ts = timestamps[min(idx, len(timestamps)-1)]
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape
        results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            fx, fy = (0.0, 0.0) if prev_gray is None else get_background_flow(prev_gray, curr_gray, lm, w, h)
            
            row = [lm[1].x, lm[1].y, lm[1].z, lm[33].x, lm[33].y, lm[33].z,
                   lm[263].x, lm[263].y, lm[263].z, lm[234].x, lm[234].y, lm[234].z,
                   lm[454].x, lm[454].y, lm[454].z, fx, fy,
                   fg[0](ts), fg[1](ts), fg[2](ts), fa[0](ts), fa[1](ts), fa[2](ts)]
            extracted.append(row)
        
        prev_gray = curr_gray.copy()
        idx += 1
    cap.release()
    return np.array(extracted)

# ================= 3. API ENDPOINT =================
@app.post("/verify")
async def verify(video: UploadFile = File(...), gyroscope: UploadFile = File(...), accelerometer: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    v_path, g_path, a_path = f"temp/{video.filename}", f"temp/{gyroscope.filename}", f"temp/{accelerometer.filename}"
    
    # Save Files
    with open(v_path, "wb") as f: shutil.copyfileobj(video.file, f)
    with open(g_path, "wb") as f: shutil.copyfileobj(gyroscope.file, f)
    with open(a_path, "wb") as f: shutil.copyfileobj(accelerometer.file, f)
    
    try:
        data = process_data(v_path, g_path, a_path)
        if data is None or len(data) < SEQ_LEN:
            return {"status": "error", "message": "Processing failed or video too short"}
            
        # Create Sequences & Predict
        X = np.array([data[i:i+SEQ_LEN] for i in range(0, len(data)-SEQ_LEN)])
        if len(X) == 0: return {"status": "error"}
        
        X_norm = scaler.transform(X.reshape(-1, 23)).reshape(X.shape)
        with torch.no_grad():
            scores = model(torch.FloatTensor(X_norm).to(DEVICE)).cpu().numpy().flatten()
            
        pass_rate = (np.sum(scores > 0.85) / len(scores)) * 100
        result = "REAL" if pass_rate >= 60 else ("FAKE" if pass_rate < 20 else "UNCERTAIN")
        
        return {"status": "success", "result": result, "pass_rate": pass_rate}
        
    finally:
        shutil.rmtree("temp", ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)