import pandas as pd
import numpy as np
import cv2
import os
import mediapipe as mp
from scipy.interpolate import interp1d
from tqdm import tqdm

# ================= ⚙️ CONFIGURATION =================
ROOT_DATA_FOLDER = 'Dataset'   
OUTPUT_CSV = 'output_dataset.csv'
GENERATE_FAKE_DATA = False
FAKE_DATA_USING_REAR_CAMERA = True  

# Performance Tuning
FLOW_RESIZE_WIDTH = 240  
FLOW_SKIP_FRAMES = 1     
MIN_IMAGES_REQUIRED = 5 # จำนวนรูปขั้นต่ำต่อ Session
# ====================================================

def read_image_safe(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def calculate_dual_optical_flow(prev_gray, curr_gray, face_landmarks, img_w, img_h):
    if prev_gray is None: return 0.0, 0.0, 0.0, 0.0

    scale = FLOW_RESIZE_WIDTH / img_w
    target_h = int(img_h * scale)
    
    if target_h <= 0 or FLOW_RESIZE_WIDTH <= 0: return 0.0, 0.0, 0.0, 0.0

    prev_small = cv2.resize(prev_gray, (FLOW_RESIZE_WIDTH, target_h))
    curr_small = cv2.resize(curr_gray, (FLOW_RESIZE_WIDTH, target_h))

    mask_bg = np.ones_like(prev_small, dtype=np.uint8) * 255
    mask_fg = np.zeros_like(prev_small, dtype=np.uint8)
    
    face_points = []
    if face_landmarks:
        for lm in face_landmarks:
            face_points.append((int(lm.x * FLOW_RESIZE_WIDTH), int(lm.y * target_h)))
    
    if face_points:
        hull = cv2.convexHull(np.array(face_points))
        cv2.fillConvexPoly(mask_bg, hull, 0)   
        cv2.fillConvexPoly(mask_fg, hull, 255) 

    def get_flow_from_mask(mask_in):
        if cv2.countNonZero(mask_in) < 20: return 0.0, 0.0
        p0 = cv2.goodFeaturesToTrack(prev_small, mask=mask_in, maxCorners=50, qualityLevel=0.05, minDistance=5)
        if p0 is None or len(p0) < 3: return 0.0, 0.0
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, p0, None, winSize=(15, 15), maxLevel=2)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) == 0: return 0.0, 0.0
            dxs = good_new[:, 0] - good_old[:, 0]
            dys = good_new[:, 1] - good_old[:, 1]
            return np.mean(dxs), np.mean(dys)
        return 0.0, 0.0

    bg_dx, bg_dy = get_flow_from_mask(mask_bg)
    fg_dx, fg_dy = get_flow_from_mask(mask_fg)
    return bg_dx, bg_dy, fg_dx, fg_dy

def process_session(session_path):
    session_name = os.path.basename(session_path)
    
    img_folder = os.path.join(session_path, 'Camera')
    gyro_path = os.path.join(session_path, 'Gyroscope.csv')
    accel_path = os.path.join(session_path, 'Accelerometer.csv')

    # 1. Load CSV
    try:
        df_gyro = pd.read_csv(gyro_path)
        df_accel = pd.read_csv(accel_path)
    except Exception as e:
        # print(f"  ❌ CSV Error: {session_name}") 
        return []

    if len(df_gyro) < 10 or len(df_accel) < 10: 
        # print(f"  ❌ Short CSV: {session_name}")
        return []

    # 2. List Images (ใช้ os.listdir แทน glob เพื่อรองรับภาษาไทย)
    try:
        all_files = os.listdir(img_folder)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = sorted([os.path.join(img_folder, f) for f in all_files if f.lower().endswith(valid_exts)])
    except Exception as e:
        # print(f"  ❌ Path Error: {session_name}")
        return []

    if len(image_files) < MIN_IMAGES_REQUIRED:
        # print(f"  ⚠️ Too few images ({len(image_files)}): {session_name}")
        return []

    # Interpolate Sensors
    start_t = df_gyro['seconds_elapsed'].min()
    end_t = df_gyro['seconds_elapsed'].max()
    fg_x = interp1d(df_gyro['seconds_elapsed'], df_gyro['x'], fill_value="extrapolate")
    fg_y = interp1d(df_gyro['seconds_elapsed'], df_gyro['y'], fill_value="extrapolate")
    fg_z = interp1d(df_gyro['seconds_elapsed'], df_gyro['z'], fill_value="extrapolate")
    fa_x = interp1d(df_accel['seconds_elapsed'], df_accel['x'], fill_value="extrapolate")
    fa_y = interp1d(df_accel['seconds_elapsed'], df_accel['y'], fill_value="extrapolate")
    fa_z = interp1d(df_accel['seconds_elapsed'], df_accel['z'], fill_value="extrapolate")

    timestamps = np.linspace(start_t, end_t, len(image_files))
    session_data = []
    
    # Fix Session Name duplication
    if session_name.lower() in ['camera', 'data', 'session']:
        parent = os.path.basename(os.path.dirname(session_path))
        session_name = f"{parent}_{session_name}"

    is_physical_fake = 'fake' in session_path.lower()
    duration = end_t - start_t
    
    mp_face_mesh = mp.solutions.face_mesh
    prev_gray = None
    prev_landmarks = None 
    c_bg_x, c_bg_y, c_fg_x, c_fg_y = 0.0, 0.0, 0.0, 0.0
    
    processed_count = 0

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for i, img_path in enumerate(image_files):
            ts = timestamps[i]
            image = read_image_safe(img_path)
            
            if image is None: continue
            
            processed_count += 1
            curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = curr_gray.shape
            
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            face_found = False
            lm_list = []
            curr_feats = np.zeros(15) 
            deltas = np.zeros(15) 

            if results.multi_face_landmarks:
                face_found = True
                lm = results.multi_face_landmarks[0].landmark
                lm_list = lm 
                
                indices = [1, 33, 263, 234, 454]
                temp_feats = []
                for idx in indices:
                    temp_feats.extend([lm[idx].x, lm[idx].y, lm[idx].z])
                curr_feats = np.array(temp_feats)

                if prev_landmarks is not None:
                    deltas = curr_feats - prev_landmarks
                prev_landmarks = curr_feats

            if face_found and prev_gray is not None:
                if i % FLOW_SKIP_FRAMES == 0:
                    c_bg_x, c_bg_y, c_fg_x, c_fg_y = calculate_dual_optical_flow(prev_gray, curr_gray, lm_list, w, h)
            else:
                if not face_found:
                    c_bg_x, c_bg_y, c_fg_x, c_fg_y = 0.0, 0.0, 0.0, 0.0

            prev_gray = curr_gray.copy()

            if is_physical_fake and FAKE_DATA_USING_REAR_CAMERA:
                c_bg_x = -c_bg_x; c_fg_x = -c_fg_x
                deltas[0] = -deltas[0]; deltas[3] = -deltas[3]
                deltas[6] = -deltas[6]; deltas[9] = -deltas[9]; deltas[12] = -deltas[12]

            if face_found and i > 0: 
                current_label = 0 if is_physical_fake else 1
                row_real = {
                    'session': session_name, 'timestamp': ts,
                    'd_nose_x': deltas[0], 'd_nose_y': deltas[1], 'd_nose_z': deltas[2],
                    'd_leye_x': deltas[3], 'd_leye_y': deltas[4], 'd_leye_z': deltas[5],
                    'd_reye_x': deltas[6], 'd_reye_y': deltas[7], 'd_reye_z': deltas[8],
                    'd_lear_x': deltas[9], 'd_lear_y': deltas[10], 'd_lear_z': deltas[11],
                    'd_rear_x': deltas[12], 'd_rear_y': deltas[13], 'd_rear_z': deltas[14],
                    'bg_flow_x': c_bg_x, 'bg_flow_y': c_bg_y, 'fg_flow_x': c_fg_x, 'fg_flow_y': c_fg_y,
                    'gyro_x': fg_x(ts), 'gyro_y': fg_y(ts), 'gyro_z': fg_z(ts),
                    'accel_x': fa_x(ts), 'accel_y': fa_y(ts), 'accel_z': fa_z(ts),
                    'label': current_label
                }
                session_data.append(row_real)

                if GENERATE_FAKE_DATA and not is_physical_fake:
                    row_fake = row_real.copy()
                    row_fake['label'] = 0 
                    shift = np.random.uniform(2.0, max(2.0, duration - 1.0))
                    fake_ts = ts + shift
                    if fake_ts > end_t: fake_ts -= duration
                    row_fake['gyro_x'] = fg_x(fake_ts); row_fake['gyro_y'] = fg_y(fake_ts); row_fake['gyro_z'] = fg_z(fake_ts)
                    row_fake['accel_x'] = fa_x(fake_ts); row_fake['accel_y'] = fa_y(fake_ts); row_fake['accel_z'] = fa_z(fake_ts)
                    session_data.append(row_fake)
    
    # Debug: ถ้าประมวลผลแล้วไม่ได้อะไรเลย ให้ฟ้อง
    if len(session_data) == 0 and processed_count > 0:
        # print(f"  ⚠️ Processed {processed_count} images but got NO data (Face not found?): {session_name}")
        pass
    elif processed_count == 0:
        print(f"  ❌ Read 0 images (Check encoding/corruption): {session_name}")

    return session_data

def main():
    print(f"🚀 Scanning folder: {ROOT_DATA_FOLDER} (V22: Windows Thai Path Fix)")
    if not os.path.exists(ROOT_DATA_FOLDER):
        print("❌ Folder not found!"); return

    valid_sessions = []
    print("🔍 Listing sessions...")
    for root, dirs, files in os.walk(ROOT_DATA_FOLDER):
        if 'Camera' in dirs and 'Gyroscope.csv' in files and 'Accelerometer.csv' in files:
            valid_sessions.append(root)

    print(f"✅ Found {len(valid_sessions)} valid sessions.")
    master_data = []
    
    # ใช้ tqdm และรับค่า return เพื่อดูว่ามีอันไหนข้ามบ้าง
    for session in tqdm(valid_sessions):
        data = process_session(session)
        master_data.extend(data)
        
    if master_data:
        df = pd.DataFrame(master_data)
        cols = ['session', 'timestamp', 'label'] + [c for c in df.columns if c not in ['session', 'timestamp', 'label']]
        df = df[cols]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ SUCCESS! Saved {len(df)} rows to {OUTPUT_CSV}")
        print(f"   Real Samples: {len(df[df['label']==1])}")
        print(f"   Fake Samples: {len(df[df['label']==0])}")
    else:
        print("\n❌ No data processed. Something is wrong with file reading.")

if __name__ == "__main__":
    main()