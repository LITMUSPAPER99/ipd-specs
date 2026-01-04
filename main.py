import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import distance
import io

# Initialize FastAPI
app = FastAPI(title="Specsit IPD Measurement API")

# Enable CORS so your website can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Constants ----
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT = CARD_WIDTH_MM / CARD_HEIGHT_MM
FOREHEAD_TO_IRIS_DEPTH_MM = 25.0
LEFT_IRIS_IDXS = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477]
FOREHEAD_TOP_IDX = 10
NOSE_BRIDGE_IDX = 6
IPD_MIN_MM = 45.0
IPD_MAX_MM = 80.0

# ---- Core Logic Functions ----

def detect_face_landmarks_and_iris(bgr_img):
    h, w = bgr_img.shape[:2]
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected.")
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        def iris_center(idxs):
            points = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in idxs], dtype=np.float32)
            (cx, cy), _ = cv2.minEnclosingCircle(points)
            return np.array([cx, cy], dtype=np.float32)

        l_center = iris_center(LEFT_IRIS_IDXS)
        r_center = iris_center(RIGHT_IRIS_IDXS)
        
        return {
            'left_iris': l_center, 
            'right_iris': r_center,
            'ipd_px': float(np.linalg.norm(l_center - r_center)),
            'forehead_top': np.array([face_landmarks[FOREHEAD_TOP_IDX].x * w, face_landmarks[FOREHEAD_TOP_IDX].y * h])
        }

def estimate_card_region(face_data, img_shape):
    h, w = img_shape[:2]
    ipd_px = face_data['ipd_px']
    eye_mid = (face_data['left_iris'] + face_data['right_iris']) / 2.0
    est_w_px = ipd_px * 1.35
    est_h_px = est_w_px / CARD_ASPECT
    
    card_center_x = eye_mid[0]
    card_center_y = eye_mid[1] - ipd_px * 1.2
    
    roi_x1 = int(max(card_center_x - est_w_px * 0.6, 0))
    roi_x2 = int(min(card_center_x + est_w_px * 0.6, w-1))
    roi_y1 = int(max(card_center_y - est_h_px * 1.0, 0))
    roi_y2 = int(min(card_center_y + est_h_px * 1.0, h-1))
    
    return {'roi': (roi_x1, roi_y1, roi_x2, roi_y2), 'est_w': est_w_px}

def get_best_card(bgr_img, face_data):
    expected = estimate_card_region(face_data, bgr_img.shape)
    x1, y1, x2, y2 = expected['roi']
    roi = bgr_img[y1:y2, x1:x2]
    
    if roi.size == 0: raise ValueError("Invalid card search region.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_candidate = None
    max_score = -1

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (wr, hr), angle = rect
        if wr == 0 or hr == 0: continue
        
        w_px, h_px = max(wr, hr), min(wr, hr)
        aspect = w_px / h_px
        
        aspect_score = np.exp(-abs(aspect - CARD_ASPECT) * 2.0)
        size_score = np.exp(-abs(np.log(w_px / expected['est_w'])) * 1.5)
        total_score = aspect_score + size_score
        
        if total_score > max_score:
            max_score = total_score
            box = cv2.boxPoints(rect)
            box[:, 0] += x1
            box[:, 1] += y1
            best_candidate = {'box': box, 'score': total_score}

    if not best_candidate or max_score < 1.0:
        raise ValueError("Could not detect reference card clearly.")
    
    return best_candidate

def calculate_final_ipd(face_data, card_data):
    pts = card_data['box'].astype("float32")
    # Order points: TL, TR, BR, BL
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    
    card_w_px = max(distance.euclidean(rect[0], rect[1]), distance.euclidean(rect[2], rect[3]))
    px_per_mm = card_w_px / CARD_WIDTH_MM
    
    # Depth correction
    depth_factor = (400.0 + FOREHEAD_TO_IRIS_DEPTH_MM) / 400.0
    return (face_data['ipd_px'] / px_per_mm) * depth_factor

# ---- Endpoints ----

@app.post("/measure")
async def measure_ipd(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # Execute Pipeline
        face_info = detect_face_landmarks_and_iris(img)
        card_info = get_best_card(img, face_info)
        final_ipd = calculate_final_ipd(face_info, card_info)
        
        confidence = "HIGH" if card_info['score'] > 1.5 else "MEDIUM"
        
        return {
            "status": "success",
            "ipd_mm": round(final_ipd, 2),
            "confidence": confidence,
            "warnings": [] if (IPD_MIN_MM <= final_ipd <= IPD_MAX_MM) else ["IPD outside typical range"]
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def health_check():
    return {"status": "online", "message": "Specsit IPD API is active."}
