import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import distance
import io

# Initialize FastAPI
app = FastAPI()

# Enable CORS for your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
IPD_MIN_MM = 45.0
IPD_MAX_MM = 80.0

# ---- Logic Functions ----

def detect_face_and_iris(bgr_img):
    h, w = bgr_img.shape[:2]
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            raise RuntimeError("No face detected.")
        
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        def iris_center(idxs):
            points = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in idxs], dtype=np.float32)
            (cx, cy), radius = cv2.minEnclosingCircle(points)
            return np.array([cx, cy], dtype=np.float32), radius
            
        left_center, _ = iris_center(LEFT_IRIS_IDXS)
        right_center, _ = iris_center(RIGHT_IRIS_IDXS)
        
        return {
            'left_iris': left_center,
            'right_iris': right_center,
            'ipd_px': float(np.linalg.norm(left_center - right_center)),
        }

def process_card_and_ipd(img, face_data):
    h, w = img.shape[:2]
    ipd_px = face_data['ipd_px']
    eye_mid = (face_data['left_iris'] + face_data['right_iris']) / 2.0
    
    # Estimate Card ROI
    est_w = ipd_px * 1.35
    est_h = est_w / CARD_ASPECT
    roi_x1 = int(max(eye_mid[0] - est_w * 0.6, 0))
    roi_x2 = int(min(eye_mid[0] + est_w * 0.6, w-1))
    roi_y1 = int(max(eye_mid[1] - est_h * 1.2, 0))
    roi_y2 = int(min(eye_mid[1] + est_h * 0.2, h-1))
    
    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0: raise RuntimeError("Card search region invalid.")
    
    # Find Rectangles
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_card = None
    max_score = -1
    
    for c in contours:
        rect = cv2.minAreaRect(c)
