import numpy as np
import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import distance
import io

# Initialize FastAPI app
app = FastAPI()

# Enable CORS so your website can talk to this API
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
NOSE_BRIDGE_IDX = 6
LEFT_TEMPLE_IDX = 234
RIGHT_TEMPLE_IDX = 454
IPD_MIN_MM = 45.0
IPD_MAX_MM = 80.0

# ---- Core Logic Functions ----

def detect_face_landmarks_and_iris(bgr_img):
    h, w = bgr_img.shape[:2]
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # Use the specific face_mesh solution we imported
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh_handler:
        results = face_mesh_handler.process(rgb_img)
        if not results.multi_face_landmarks:
            raise RuntimeError("No face detected.")
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        def iris_center(idxs):
            points = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in idxs], dtype=np.float32)
            (cx, cy), radius = cv2.minEnclosingCircle(points)
            return np.array([cx, cy], dtype=np.float32), radius
            
        left_center, left_r = iris_center(LEFT_IRIS_IDXS)
        right_center, right_r = iris_center(RIGHT_IRIS_IDXS)
        
        if left_r < 1 or right_r < 1:
            raise RuntimeError("Iris detection failed.")
            
        return {
            'left_iris': left_center,
            'right_iris': right_center,
            'ipd_px': float(np.linalg.norm(left_center - right_center)),
            'forehead_top': np.array([face_landmarks[FOREHEAD_TOP_IDX].x * w, face_landmarks[FOREHEAD_TOP_IDX].y * h]),
        }

def estimate_card_region_from_face(face_data, img_shape):
    h, w = img_shape[:2]
    ipd_px = face_data['ip
