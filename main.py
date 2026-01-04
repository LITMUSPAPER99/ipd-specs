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
    allow_origins=["*"], # In production, replace with your actual domain
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
    #--------mp_face_mesh = mp.solutions.face_mesh
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
    ipd_px = face_data['ipd_px']
    eye_mid = (face_data['left_iris'] + face_data['right_iris']) / 2.0
    estimated_card_width_px = ipd_px * 1.35
    estimated_card_height_px = estimated_card_width_px / CARD_ASPECT
    card_center_x = eye_mid[0]
    card_center_y = eye_mid[1] - ipd_px * 1.2
    search_margin_x = estimated_card_width_px * 0.6
    search_margin_y = estimated_card_height_px * 1.0
    roi_x1 = int(max(card_center_x - search_margin_x, 0))
    roi_x2 = int(min(card_center_x + search_margin_x, w-1))
    roi_y1 = int(max(card_center_y - search_margin_y, 0))
    roi_y2 = int(min(card_center_y + search_margin_y, h-1))
    return {
        'roi': (roi_x1, roi_y1, roi_x2, roi_y2),
        'expected_center': (card_center_x, card_center_y),
        'expected_width_px': estimated_card_width_px,
        'expected_height_px': estimated_card_height_px,
    }

def find_rectangular_contours(roi, min_area_px):
    all_contours = []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours1)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 70, 255]))
    contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours2)
    return [c for c in all_contours if cv2.contourArea(c) >= min_area_px]

def score_card_candidate(contour, expected_data, roi_offset):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w_rect, h_rect), angle = rect
    if w_rect == 0 or h_rect == 0: return None, None
    width_px, height_px = max(w_rect, h_rect), min(w_rect, h_rect)
    aspect = width_px / height_px
    roi_x1, roi_y1 = roi_offset
    center_full = np.array([cx + roi_x1, cy + roi_y1])
    aspect_score = np.exp(-abs(aspect - CARD_ASPECT) * 2.0)
    size_ratio = width_px / expected_data['expected_width_px']
    size_score = np.exp(-abs(np.log(size_ratio)) * 1.5)
    total_score = (aspect_score * 3.0 + size_score * 3.0)
    box = cv2.boxPoints(rect)
    box[:, 0] += roi_x1
    box[:, 1] += roi_y1
    return total_score, {'box': box, 'width_px': float(width_px), 'aspect': float(aspect), 'score': float(total_score)}

def apply_perspective_correction(bgr_img, card_box):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        return rect
    rect = order_points(card_box)
    (tl, tr, br, bl) = rect
    dst_width = max(int(distance.euclidean(br, bl)), int(distance.euclidean(tr, tl)))
    dst_height = int(dst_width / CARD_ASPECT)
    dst = np.array([[0, 0], [dst_width-1, 0], [dst_width-1, dst_height-1], [0, dst_height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr_img, M, (dst_width, dst_height)), dst_width

def compute_ipd_mm(ipd_pixels, card_width_px):
    pixels_per_mm = card_width_px / CARD_WIDTH_MM
    ipd_mm_uncorrected = ipd_pixels / pixels_per_mm
    depth_factor = (400.0 + FOREHEAD_TO_IRIS_DEPTH_MM) / 400.0
    return {'ipd_mm': float(ipd_mm_uncorrected * depth_factor)}

# ---- API Endpoints ----

@app.get("/")
def home():
    return {"message": "Specsit IPD API is running", "docs": "/docs"}

@app.post("/measure-ipd")
async def measure_ipd(file: UploadFile = File(...)):
    # 1. Read Image
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Invalid image")

    try:
        # 2. Process
        face_data = detect_face_landmarks_and_iris(img)
        expected = estimate_card_region_from_face(face_data, img.shape)
        min_area = expected['expected_width_px'] * expected['expected_height_px'] * 0.3
        roi = img[expected['roi'][1]:expected['roi'][3], expected['roi'][0]:expected['roi'][2]]
        contours = find_rectangular_contours(roi, min_area)
        
        candidates = []
        for c in contours:
            s, cand = score_card_candidate(c, expected, (expected['roi'][0], expected['roi'][1]))
            if cand: candidates.append((s, cand))
        
        if not candidates: raise RuntimeError("No card found")
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_cand = candidates[0][1]
        
        warped, corrected_width = apply_perspective_correction(img, best_cand['box'])
        result = compute_ipd_mm(face_data['ipd_px'], corrected_width)

        return {
            "status": "success",
            "ipd_mm": round(result['ipd_mm'], 2),
            "confidence_score": round(best_cand['score'], 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
