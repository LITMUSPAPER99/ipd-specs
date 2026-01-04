from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import mediapipe as mp
import io

app = FastAPI()

# ---- Constants ----

CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT = CARD_WIDTH_MM / CARD_HEIGHT_MM

FOREHEAD_TO_IRIS_DEPTH_MM = 25.0

LEFT_IRIS_IDXS = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477]

# Key face landmarks for better card localization
FOREHEAD_TOP_IDX = 10
NOSE_BRIDGE_IDX = 6
LEFT_TEMPLE_IDX = 234
RIGHT_TEMPLE_IDX = 454

IPD_MIN_MM = 45.0
IPD_MAX_MM = 80.0

# ---- Iris & Face Landmark Detection ----

def detect_face_landmarks_and_iris(bgr_img):
    """
    Detect iris centres and key face landmarks for precise card localization.
    """
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
            raise RuntimeError("No face detected. Make sure the face is clearly visible.")

        if len(results.multi_face_landmarks) > 1:
            raise RuntimeError("More than one face detected. Use an image with a single subject.")

        face_landmarks = results.multi_face_landmarks[0].landmark

        # Iris centres
        def iris_center(idxs):
            points = np.array(
                [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in idxs],
                dtype=np.float32,
            )
            (cx, cy), radius = cv2.minEnclosingCircle(points)
            return np.array([cx, cy], dtype=np.float32), radius

        left_center, left_r = iris_center(LEFT_IRIS_IDXS)
        right_center, right_r = iris_center(RIGHT_IRIS_IDXS)

        if left_r < 1 or right_r < 1:
            raise RuntimeError("Iris detection failed - face too far or unclear.")

        # Key face points for card localization
        forehead_top = np.array([
            face_landmarks[FOREHEAD_TOP_IDX].x * w,
            face_landmarks[FOREHEAD_TOP_IDX].y * h
        ], dtype=np.float32)

        nose_bridge = np.array([
            face_landmarks[NOSE_BRIDGE_IDX].x * w,
            face_landmarks[NOSE_BRIDGE_IDX].y * h
        ], dtype=np.float32)

        left_temple = np.array([
            face_landmarks[LEFT_TEMPLE_IDX].x * w,
            face_landmarks[LEFT_TEMPLE_IDX].y * h
        ], dtype=np.float32)

        right_temple = np.array([
            face_landmarks[RIGHT_TEMPLE_IDX].x * w,
            face_landmarks[RIGHT_TEMPLE_IDX].y * h
        ], dtype=np.float32)

        return {
            'left_iris': left_center,
            'right_iris': right_center,
            'left_iris_r': left_r,
            'right_iris_r': right_r,
            'forehead_top': forehead_top,
            'nose_bridge': nose_bridge,
            'left_temple': left_temple,
            'right_temple': right_temple,
            'ipd_px': float(np.linalg.norm(left_center - right_center)),
        }

# ---- Geometry-Guided Card Detection ----

def estimate_card_region_from_face(face_data, img_shape):
    """
    Use face geometry to predict where the card SHOULD be.
    Returns a focused search region and expected card dimensions.
    """
    h, w = img_shape[:2]

    ipd_px = face_data['ipd_px']
    eye_mid = (face_data['left_iris'] + face_data['right_iris']) / 2.0
    forehead_top = face_data['forehead_top']

    # Estimate card width in pixels
    # Typical adult IPD: 60-65mm, card width: 85.6mm
    # So card width ≈ 1.3-1.4 × IPD in pixels
    estimated_card_width_px = ipd_px * 1.35
    estimated_card_height_px = estimated_card_width_px / CARD_ASPECT

    # Card should be centered horizontally above eyes
    card_center_x = eye_mid[0]

    # Card should be positioned between eye line and top of forehead
    # Typically: card bottom edge is ~0.8-1.0 × IPD above eye line
    card_center_y = eye_mid[1] - ipd_px * 1.2

    # Define search region (generous but focused)
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
    """
    Find rectangular contours using multiple preprocessing methods.
    Returns list of contours from all methods.
    """
    all_contours = []

    # Method 1: Canny edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours1)

    # Method 2: Color-based (white/light detection)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 140])
    upper = np.array([180, 70, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours2)

    # Method 3: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours3, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours3)

    # Filter by minimum area
    filtered = [c for c in all_contours if cv2.contourArea(c) >= min_area_px]

    return filtered

def score_card_candidate(contour, expected_data, roi_offset):
    """
    Score a contour as a potential card based on:
    1. Size match with expected dimensions
    2. Aspect ratio match
    3. Position match with expected location
    4. Rectangularity
    """
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w_rect, h_rect), angle = rect

    if w_rect == 0 or h_rect == 0:
        return None, None

    width_px = max(w_rect, h_rect)
    height_px = min(w_rect, h_rect)
    aspect = width_px / height_px

    # Convert center from ROI to full image coordinates
    roi_x1, roi_y1 = roi_offset
    center_full = np.array([cx + roi_x1, cy + roi_y1])

    # 1. Aspect ratio score (closer to 1.586 is better)
    aspect_error = abs(aspect - CARD_ASPECT)
    aspect_score = np.exp(-aspect_error * 2.0)  # 0 to 1, higher is better

    # 2. Size match score
    expected_width = expected_data['expected_width_px']
    size_ratio = width_px / expected_width if expected_width > 0 else 0
    size_score = np.exp(-abs(np.log(size_ratio)) * 1.5) if size_ratio > 0 else 0

    # 3. Position score (how close to expected center)
    expected_center = np.array(expected_data['expected_center'])
    distance_from_expected = np.linalg.norm(center_full - expected_center)
    position_score = np.exp(-distance_from_expected / expected_width * 2.0)

    # 4. Rectangularity score
    area_contour = cv2.contourArea(contour)
    area_rect = width_px * height_px
    rectangularity = area_contour / area_rect if area_rect > 0 else 0
    rectangularity_score = rectangularity

    # 5. Orientation score (card should be roughly horizontal)
    # Angle should be close to 0 or 90 degrees
    angle_normalized = abs(angle % 90)
    if angle_normalized > 45:
        angle_normalized = 90 - angle_normalized
    orientation_score = np.exp(-angle_normalized / 15.0)

    # Combined score (weighted)
    total_score = (
        aspect_score * 3.0 +
        size_score * 3.0 +
        position_score * 2.0 +
        rectangularity_score * 1.5 +
        orientation_score * 1.0
    )

    # Get box points
    box = cv2.boxPoints(rect)
    box = box.astype(np.float32)
    box[:, 0] += roi_x1
    box[:, 1] += roi_y1

    candidate = {
        'box': box,
        'width_px': float(width_px),
        'height_px': float(height_px),
        'aspect': float(aspect),
        'center': center_full,
        'angle': float(angle),
        'rectangularity': float(rectangularity),
        'scores': {
            'aspect': float(aspect_score),
            'size': float(size_score),
            'position': float(position_score),
            'rectangularity': float(rectangularity_score),
            'orientation': float(orientation_score),
            'total': float(total_score),
        }
    }

    return total_score, candidate

def detect_card_with_face_guidance(bgr_img, face_data):
    """
    Detect card using face geometry to guide the search.
    """
    h, w = bgr_img.shape[:2]

    # Get expected card region
    expected = estimate_card_region_from_face(face_data, bgr_img.shape)
    roi_x1, roi_y1, roi_x2, roi_y2 = expected['roi']

    print(f"[INFO] Expected card position: center=({expected['expected_center'][0]:.0f}, {expected['expected_center'][1]:.0f}), "
          f"size=({expected['expected_width_px']:.0f} × {expected['expected_height_px']:.0f}) px")
    print(f"[INFO] Search region: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")

    # Extract ROI
    roi = bgr_img[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]

    if roi_h < 10 or roi_w < 10:
        raise RuntimeError("Invalid ROI - face detection may be inaccurate")

    # Find contours
    min_area = expected['expected_width_px'] * expected['expected_height_px'] * 0.3
    contours = find_rectangular_contours(roi, min_area)

    print(f"[INFO] Found {len(contours)} potential contours")

    if len(contours) == 0:
        raise RuntimeError(
            "No contours found in expected card region. "
            "Ensure card is visible, well-lit, and contrasts with background."
        )

    # Score all candidates
    candidates = []
    for contour in contours:
        score, candidate = score_card_candidate(contour, expected, (roi_x1, roi_y1))
        if candidate is not None:
            candidates.append((score, candidate))

    if len(candidates) == 0:
        raise RuntimeError("No valid card candidates found")

    # Sort by score and select best
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_candidate = candidates[0]

    print(f"[INFO] Best candidate score: {best_score:.2f}")
    print(f"[INFO]   - Aspect ratio: {best_candidate['aspect']:.3f} (target: {CARD_ASPECT:.3f})")
    print(f"[INFO]   - Size: {best_candidate['width_px']:.1f} × {best_candidate['height_px']:.1f} px")
    print(f"[INFO]   - Rectangularity: {best_candidate['rectangularity']:.3f}")
    print(f"[INFO]   - Score breakdown: {best_candidate['scores']}")

    # Quality assessment
    quality = {
        'is_good': best_score > 6.0 and best_candidate['scores']['aspect'] > 0.7,
        'score': best_score,
        'aspect_match': best_candidate['scores']['aspect'],
        'size_match': best_candidate['scores']['size'],
        'position_match': best_candidate['scores']['position'],
    }

    return best_candidate, expected, quality, candidates

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_perspective_correction(bgr_img, card_box):
    """Apply perspective transform to get frontal view of card"""
    rect = order_points(card_box)
    (tl, tr, br, bl) = rect

    width_a = distance.euclidean(br, bl)
    width_b = distance.euclidean(tr, tl)
    max_width = max(int(width_a), int(width_b))

    height_a = distance.euclidean(tr, br)
    height_b = distance.euclidean(tl, bl)
    max_height = max(int(height_a), int(height_b))

    dst_width = max_width
    dst_height = int(dst_width / CARD_ASPECT)

    dst = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(bgr_img, M, (dst_width, dst_height))

    # Check perspective quality
    edge_lengths = [distance.euclidean(rect[i], rect[(i+1)%4]) for i in range(4)]
    edge_lengths = np.array(edge_lengths)
    long_edges = np.sort(edge_lengths)[-2:]
    short_edges = np.sort(edge_lengths)[:2]

    long_ratio = long_edges[0] / long_edges[1] if long_edges[1] > 0 else 0
    short_ratio = short_edges[0] / short_edges[1] if short_edges[1] > 0 else 0

    perspective_quality = {
        'long_edge_ratio': float(long_ratio),
        'short_edge_ratio': float(short_ratio),
        'is_good': long_ratio > 0.85 and short_ratio > 0.85,
        'corrected_width': float(dst_width),
        'corrected_height': float(dst_height),
    }

    return warped, dst_width, perspective_quality

# ---- IPD Calculation ----

def compute_ipd_mm(ipd_pixels, card_width_px):
    """
    Convert IPD from pixels to mm with depth correction.
    """
    pixels_per_mm = card_width_px / CARD_WIDTH_MM
    ipd_mm_uncorrected = ipd_pixels / pixels_per_mm

    # Depth correction
    TYPICAL_SELFIE_DISTANCE_MM = 400.0
    depth_factor = (TYPICAL_SELFIE_DISTANCE_MM + FOREHEAD_TO_IRIS_DEPTH_MM) / TYPICAL_SELFIE_DISTANCE_MM
    ipd_mm_corrected = ipd_mm_uncorrected * depth_factor

    return {
        'ipd_mm': float(ipd_mm_corrected),
        'ipd_mm_uncorrected': float(ipd_mm_uncorrected),
        'depth_correction_mm': float(ipd_mm_corrected - ipd_mm_uncorrected),
        'pixels_per_mm': float(pixels_per_mm),
    }

# ---- Visualization ----

def draw_visualization(bgr_img, face_data, card_candidate, expected, ipd_result, quality):
    """Draw comprehensive visualization"""
    vis = bgr_img.copy()

    # Draw expected region (light blue rectangle)
    roi_x1, roi_y1, roi_x2, roi_y2 = expected['roi']
    cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 200, 100), 2)

    # Draw expected center (cyan cross)
    exp_cx, exp_cy = expected['expected_center']
    cv2.drawMarker(vis, (int(exp_cx), int(exp_cy)), (255, 255, 0),
                   cv2.MARKER_CROSS, 30, 2)

    # Draw detected card box
    box = card_candidate['box'].astype(int)
    color = (0, 255, 0) if quality['is_good'] else (0, 165, 255)
    cv2.drawContours(vis, [box], -1, color, 3)
    for corner in box:
        cv2.circle(vis, tuple(corner), 8, color, -1)

    # Draw iris centers and IPD line
    for iris in [face_data['left_iris'], face_data['right_iris']]:
        cv2.circle(vis, (int(iris[0]), int(iris[1])), 6, (0, 255, 0), -1)
        cv2.circle(vis, (int(iris[0]), int(iris[1])), 9, (0, 255, 0), 2)

    cv2.line(vis,
             (int(face_data['left_iris'][0]), int(face_data['left_iris'][1])),
             (int(face_data['right_iris'][0]), int(face_data['right_iris'][1])),
             (0, 255, 255), 3)

    # Text overlay
    def put_text_bg(img, text, pos, color, bg=(0, 0, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2
        size, _ = cv2.getTextSize(text, font, scale, thick)
        w, h = size
        x, y = pos
        cv2.rectangle(img, (x-5, y-h-5), (x+w+5, y+5), bg, -1)
        cv2.putText(img, text, pos, font, scale, color, thick)

    y_offset = 35
    put_text_bg(vis, f"IPD: {ipd_result['ipd_mm']:.2f} mm", (20, y_offset), (50, 255, 50))
    y_offset += 35
    put_text_bg(vis, f"Card: {card_candidate['width_px']:.0f}x{card_candidate['height_px']:.0f} px",
                (20, y_offset), (200, 200, 200))
    y_offset += 35
    quality_text = "Quality: " + ("GOOD" if quality['is_good'] else "FAIR")
    quality_color = (0, 255, 0) if quality['is_good'] else (0, 165, 255)
    put_text_bg(vis, quality_text, (20, y_offset), quality_color)

    return vis

# ---- Main Pipeline ----

def main(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"Could not read image from '{image_path}'.")

    print("\n" + "="*60)
    print("IPD MEASUREMENT - GEOMETRY-GUIDED VERSION v3.0")
    print("="*60)

    print("\n[STEP 1/4] Detecting face landmarks and iris positions...")
    face_data = detect_face_landmarks_and_iris(bgr)
    print(f"✓ Face detected successfully")
    print(f"  - IPD (pixels): {face_data['ipd_px']:.2f} px")
    print(f"  - Iris radii: L={face_data['left_iris_r']:.1f}, R={face_data['right_iris_r']:.1f} px")

    print("\n[STEP 2/4] Detecting card using face geometry guidance...")
    card_candidate, expected, quality, all_candidates = detect_card_with_face_guidance(bgr, face_data)
    print(f"✓ Card detected (quality score: {quality['score']:.2f}/10)")

    print("\n[STEP 3/4] Applying perspective correction...")
    warped, corrected_width, persp_quality = apply_perspective_correction(bgr, card_candidate['box'])
    print(f"✓ Perspective corrected")
    print(f"  - Corrected card width: {corrected_width:.1f} px")
    print(f"  - Edge ratios: long={persp_quality['long_edge_ratio']:.3f}, short={persp_quality['short_edge_ratio']:.3f}")

    print("\n[STEP 4/4] Computing IPD...")
    ipd_result = compute_ipd_mm(face_data['ipd_px'], corrected_width)
    print(f"✓ IPD calculated")
    print(f"  - Uncorrected: {ipd_result['ipd_mm_uncorrected']:.2f} mm")
    print(f"  - Depth correction: +{ipd_result['depth_correction_mm']:.2f} mm")
    print(f"  - FINAL IPD: {ipd_result['ipd_mm']:.2f} mm")

    # Validation
    print("\n[STEP 5/5] Quality validation...")
    warnings = []

    if not (IPD_MIN_MM <= ipd_result['ipd_mm'] <= IPD_MAX_MM):
        warnings.append(f"IPD outside typical range ({IPD_MIN_MM}-{IPD_MAX_MM}mm)")

    if not quality['is_good']:
        warnings.append(f"Card detection quality could be better (score: {quality['score']:.1f}/10)")

    if not persp_quality['is_good']:
        warnings.append("Perspective distortion detected - try holding card more perpendicular")

    if quality['aspect_match'] < 0.5:
        warnings.append("Card aspect ratio significantly different from expected")

    if warnings:
        print("⚠ WARNINGS:")
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
    else:
        print("✓ All quality checks passed!")

    confidence = 'HIGH' if len(warnings) == 0 else 'MEDIUM' if len(warnings) <= 2 else 'LOW'

    print("\n" + "="*60)
    print(f"ESTIMATED IPD: {ipd_result['ipd_mm']:.2f} mm")
    print(f"CONFIDENCE: {confidence}")
    print("="*60 + "\n")

    vis = draw_visualization(bgr, face_data, card_candidate, expected, ipd_result, quality)

    return ipd_result['ipd_mm'], vis, {
        'face_data': face_data,
        'card_candidate': card_candidate,
        'ipd_result': ipd_result,
        'quality': quality,
        'perspective_quality': persp_quality,
        'warnings': warnings,
        'confidence': confidence,
    }

# ---- Entry Point ----

print("="*60)
print("IPD MEASUREMENT - GEOMETRY-GUIDED VERSION v3.0")
print("="*60)
print("\nIMPROVEMENTS:")
print("✓ Uses face geometry to predict card location")
print("✓ Focused search in expected region")
print("✓ Smart scoring based on size, position, aspect ratio")
print("✓ More reliable card detection")
print("\nInstructions:")
print("1. Hold credit card horizontally on forehead")
print("2. Center card above eyes")
print("3. Hold by one corner to minimize occlusion")
print("4. Face camera directly with good lighting")
print("\nUploading...")
print("="*60 + "\n")

uploaded = files.upload()

if len(uploaded) == 0:
    raise RuntimeError("No file uploaded.")

img_path = list(uploaded.keys())[0]
print(f"\n[INFO] Processing: {img_path}")

try:
    ipd_mm, vis, debug_data = main(img_path)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 10))
    plt.imshow(vis_rgb)
    plt.axis("off")
    plt.title(f"IPD: {ipd_mm:.2f} mm | Confidence: {debug_data['confidence']}",
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n[SUCCESS] Measurement complete!")
    print("\nLEGEND:")
    print("  🟩 Green box = Detected card")
    print("  🟦 Blue rectangle = Expected search region")
    print("  ✖ Cyan cross = Expected card center")
    print("  🟢 Green circles = Iris centers")
    print("  🟡 Yellow line = IPD measurement")

except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()

@app.post("/measure-ipd")
async def measure_ipd_api(file: UploadFile = File(...)):
    # 1. Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format"}

    try:
        # 2. Run your pipeline (your 'main' function logic)
        face_data = detect_face_landmarks_and_iris(img)
        card_candidate, expected, quality, _ = detect_card_with_face_guidance(img, face_data)
        warped, corrected_width, persp_quality = apply_perspective_correction(img, card_candidate['box'])
        ipd_result = compute_ipd_mm(face_data['ipd_px'], corrected_width)

        # 3. Return only data (API shouldn't return plots)
        return {
            "ipd_mm": round(ipd_result['ipd_mm'], 2),
            "confidence": "HIGH" if quality['is_good'] else "LOW",
            "measurements": {
                "uncorrected_mm": ipd_result['ipd_mm_uncorrected'],
                "depth_correction": ipd_result['depth_correction_mm']
            }
        }
    except Exception as e:
        return {"error": str(e)}