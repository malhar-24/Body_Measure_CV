from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = BASE_DIR / "weights"

CARD_MODEL = None
BODY_POSE_MODEL = None
BODY_SEG_MODEL = None

def load_models():
    """
    Load all required models.

    Parameters
    ----------
    card_model_path : str
        Path to trained card pose model.

    Returns
    -------
    card_model : YOLO
    body_pose_model : YOLO
    body_seg_model : YOLO
    """

    print("Loading models...")

    card_model = YOLO(str(WEIGHTS_DIR / "card_pose.pt"))

    body_pose_model = YOLO(str(WEIGHTS_DIR / "yolo26n-pose.pt"))

    body_seg_model = YOLO(str(WEIGHTS_DIR / "yolo11l-seg.pt"))

    print("✓ Card Pose Model Loaded")
    print("✓ Body Pose Model Loaded")
    print("✓ Body Segmentation Model Loaded")

    return card_model, body_pose_model, body_seg_model

import cv2
import numpy as np

CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98


def order_points(pts):
    """
    Order points as:
    Top Left, Top Right, Bottom Right, Bottom Left
    """

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top Left
    rect[2] = pts[np.argmax(s)]   # Bottom Right

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # Top Right
    rect[3] = pts[np.argmax(d)]   # Bottom Left

    return rect


def calculate_scale(card_model, image_path, conf=0.25, pixel_bias=0):
    """
    Detect the reference card and calculate pixels/mm.

    pixel_bias : float
        Positive value added to both card width and height (in pixels).
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    results = card_model.predict(
        image_path,
        conf=conf,
        verbose=False
    )

    if len(results) == 0 or results[0].keypoints is None:
        raise Exception("Card not detected.")

    keypoints = results[0].keypoints.xy.cpu().numpy()

    if len(keypoints) == 0:
        raise Exception("Card keypoints not found.")

    pts = order_points(keypoints[0])

    TL, TR, BR, BL = pts

    # ----------------------------
    # Card dimensions in pixels
    # ----------------------------
    widthA = np.linalg.norm(BR - BL)
    widthB = np.linalg.norm(TR - TL)
    card_width_px = max(widthA, widthB)

    heightA = np.linalg.norm(TR - BR)
    heightB = np.linalg.norm(TL - BL)
    card_height_px = max(heightA, heightB)

    # Add positive bias
    card_width_px += pixel_bias
    card_height_px += pixel_bias

    # ----------------------------
    # Perspective transform
    # ----------------------------
    dst = np.array([
        [0, 0],
        [card_width_px - 1, 0],
        [card_width_px - 1, card_height_px - 1],
        [0, card_height_px - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(
        pts.astype(np.float32),
        dst.astype(np.float32)
    )

    warped = cv2.warpPerspective(
        image,
        M,
        (int(card_width_px), int(card_height_px))
    )

    # ----------------------------
    # Scale
    # ----------------------------
    pixels_per_mm_width = card_width_px / CARD_WIDTH_MM
    pixels_per_mm_height = card_height_px / CARD_HEIGHT_MM

    pixels_per_mm = (
        pixels_per_mm_width +
        pixels_per_mm_height
    ) / 2

    return {
        "pixels_per_mm": pixels_per_mm,
        "card_width_px": card_width_px,
        "card_height_px": card_height_px,
        "ordered_points": pts,
        "perspective_matrix": M,
        "warped_card": warped
    }

import cv2
import numpy as np

# COCO Keypoint Indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def get_person_mask(seg_model, image_path, conf=0.5):
    """
    Detect the person and return the largest segmentation mask.

    Parameters
    ----------
    seg_model : YOLO
    image_path : str
    conf : float

    Returns
    -------
    mask : np.ndarray
        Binary mask (0/1) with original image size.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]

    results = seg_model.predict(
        image_path,
        conf=conf,
        verbose=False
    )

    result = results[0]

    if result.masks is None:
        raise Exception("No person detected.")

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()

    # Select the largest detected person
    areas = []

    for box in boxes:
        x1, y1, x2, y2 = box
        areas.append((x2 - x1) * (y2 - y1))

    idx = np.argmax(areas)

    mask = masks[idx]

    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    mask = (mask > 0.5).astype(np.uint8)

    return mask

import cv2
import matplotlib.pyplot as plt

# COCO Keypoint Indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def get_body_keypoints(body_pose_model, image_path, conf=0.5, visualize=True):
    """
    Detect body pose and return key body landmarks.

    Parameters
    ----------
    body_pose_model : YOLO
    image_path : str
    conf : float
    visualize : bool

    Returns
    -------
    dict
    """

    results = body_pose_model.predict(
        image_path,
        conf=conf,
        verbose=False
    )

    result = results[0]

    if result.keypoints is None:
        raise Exception("No person pose detected.")

    keypoints = result.keypoints.xy.cpu().numpy()

    if len(keypoints) == 0:
        raise Exception("No keypoints found.")

    kp = keypoints[0]

    # -----------------------------
    # Visualization
    # -----------------------------
    if visualize:

        # Built-in pose visualization
        pred = result.plot()

        # Draw index number for every keypoint
        for i, (x, y) in enumerate(kp):

            x = int(x)
            y = int(y)

            cv2.circle(pred, (x, y), 5, (0, 0, 255), -1)

            cv2.putText(
                pred,
                str(i),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(pred)
        plt.axis("off")
        plt.title("Body Pose with Keypoint Indices")
        plt.show()

    return {

        "left_shoulder": tuple(kp[LEFT_SHOULDER]),
        "right_shoulder": tuple(kp[RIGHT_SHOULDER]),

        "left_hip": tuple(kp[LEFT_HIP]),
        "right_hip": tuple(kp[RIGHT_HIP]),

        "left_ankle": tuple(kp[LEFT_ANKLE]),
        "right_ankle": tuple(kp[RIGHT_ANKLE]),

        "all_keypoints": kp

    }

def get_body_keypoints(body_pose_model, image_path, conf=0.5):
    """
    Detect body pose and return key body landmarks.

    Returns
    -------
    dict
    """

    results = body_pose_model.predict(
        image_path,
        conf=conf,
        verbose=False
    )

    result = results[0]

    if result.keypoints is None:
        raise Exception("No person pose detected.")

    keypoints = result.keypoints.xy.cpu().numpy()

    if len(keypoints) == 0:
        raise Exception("No keypoints found.")

    kp = keypoints[0]

    return {

        "left_shoulder": tuple(kp[LEFT_SHOULDER]),
        "right_shoulder": tuple(kp[RIGHT_SHOULDER]),

        "left_hip": tuple(kp[LEFT_HIP]),
        "right_hip": tuple(kp[RIGHT_HIP]),

        "left_ankle": tuple(kp[LEFT_ANKLE]),
        "right_ankle": tuple(kp[RIGHT_ANKLE]),

    }

import numpy as np

def calculate_height(mask, pose, pixels_per_mm):
    """
    Calculate body height using a vertical line through the body center.

    Returns
    -------
    dict
    """

    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        raise Exception("No body segmentation found.")

    # Highest and lowest segmentation rows
    top_y = int(ys.min())
    bottom_y = int(ys.max())

    # Center X from ankle midpoint
    left_ankle = pose["left_ankle"]
    right_ankle = pose["right_ankle"]

    center_x = int((left_ankle[0] + right_ankle[0]) / 2)

    # Top point on vertical measurement line
    top = (center_x, top_y)

    # Bottom point on vertical measurement line
    bottom = (center_x, bottom_y)

    # Vertical height
    height_px = bottom_y - top_y

    height_cm = height_px / pixels_per_mm / 10

    return {
        "top": top,
        "bottom": bottom,
        "height_px": int(height_px),
        "height_cm": float(height_cm)
    }
import numpy as np

def calculate_waist(mask, pose, pixels_per_mm, band=5):
    """
    Calculate waist width from segmentation.

    Uses:
    - Hip keypoints (11,12) to locate the waist level.
    - Only rows ABOVE the hip line.
    - Widest continuous mask segment on each row.

    Returns
    -------
    dict
    """

    # ----------------------------
    # Waist height from pose
    # ----------------------------
    lh = pose["left_hip"]
    rh = pose["right_hip"]

    waist_y = int((lh[1] + rh[1]) / 2)

    H, W = mask.shape

    left_edges = []
    right_edges = []

    # ----------------------------
    # Scan rows above waist
    # ----------------------------
    for y in range(max(0, waist_y-band), waist_y+1):

        xs = np.where(mask[y] > 0)[0]

        if len(xs) == 0:
            continue

        # Split into continuous segments
        split_idx = np.where(np.diff(xs) > 1)[0]
        segments = np.split(xs, split_idx + 1)

        # Choose widest segment
        widest = max(segments, key=len)

        left_edges.append(widest[0])
        right_edges.append(widest[-1])

    if len(left_edges) == 0:
        raise Exception("No body detected around waist.")

    # Average edges
    left_x = int(np.mean(left_edges))
    right_x = int(np.mean(right_edges))

    width_px = right_x - left_x

    front_mm = width_px / pixels_per_mm
    circumference_mm = front_mm * 2

    return {
        "left": (left_x, waist_y),
        "right": (right_x, waist_y),
        "front_width_px": width_px,
        "front_width_mm": front_mm,
        "front_width_cm": front_mm / 10,
        "circumference_mm": circumference_mm,
        "circumference_cm": circumference_mm / 10
    }
import numpy as np

def calculate_shoulder(mask, pose, pixels_per_mm, band=5):
    """
    Calculate shoulder width from segmentation.

    Uses:
    - Shoulder keypoints (5,6) to determine measurement height.
    - Averages over a vertical band.
    - Uses the widest continuous body segment on each row.

    Returns
    -------
    dict
    """

    # ------------------------------------
    # Shoulder height
    # ------------------------------------
    ls = pose["left_shoulder"]
    rs = pose["right_shoulder"]

    shoulder_y = int((ls[1] + rs[1]) / 2)

    left_edges = []
    right_edges = []

    # ------------------------------------
    # Scan around shoulder
    # ------------------------------------
    for y in range(max(0, shoulder_y - band), shoulder_y + band + 1):

        xs = np.where(mask[y] > 0)[0]

        if len(xs) == 0:
            continue

        # Split into continuous segments
        split_idx = np.where(np.diff(xs) > 1)[0]
        segments = np.split(xs, split_idx + 1)

        # Select widest connected segment
        widest = max(segments, key=len)

        left_edges.append(widest[0])
        right_edges.append(widest[-1])

    if len(left_edges) == 0:
        raise Exception("No body detected around shoulder.")

    left_x = int(np.mean(left_edges))
    right_x = int(np.mean(right_edges))

    width_px = right_x - left_x

    width_mm = width_px / pixels_per_mm
    width_cm = width_mm / 10

    return {
        "left": (left_x, shoulder_y),
        "right": (right_x, shoulder_y),
        "width_px": width_px,
        "width_mm": width_mm,
        "width_cm": width_cm
    }

import cv2
import os

def plot_measurements(
        image_path,
        mask,
        height,
        shoulder,
        waist,
        output_path
):
    """
    Draw measurements on the image and save it.

    Parameters
    ----------
    image_path : str
    mask : ndarray
    height : dict
    shoulder : dict
    waist : dict
    output_path : str

    Returns
    -------
    str
        Saved image path.
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Overlay segmentation
    # -----------------------------
    overlay = image.copy()
    overlay[mask == 1] = (0, 255, 0)

    image = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    # =============================
    # HEIGHT
    # =============================
    cv2.circle(image, height["top"], 8, (255, 0, 0), -1)
    cv2.circle(image, height["bottom"], 8, (255, 0, 0), -1)

    cv2.line(
        image,
        height["top"],
        height["bottom"],
        (255, 0, 0),
        3
    )

    cv2.putText(
        image,
        f'{height["height_cm"]:.1f} cm',
        (height["top"][0] + 10, height["top"][1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =============================
    # SHOULDER
    # =============================
    cv2.circle(image, shoulder["left"], 6, (0, 0, 255), -1)
    cv2.circle(image, shoulder["right"], 6, (0, 0, 255), -1)

    cv2.line(
        image,
        shoulder["left"],
        shoulder["right"],
        (0, 0, 255),
        3
    )

    sx = (shoulder["left"][0] + shoulder["right"][0]) // 2
    sy = shoulder["left"][1] - 10

    cv2.putText(
        image,
        f'{shoulder["width_cm"]:.1f} cm',
        (sx, sy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =============================
    # WAIST
    # =============================
    cv2.circle(image, waist["left"], 6, (255, 255, 0), -1)
    cv2.circle(image, waist["right"], 6, (255, 255, 0), -1)

    cv2.line(
        image,
        waist["left"],
        waist["right"],
        (255, 255, 0),
        3
    )

    wx = (waist["left"][0] + waist["right"][0]) // 2
    wy = waist["left"][1] - 10

    cv2.putText(
        image,
        f'{waist["circumference_cm"]:.1f} cm',
        (wx, wy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save image
    cv2.imwrite(output_path, image)

    return output_path


def initialize_models():
    """
    Load all models once when Django starts.
    """

    global CARD_MODEL
    global BODY_POSE_MODEL
    global BODY_SEG_MODEL

    if CARD_MODEL is None:

        CARD_MODEL, BODY_POSE_MODEL, BODY_SEG_MODEL = load_models()

        print("\n===================================")
        print("      AI Models Ready")
        print("===================================\n")


def measure_body(image_path):

    initialize_models()

    scale = calculate_scale(
        CARD_MODEL,
        image_path,
        conf=0.25,
        pixel_bias=-0.2
    )
    print(scale["pixels_per_mm"])
    pixels_per_mm = scale["pixels_per_mm"]

    mask = get_person_mask(
        BODY_SEG_MODEL,
        image_path
    )

    pose = get_body_keypoints(
        BODY_POSE_MODEL,
        image_path
    )

    height = calculate_height(
        mask,
        pose,
        pixels_per_mm
    )

    shoulder = calculate_shoulder(
        mask,
        pose,
        pixels_per_mm
    )

    waist = calculate_waist(
        mask,
        pose,
        pixels_per_mm
    )

    result_path = BASE_DIR / "media" / "results" / "result.jpg"

    plot_measurements(
        image_path,
        mask,
        height,
        shoulder,
        waist,
        str(result_path)
    )

    return {
    "height": round(float(height["height_cm"]), 1),
    "shoulder": round(float(shoulder["width_cm"]), 1),
    "waist": round(float(waist["circumference_cm"]), 1),
    "image": "/media/results/result.jpg",
    }



