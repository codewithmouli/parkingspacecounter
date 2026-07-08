"""
Parking Space Occupancy Counter using OpenCV
------------------------------------------------
Uses parking-space rectangles marked with position_picker.py, then analyzes
each region in a video (or image) feed to classify it as FREE or OCCUPIED
based on how many non-zero (edge/text-like) pixels appear after thresholding.

Usage:
    python main.py --video samples/parking_lot.mp4
    python main.py --image samples/parking_reference.png

Controls:
    q - quit
"""

import cv2
import pickle
import numpy as np
import argparse
import os

POS_FILE = "CarParkPos.pkl"
WIDTH, HEIGHT = 107, 48
OCCUPIED_PIXEL_THRESHOLD = 900  # tweak based on your footage/lighting


def load_positions():
    if not os.path.exists(POS_FILE):
        raise FileNotFoundError(
            f"'{POS_FILE}' not found. Run position_picker.py first to mark parking spots."
        )
    with open(POS_FILE, "rb") as f:
        return pickle.load(f)


def preprocess_frame(frame):
    """Convert frame into a binary image that highlights car edges/texture."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16
    )
    median = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)
    return dilated


def check_parking_spaces(frame, processed, positions):
    free_count = 0

    for pos in positions:
        x, y = pos
        crop = processed[y:y + HEIGHT, x:x + WIDTH]
        non_zero_count = cv2.countNonZero(crop)

        if non_zero_count < OCCUPIED_PIXEL_THRESHOLD:
            color = (0, 255, 0)  # green = free
            thickness = 3
            free_count += 1
        else:
            color = (0, 0, 255)  # red = occupied
            thickness = 2

        cv2.rectangle(frame, pos, (x + WIDTH, y + HEIGHT), color, thickness)
        cv2.putText(frame, str(non_zero_count), (x, y + HEIGHT - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    total = len(positions)
    cv2.rectangle(frame, (10, 10), (280, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"Free: {free_count}/{total}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame, free_count


def run_on_video(video_path, positions):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video

        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_frame(frame)
        output, free_count = check_parking_spaces(frame, processed, positions)

        cv2.imshow("Parking Space Counter", output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_image(image_path, positions):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    processed = preprocess_frame(frame)
    output, free_count = check_parking_spaces(frame, processed, positions)

    print(f"[INFO] Free spaces: {free_count}/{len(positions)}")
    cv2.imshow("Parking Space Counter", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Space Occupancy Counter")
    parser.add_argument("--video", type=str, help="Path to a parking lot video")
    parser.add_argument("--image", type=str, help="Path to a single parking lot image")
    parser.add_argument("--threshold", type=int, default=OCCUPIED_PIXEL_THRESHOLD,
                         help="Pixel-count threshold to classify a spot as occupied")
    args = parser.parse_args()

    OCCUPIED_PIXEL_THRESHOLD = args.threshold
    positions = load_positions()

    if args.video:
        run_on_video(args.video, positions)
    elif args.image:
        run_on_image(args.image, positions)
    else:
        print("Please provide --video <path> or --image <path>. Use -h for help.")
