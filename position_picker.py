"""
Parking Space Position Picker
--------------------------------
A small utility to mark rectangular parking-space regions on a reference
image (e.g. the first frame of your parking lot video/CCTV feed).
Left-click to add a parking-space rectangle, right-click to remove one.
Positions are saved to 'CarParkPos.pkl' for use by main.py.

Usage:
    python position_picker.py --image samples/parking_reference.png

Controls:
    Left click  -> add a parking spot at that location
    Right click -> remove the parking spot under the cursor
    s           -> save positions to CarParkPos.pkl
    q           -> quit
"""

import cv2
import pickle
import os
import argparse

WIDTH, HEIGHT = 107, 48  # default size of a parking space rectangle (tweak to your footage)
POS_FILE = "CarParkPos.pkl"


def load_positions():
    if os.path.exists(POS_FILE):
        with open(POS_FILE, "rb") as f:
            return pickle.load(f)
    return []


def save_positions(positions):
    with open(POS_FILE, "wb") as f:
        pickle.dump(positions, f)
    print(f"[INFO] Saved {len(positions)} parking positions to {POS_FILE}")


def mouse_callback(event, x, y, flags, param):
    positions = param["positions"]

    if event == cv2.EVENT_LBUTTONDOWN:
        positions.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(positions):
            x1, y1 = pos
            if x1 <= x <= x1 + WIDTH and y1 <= y <= y1 + HEIGHT:
                positions.pop(i)
                break


def run(image_path):
    base_image = cv2.imread(image_path)
    if base_image is None:
        raise FileNotFoundError(f"Could not load reference image: {image_path}")

    positions = load_positions()
    param = {"positions": positions}

    cv2.namedWindow("Parking Space Picker")
    cv2.setMouseCallback("Parking Space Picker", mouse_callback, param)

    print("[INFO] Left-click to add a spot, right-click to remove, 's' to save, 'q' to quit.")

    while True:
        display = base_image.copy()
        for pos in positions:
            cv2.rectangle(display, pos, (pos[0] + WIDTH, pos[1] + HEIGHT), (255, 0, 255), 2)

        cv2.putText(display, f"Spots marked: {len(positions)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Parking Space Picker", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            save_positions(positions)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark parking space positions on a reference image")
    parser.add_argument("--image", type=str, required=True,
                         help="Path to a reference image (e.g. a single frame from your parking lot video)")
    parser.add_argument("--width", type=int, default=WIDTH, help="Width of each parking space box")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Height of each parking space box")
    args = parser.parse_args()

    WIDTH, HEIGHT = args.width, args.height
    run(args.image)
