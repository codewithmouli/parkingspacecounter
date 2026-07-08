# 🅿️ Parking Space Occupancy Counter (OpenCV)

Detects which spaces in a parking lot are free or occupied from a static camera feed —
no machine learning model required, just classical image processing. A well-known
practical computer-vision project that's easy to demo visually.

## How it works
1. **Mark parking spots once** using `position_picker.py` by clicking on a reference frame from your footage.
2. For each video frame:
   - Convert to grayscale, blur, and apply **adaptive thresholding** to highlight edges/texture (cars have far more edges/texture than empty asphalt).
   - Crop out each marked parking-space rectangle.
   - Count non-zero pixels in that crop — a high count means a car is likely parked there (lots of edges), a low count means the space is empty.
3. Draw **green** boxes for free spots and **red** boxes for occupied ones, plus a running "Free: X/Y" counter.

## Installation
```bash
git clone https://github.com/<your-username>/parking-space-counter-opencv.git
cd parking-space-counter-opencv
pip install -r requirements.txt
```

## Usage

### Step 1 — Mark parking spaces (one-time setup per camera angle)
Grab a single reference frame from your parking-lot video/image and mark each spot:
```bash
python position_picker.py --image samples/parking_reference.png
```
- Left-click to add a spot
- Right-click to remove a spot
- Press `s` to save to `CarParkPos.pkl`
- Press `q` to quit

### Step 2 — Run the counter
On a video:
```bash
python main.py --video samples/parking_lot.mp4
```
On a single image:
```bash
python main.py --image samples/parking_reference.png
```

Tune sensitivity if spots are misclassified:
```bash
python main.py --video samples/parking_lot.mp4 --threshold 1100
```

## Project Structure
```
parking-space-counter-opencv/
├── position_picker.py     # one-time tool to mark parking spot rectangles
├── main.py                # runs detection on video/image and shows free/occupied counts
├── requirements.txt
├── README.md
├── samples/                # add your own parking lot video/images here
└── CarParkPos.pkl           # auto-generated after running position_picker.py
```

> **Note:** You'll need your own parking-lot footage/image (a short static-camera clip works
> great — even a phone video of a parking lot from a window will do). Free sample parking-lot
> datasets/videos are also available online (search "PKLot dataset").

## Key Concepts Demonstrated
- Adaptive thresholding for texture-based classification
- ROI (Region of Interest) extraction
- Mouse-event callbacks for interactive annotation tools
- Pickle-based configuration persistence
- Real-time video analytics loop

## Possible Extensions
- Auto-detect parking space boundaries instead of manual marking (e.g. using Hough lines).
- Log occupancy history over time to a CSV and plot utilization trends.
- Add a simple web dashboard (Flask) to show live free-space count remotely.

## License
MIT
