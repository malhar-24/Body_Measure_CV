# Body Measure CV

Body Measure CV is a computer vision project for estimating human body measurements from a single RGB image using pose estimation, body segmentation, and reference object calibration.

## Features

- 📏 Height estimation
- 💪 Shoulder width measurement
- 👕 Waist width measurement
- 🪪 Reference object calibration (e.g., PAN card or ID card)
- 🧍 Pose estimation
- 🎯 Body segmentation
- 🌐 Web interface for easy use
- 🐍 Modular Python implementation

---

## Demo

### Video

```
[▶ Watch Demo]([demo.mp4](https://github.com/malhar-24/Body_Measure_CV/blob/8c01b672b5375042d6ef0cabc599f6022b129eef/demo.mp4))
```

---

## Project Structure

```
Body_Measure_CV/
│
├── body_measurement/
│   ├── measure/
│   ├── models/
│   ├── utils/
│   ├── ...
│
├── website/
│
├── demo/
│   └── demo.mp4
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/malhar-24/Body_Measure_CV.git
```

Move into the project

```bash
cd Body_Measure_CV
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Flask
- HTML
- CSS
- JavaScript

---

## Future Improvements

- 3D body measurement
- Multi-view reconstruction
- Higher measurement accuracy
- Mobile application
- Additional anthropometric measurements

---
