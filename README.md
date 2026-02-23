# CrowdShield AI 🛡️

CrowdShield AI is an intelligent crowd monitoring and safety management system. It leverages computer vision (YOLOv8) to detect and analyze crowd density in real-time, providing actionable safety guidelines and evacuation protocols based on a custom advisory engine.

## Features
- **Real-Time Crowd Detection:** Utilizes YOLOv8 for accurate crowd counting and density estimation.
- **Interactive Dashboard:** Built with Streamlit for live monitoring and data visualization.
- **Automated Advisory Engine:** Generates real-time alerts and guidelines based on predefined disaster protocols and evacuation rules.
- **Data Logging:** Tracks crowd metrics over time in `crowd_log.csv`.

## Project Structure
- `main.py`: Core script for running the crowd detection model.
- `dashboard.py`: Streamlit application for the interactive user interface.
- `advisory_engine.py`: Logic for generating safety advisories using AI.
- `database.py`: Database operations and interaction handling.
- `knowledge/`: Contains text rules and protocols (`crowd_safety_guidelines.txt`, `disaster_protocols.txt`, `evacuation_rules.txt`).
- `requirements.txt`: Python package dependencies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dev200413y/crowdSheild_ai.git
   cd CrowdShield_AI
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**1. To run the main backend/detection script:**
```bash
python main.py
```

**2. To launch the interactive dashboard:**
```bash
streamlit run dashboard.py
```

## Setup Notes
- Make sure you have a `crowd.mp4` video file in the root directory if testing with recorded video.
- The `yolov8n.pt` model weights will be used automatically by the Ultralytics library.
