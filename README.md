# 🛡️ CrowdShield AI: Edge-AI Predictive Crowd Risk Intelligence

CrowdShield AI is an enterprise-grade, real-time crowd monitoring and stampede prevention system. It combines Computer Vision (YOLOv8) for dynamic crowd density analysis with a **RAG-powered Edge AI Copilot (Llama-3)** to instantly generate standard operating procedures (SOPs) and evacuation protocols locally, ensuring zero-latency and maximum data privacy.

## 🚀 Key Features
- **Real-Time Crowd Vision & Heatmaps:** Utilizes YOLOv8n to accurately detect individuals frame-by-frame, overlaid with OpenCV Gaussian-blur heatmaps to visually isolate high-density danger zones.
- **RAG-Powered Security Copilot:** Features an integrated chatbot powered by a local Large Language Model (Llama-3). It uses FAISS Vector Database to retrieve official disaster protocols and injects live camera metrics (Context-Aware Prompting) to give precise, rule-based answers.
- **Interactive Command Center:** A full-stack Streamlit dashboard providing live video feeds, historical crowd density line charts, and dynamic visual/textual alerts when critical thresholds are breached.
- **100% Offline Edge Deployment:** Runs entirely on-premise without relying on external APIs (like OpenAI), ensuring continuous security monitoring even without internet access.

## 🛠️ Tech Stack
- **Computer Vision:** Ultralytics YOLOv8, OpenCV, NumPy
- **Generative AI & LLMs:** Llama-3 (8B) via Ollama
- **RAG Pipeline:** LangChain, FAISS (Vector DB), HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- **Frontend & Data Visualization:** Streamlit, Pandas

## 📂 Project Structure
- `app.py` / `dashboard.py`: The main Streamlit application featuring the Live Command Center and Sidebar Copilot.
- `camera_simulator.py` / `main.py`: Backend scripts for testing raw detection and simulating data.
- `knowledge/`: Contains official text rules and protocols used by the RAG Vector DB (`crowd_safety_guidelines.txt`, `evacuation_rules.txt`).
- `requirements.txt`: Python package dependencies.
- `crowd.mp4`: Sample CCTV footage for testing the vision pipeline.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dev200413y/crowdshield_ai.git](https://github.com/dev200413y/crowdshield_ai.git)
   cd CrowdShield_AI
