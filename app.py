import streamlit as st
import requests
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time
import warnings
warnings.filterwarnings('ignore')

# --- RAG LIBRARIES ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------
# 1. INITIALIZE RAG KNOWLEDGE BASE (Vector DB)
# ---------------------------------------------------------
@st.cache_resource
def setup_rag():
    print("⏳ Loading Knowledge Base & Embeddings...")
    # Ye teri Official PDF ya Rulebook ka text hai (Mock Data)
    sop_rules = [
        "Escalator Rules: If crowd exceeds 500 near escalators, immediately halt the escalators and redirect traffic to stairs.",
        "Exit Gate Protocol: If exit gates reach critical capacity, open emergency exits Alpha and Beta immediately.",
        "Stampede Prevention: Deploy Quick Response Team (QRT) in a human chain formation to divide the crowd.",
        "Evacuation Communication: Use public address systems to calmly guide people. Do not use loud panic alarms."
    ]
    
    # Text ko documents mein convert kiya
    docs = [Document(page_content=rule) for rule in sop_rules]
    
    # Wahi tera purana aur fast embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # FAISS Vector Database banaya
    vector_db = FAISS.from_documents(docs, embeddings)
    print("✅ RAG Vector DB Ready!")
    return vector_db

vector_db = setup_rag()

# ---------------------------------------------------------
# 2. INITIALIZE YOLO MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt") 

model = load_yolo()

# ---------------------------------------------------------
# 3. RAG + LLM CHAT ENGINE
# ---------------------------------------------------------
def chat_with_rag(prompt, current_crowd):
    """Ye function pehle Vector DB mein search karega, phir LLM ko dega (Asli RAG)"""
    
    # 1. RETRIEVE: User ke question ke hisaab se SOP Vector DB se rule dhoondo
    retrieved_docs = vector_db.similarity_search(prompt, k=2)
    rag_context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. AUGMENT: Live data aur RAG context ko ek sath prompt mein daalo
    system_prompt = f"""
    You are the 'CrowdShield AI' Security Copilot. 
    
    [LIVE CAMERA DATA]: {current_crowd} people currently detected.
    [OFFICIAL SOP RULES]: {rag_context}
    
    Answer the user's query professionally using ONLY the official SOP rules provided and the live data. Keep it short.
    """
    
    full_prompt = f"{system_prompt}\nUser: {prompt}"
    
    # 3. GENERATE: Llama-3 ko bhej do
    try:
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json={"model": "llama3", "prompt": full_prompt, "stream": False}
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"⚠️ LLM Connection Error: {e}"

# ---------------------------------------------------------
# 4. STREAMLIT UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="CrowdShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your RAG-Powered AI Copilot. Ask me about protocols or the live feed."}]
if "current_crowd" not in st.session_state:
    st.session_state.current_crowd = 0

# --- SIDEBAR RAG CHATBOT UI ---
with st.sidebar:
    st.title("💬 RAG Security Copilot")
    st.markdown("Equipped with Official SOPs & Live Vision")
    st.divider()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("E.g., What is the escalator protocol?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching SOPs & Analyzing Video..."):
                reply = chat_with_rag(prompt, st.session_state.current_crowd)
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------
# 5. MAIN DASHBOARD UI & VISION LOOP
# ---------------------------------------------------------
st.title("🛡️ CrowdShield AI - Advanced Command Center")
st.markdown("### Real-Time Vision, Heatmap & RAG-Powered Intelligence")
st.divider()

col_video, col_data = st.columns([1.5, 1])

with col_video:
    st.markdown("#### 🎥 Live Camera Feed & Heatmap")
    video_placeholder = st.empty()

with col_data:
    st.markdown("#### 📊 Real-Time Analytics")
    metric_placeholder = st.empty()
    graph_placeholder = st.empty()
    alert_placeholder = st.empty()

video_source = 'crowd.mp4'

if st.button("▶️ Start Live Analytics"):
    cap = cv2.VideoCapture(video_source)
    CRITICAL_THRESHOLD = 15 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video Feed Ended.")
            break
            
        frame = cv2.resize(frame, (640, 480))
        results = model.predict(frame, classes=[0], conf=0.3, verbose=False)
        
        person_count = 0
        centers = []
        
        for box in results[0].boxes:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        heatmap_layer = np.zeros((480, 640), dtype=np.uint8)
        for (x, y) in centers:
            cv2.circle(heatmap_layer, (x, y), 40, (255), -1)
            
        heatmap_layer = cv2.GaussianBlur(heatmap_layer, (51, 51), 0)
        heatmap_colored = cv2.applyColorMap(heatmap_layer, cv2.COLORMAP_JET)
        final_output = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        video_placeholder.image(final_output, channels="BGR", use_container_width=True)
        
        st.session_state.current_crowd = person_count
        
        st.session_state.history.append(person_count)
        if len(st.session_state.history) > 50:
            st.session_state.history.pop(0)
            
        df = pd.DataFrame(st.session_state.history, columns=["Crowd Density"])
        
        with col_data:
            metric_placeholder.metric("👥 Active Crowd Count", f"{person_count} People")
            graph_placeholder.line_chart(df, height=200, use_container_width=True)
            
            if person_count > CRITICAL_THRESHOLD:
                with alert_placeholder:
                    st.error(f"🚨 ALERT: Density Exceeded ({person_count} detected)!")
            else:
                alert_placeholder.success("✅ Safe Limits")
                
        time.sleep(0.1)

    cap.release()