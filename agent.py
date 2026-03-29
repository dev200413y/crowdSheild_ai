"""
CrowdShield AI - Autonomous Monitoring Agent
============================================
A ReAct-style (Reason + Act) agent that:
  1. Observes crowd data from crowd_log.csv (written by main.py)
  2. Reasons about the risk using a RAG knowledge base
  3. Acts by generating a mitigation plan, logging the incident to SQLite,
     and printing a structured alert

Run directly:
    python agent.py [--once]          # --once exits after one observation cycle

Or import and call run_agent_loop() from another module.
"""

import argparse
import csv
import json
import os
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ── Optional LangChain RAG (falls back to keyword search if unavailable) ──────
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

from database import log_incident, log_agent_event

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CSV_LOG_PATH = os.path.join(os.path.dirname(__file__), "crowd_log.csv")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

RISK_THRESHOLDS = {
    "CRITICAL": 7.0,
    "DANGER": 5.0,
    "SURGE": 3.0,
    "CROWDING": 2.0,
}

POLL_INTERVAL_SECONDS = 3   # how often the agent checks for new data
MAX_RAG_DOCS = 3            # number of knowledge chunks retrieved per query

# ─────────────────────────────────────────────────────────────────────────────
# RAG SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _load_knowledge_docs() -> list[str]:
    """Read all .txt files from the knowledge/ directory."""
    docs: list[str] = []
    knowledge_path = Path(KNOWLEDGE_DIR)
    for txt_file in sorted(knowledge_path.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()
        # Split on blank lines to get paragraph-level chunks
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        docs.extend(chunks)
    return docs


def _build_vector_db():
    """Return a FAISS vector DB over the knowledge files (or None if unavailable)."""
    if not _LANGCHAIN_AVAILABLE:
        return None
    try:
        raw_docs = _load_knowledge_docs()
        if not raw_docs:
            return None
        lc_docs = [Document(page_content=d) for d in raw_docs]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(lc_docs, embeddings)
    except Exception as exc:
        print(f"[Agent] ⚠️  Could not build vector DB: {exc}")
        return None


def _keyword_search(query: str, docs: list[str], top_k: int = MAX_RAG_DOCS) -> list[str]:
    """Simple keyword overlap fallback when LangChain is not available."""
    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        doc_words = set(doc.lower().split())
        score = len(query_words & doc_words)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def tool_read_crowd_data() -> Optional[dict]:
    """
    Tool: Read the latest row from crowd_log.csv.
    Returns a dict with keys: timestamp, gate_id, people_count,
    avg_velocity, pressure, predicted_pressure, risk_label.
    Returns None if the file does not exist or is empty.
    """
    if not os.path.exists(CSV_LOG_PATH):
        return None
    try:
        with open(CSV_LOG_PATH, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        latest = rows[-1]
        result = {
            "timestamp": latest.get("timestamp", ""),
            "gate_id": latest.get("gate_id", "Unknown"),
            "people_count": int(float(latest.get("people_count", 0))),
            "avg_velocity": float(latest.get("avg_velocity", 0)),
            "pressure": float(latest.get("pressure", 0)),
            "predicted_pressure": float(latest.get("predicted_pressure", 0)),
            "risk_label": latest.get("risk_label", "SAFE").strip().upper(),
            "_row_count": len(rows),  # used for change-detection
        }
        return result
    except Exception as exc:
        print(f"[Agent] ⚠️  Error reading crowd data: {exc}")
        return None


def tool_query_rag(query: str, vector_db=None, fallback_docs: Optional[list[str]] = None) -> str:
    """
    Tool: Retrieve the most relevant SOP/protocol chunks for a given query.
    Uses FAISS vector search if available, otherwise keyword search.
    """
    if vector_db is not None:
        try:
            results = vector_db.similarity_search(query, k=MAX_RAG_DOCS)
            return "\n---\n".join(doc.page_content for doc in results)
        except Exception:
            pass
    # Fallback
    docs = fallback_docs or _load_knowledge_docs()
    chunks = _keyword_search(query, docs)
    return "\n---\n".join(chunks) if chunks else "No relevant SOP found."


def tool_assess_risk(data: dict) -> str:
    """
    Tool: Determine the current risk level from crowd data.
    Returns one of: CRITICAL, DANGER, SURGE, CROWDING, SAFE.
    """
    pressure = data.get("pressure", 0)
    predicted = data.get("predicted_pressure", 0)
    label = data.get("risk_label", "SAFE")

    # Use the label from the vision engine if already classified
    if label in ("CRITICAL", "DANGER", "SURGE", "CROWDING"):
        return label

    # Fall back to pressure-based classification
    effective_pressure = max(pressure, predicted)
    if effective_pressure >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if effective_pressure >= RISK_THRESHOLDS["DANGER"]:
        return "DANGER"
    if effective_pressure >= RISK_THRESHOLDS["SURGE"]:
        return "SURGE"
    if effective_pressure >= RISK_THRESHOLDS["CROWDING"]:
        return "CROWDING"
    return "SAFE"


def tool_generate_action_plan(zone: str, data: dict, rag_context: str, risk_level: str) -> str:
    """
    Tool: Ask the local LLM (Ollama/Llama-3) for a concrete mitigation plan,
    grounded in the retrieved SOP context.  Falls back to a rule-based plan
    if the LLM is unreachable.
    """
    system_prompt = textwrap.dedent(f"""
        You are the 'CrowdShield AI' Emergency Response Agent.

        [LIVE DATA]
        Zone: {zone}
        People Detected: {data.get('people_count', 'N/A')}
        Crowd Pressure: {data.get('pressure', 'N/A')} (predicted: {data.get('predicted_pressure', 'N/A')})
        Average Velocity: {data.get('avg_velocity', 'N/A')} m/s
        Risk Level: {risk_level}

        [OFFICIAL SOP CONTEXT]
        {rag_context}

        Based ONLY on the SOP context and live data above, provide a concise, numbered
        3-step mitigation plan for the ground security team. No intro or outro.
    """).strip()

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": system_prompt, "stream": False},
            timeout=8,
        )
        resp.raise_for_status()
        plan = resp.json().get("response", "").strip()
        if plan:
            return plan
    except Exception:
        pass

    # ── Rule-based fallback ───────────────────────────────────────────────────
    fallback_plans = {
        "CRITICAL": (
            "1. Activate full evacuation — open all emergency exits Alpha, Beta and Gamma.\n"
            "2. QRT deploys human-chain formation to divide and redirect crowd flow.\n"
            "3. Call emergency services and notify Incident Commander immediately."
        ),
        "DANGER": (
            "1. Open emergency exits Alpha and Beta; redirect crowd to East Wing.\n"
            "2. Halt all escalators and travelators to prevent pile-ups.\n"
            "3. Play calm PA announcement: 'Please move slowly to the nearest exit.'"
        ),
        "SURGE": (
            "1. Close secondary entry gates; enforce single-file entry at main gate.\n"
            "2. Deploy marshals to affected zone and request QRT standby.\n"
            "3. Activate LED signboards directing crowd to least-congested exits."
        ),
        "CROWDING": (
            "1. Play automated audio asking people to keep moving; no floor sitting.\n"
            "2. Reduce incoming footfall by 50% at nearby entrances.\n"
            "3. Monitor velocity; escalate to SURGE protocol if speed drops below 0.2 m/s."
        ),
    }
    return fallback_plans.get(risk_level, "Situation is under control. Continue monitoring.")


def tool_trigger_alert(risk_level: str, zone: str, plan: str, data: dict) -> None:
    """
    Tool: Print a formatted alert to the console and log it to the database.
    In a production system this would also send SMS/email/push notifications.
    """
    separator = "=" * 70
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    level_icons = {
        "CRITICAL": "🔴",
        "DANGER": "🟠",
        "SURGE": "🟡",
        "CROWDING": "🔵",
        "SAFE": "🟢",
    }
    icon = level_icons.get(risk_level, "⚪")

    print(separator)
    print(f"{icon} [{ts}] CROWDSHIELD AI AGENT ALERT")
    print(f"   Risk Level  : {risk_level}")
    print(f"   Zone        : {zone}")
    print(f"   People Count: {data.get('people_count', 'N/A')}")
    print(f"   Pressure    : {data.get('pressure', 'N/A'):.2f} "
          f"(predicted: {data.get('predicted_pressure', 'N/A'):.2f})")
    print(f"   Velocity    : {data.get('avg_velocity', 'N/A'):.3f} m/s")
    print()
    print("📋 MITIGATION PLAN:")
    for line in plan.splitlines():
        print(f"   {line}")
    print(separator)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

class CrowdShieldAgent:
    """
    Autonomous ReAct agent for CrowdShield AI.

    Cycle:
        Observe → Reason → Act → Repeat
    """

    def __init__(self) -> None:
        print("[Agent] 🔄 Initialising knowledge base…")
        self._vector_db = _build_vector_db()
        self._fallback_docs = _load_knowledge_docs()
        self._last_seen_row_count: int = 0   # track by row count for reliable deduplication
        self._last_alerted_risk: str = "SAFE"
        if self._vector_db:
            print("[Agent] ✅ FAISS vector DB ready.")
        else:
            print("[Agent] ⚠️  Using keyword-search fallback (LangChain unavailable).")

    # ── OBSERVE ───────────────────────────────────────────────────────────────
    def observe(self) -> Optional[dict]:
        data = tool_read_crowd_data()
        return data

    # ── REASON ────────────────────────────────────────────────────────────────
    def reason(self, data: dict) -> tuple[str, str]:
        """Returns (risk_level, rag_context)."""
        risk_level = tool_assess_risk(data)
        query = (
            f"Risk level {risk_level} detected in zone {data.get('gate_id', 'unknown')} "
            f"with crowd pressure {data.get('pressure', 0):.1f} and "
            f"{data.get('people_count', 0)} people. "
            f"What are the recommended protocols and actions?"
        )
        rag_context = tool_query_rag(query, self._vector_db, self._fallback_docs)
        return risk_level, rag_context

    # ── ACT ───────────────────────────────────────────────────────────────────
    def act(self, data: dict, risk_level: str, rag_context: str) -> None:
        zone = data.get("gate_id", "Unknown")
        plan = tool_generate_action_plan(zone, data, rag_context, risk_level)

        # Always log to DB
        incident_id = log_incident(
            zone=zone,
            people_count=data.get("people_count", 0),
            pressure=data.get("pressure", 0.0),
            risk_label=risk_level,
            action_taken=f"Agent triggered at risk level {risk_level}",
            ai_plan=plan,
        )
        log_agent_event(
            event_type=f"ALERT_{risk_level}",
            zone=zone,
            details=json.dumps(
                {"incident_id": incident_id, "pressure": data.get("pressure")}
            ),
        )

        # Only print alert for non-SAFE events or when risk has changed
        if risk_level != "SAFE":
            tool_trigger_alert(risk_level, zone, plan, data)
        elif self._last_alerted_risk != "SAFE":
            print(f"[Agent] 🟢 [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                  f"Zone {zone} returned to SAFE. Monitoring…")

        self._last_alerted_risk = risk_level

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    def run_once(self) -> None:
        """Execute a single observe-reason-act cycle."""
        data = self.observe()
        if data is None:
            print("[Agent] ⏳ No crowd data yet. Is main.py running?")
            return

        # Skip if no new rows have been appended since last cycle
        row_count = data.get("_row_count", 0)
        if row_count <= self._last_seen_row_count:
            return
        self._last_seen_row_count = row_count

        risk_level, rag_context = self.reason(data)
        self.act(data, risk_level, rag_context)

    def run_loop(self, poll_interval: float = POLL_INTERVAL_SECONDS) -> None:
        """Run the agent indefinitely, polling for new data every `poll_interval` seconds."""
        print(f"[Agent] 🚀 CrowdShield Autonomous Agent started (poll every {poll_interval}s). Ctrl+C to stop.")
        try:
            while True:
                self.run_once()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[Agent] 🛑 Agent stopped by user.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_loop(poll_interval: float = POLL_INTERVAL_SECONDS) -> None:
    """Public helper to start the agent from another module."""
    agent = CrowdShieldAgent()
    agent.run_loop(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrowdShield AI Autonomous Agent")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single observe-reason-act cycle and exit (useful for testing).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL_SECONDS,
        help=f"Polling interval in seconds (default: {POLL_INTERVAL_SECONDS}).",
    )
    args = parser.parse_args()

    agent = CrowdShieldAgent()
    if args.once:
        agent.run_once()
    else:
        agent.run_loop(args.interval)
