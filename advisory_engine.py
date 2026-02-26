# advisory_engine.py
import requests
import json
import time

# ---------------------------------------------------------
# MAIN FUNCTION (Jo Simulator se call hoga)
# ---------------------------------------------------------
def get_mitigation_plan(zone, crowd_count):
    """
    Ye function background mein chal rahe Ollama (Llama 3) se direct baat karega.
    """
    
    # Llama 3 ke liye ekdum sharp prompt
    prompt = f"""
    You are the emergency response AI for 'CrowdShield AI'.
    URGENT ALERT: The crowd at {zone} has reached a CRITICAL danger level of {crowd_count} people.
    Provide a direct, strictly 3-step actionable mitigation plan to prevent a stampede. 
    Keep it professional, extremely short, and practical. No intro or outro text.
    """
    
    # Ollama ka local URL (Ye tere PC pe hi run ho raha hai)
    url = "http://localhost:11434/api/generate"
    
    # Data jo hum Ollama ko bhej rahe hain
    payload = {
        "model": "llama3", # Agar tune 'llama3' naam se download kiya hai
        "prompt": prompt,
        "stream": False # Taki pura answer ek saath aaye
    }
    
    try:
        # Llama 3 ko request bhej rahe hain
        response = requests.post(url, json=payload)
        response.raise_for_status() # Agar koi error aaya toh catch karega
        
        # Answer ko nikalna
        data = response.json()
        ai_response = data.get("response", "").strip()
        
        return ai_response
        
    except requests.exceptions.ConnectionError:
        return "⚠️ ERROR: Bhai, tera Ollama background mein on nahi hai! Pehle Ollama app start kar le."
    except Exception as e:
        return f"⚠️ Llama 3 Error: {e}"

# ---------------------------------------------------------
# DIRECT TESTING
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting CrowdShield Local Llama 3 Engine...\n")
    print("🤖 Checking connection with Ollama for Zone-2B (850 people)...\n")
    
    # Test run
    result = get_mitigation_plan("Zone-2B", 850)
    print(result)
    print("\n" + "-" * 50)