import time
import random
# Humare engine se brain import kar rahe hain
from advisory_engine import get_mitigation_plan 

def generate_crowd_data():
    zones = ["Entry Gate-1", "Escalator Zone-2B", "Platform-3", "East Wing Exit"]
    
    while True:
        zone = random.choice(zones)
        people_count = random.randint(50, 1000)
        
        if people_count < 300:
            status = "🟢 NORMAL"
        elif people_count < 700:
            status = "🟠 WARNING"
        else:
            status = "🔴 CRITICAL - Action Needed!"
            
        print(f"📡 [LIVE FEED] Zone: {zone} | Crowd: {people_count} | Status: {status}")
        
        # AI TRIGGER: Jaise hi CRITICAL aaya, RAG ko hit karo!
        if "CRITICAL" in status:
            print(f"\n🚨 ALERT TRIGGERED for {zone}! Asking CrowdShield AI for a mitigation plan...")
            
            # Engine ko bulaya aur live problem batayi
            ai_solution = get_mitigation_plan(zone, people_count)
            
            print(f"🤖 CROWDSHIELD AI RESPONSE:\n{ai_solution}\n")
            print("-" * 65)
            print("Resuming Live Feed in 5 seconds...\n")
            time.sleep(5) # Action lene ka time de rahe hain
            
        else:
            time.sleep(2) # Normal/Warning status mein camera har 2 sec mein refresh hoga

if __name__ == "__main__":
    print("🚀 Starting CrowdShield AI Live Simulator...\n")
    print("-" * 65)
    generate_crowd_data()