import os
import sys
import json
from pathlib import Path

# Add src to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engines.content.researcher import gather_research
from src.engines.content.generator import generate_hooks, generate_script

def run_content_agent():
    print("=========================================")
    print("🤖 Welcome to the ContentOS Agent")
    print("=========================================")
    
    # 1. Get input from the user (Agent Interaction)
    topic = input("\n[Agent]: What topic would you like to create a 5-minute video about?\n> ")
    if not topic.strip():
        print("Topic cannot be empty. Exiting.")
        return

    # 2. Research
    print(f"\n[Agent]: Excellent choice. I'm now browsing the web to gather factual research on '{topic}'...")
    brief = gather_research(topic)
    print("\n[Agent]: Research complete! Here is a quick snippet of what I found:")
    print(f"         {brief[:150]}...")
    
    # 3. Hook Generation
    print("\n[Agent]: Now I'll draft 3 potential hooks to start the video. Give me a moment...")
    hooks = generate_hooks(topic, brief)
    
    print("\n[Agent]: Here are the 3 hooks I came up with:")
    for i, hook in enumerate(hooks):
        print(f"\n--- Option {i + 1} ---")
        print(hook)
        
    # 4. Get User Decision (Agent Interaction)
    while True:
        try:
            choice = int(input("\n[Agent]: Which hook do you prefer? (Enter 1, 2, or 3):\n> "))
            if choice in [1, 2, 3]:
                chosen_hook = hooks[choice - 1]
                break
            else:
                print("Please enter a valid number (1, 2, or 3).")
        except ValueError:
            print("Please enter a number.")
            
    print(f"\n[Agent]: Perfect. I will use Option {choice}.")
    
    # 5. Full Script Generation
    print("\n[Agent]: I am now writing the full 5-minute video script based on the research and your chosen hook. This might take a minute...")
    script = generate_script(topic, brief, chosen_hook)
    
    # 6. Save Artifacts for TTS/Video Engines
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    file_path = output_dir / "latest_script.json"
    with open(file_path, "w") as f:
        # Pydantic v2 serialization
        f.write(script.model_dump_json(indent=4))
        
    print(f"\n[Agent]: 🎉 Success! The 5-minute script '{script.title}' is ready.")
    print(f"[Agent]: It contains {len(script.scenes)} scenes.")
    print(f"[Agent]: The complete JSON data structure has been saved to: {file_path}")
    print("[Agent]: This JSON will be used by our future TTS and Video Generation engines!")

if __name__ == "__main__":
    run_content_agent()
