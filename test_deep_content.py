import os
import sys

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.engines.content.researcher import gather_research
from src.engines.content.generator import generate_hooks, generate_script

def main():
    topic = "The Wow! Signal of 1977"
    platform = "TikTok"
    
    print("=== STEP 1: RESEARCH ===")
    brief = gather_research(topic)
    print(f"\nResearch Brief Preview (First 200 chars):\n{brief[:200]}...\n")
    
    print("=== STEP 2: HOOK GENERATION ===")
    hooks = generate_hooks(topic, brief, platform)
    print("\nGenerated Hooks:")
    for i, hook in enumerate(hooks):
        print(f"{i + 1}. {hook}")
        
    # Agent decides on Hook 1 (in a real system, the agent or user chooses)
    chosen_hook = hooks[0]
    print(f"\n[Agent Logic] Chose Hook #1: {chosen_hook}\n")
    
    print("=== STEP 3: SCRIPT GENERATION ===")
    script = generate_script(topic, brief, chosen_hook, platform)
    
    print(f"\nFinal Title: {script.title}")
    print(f"Total Scenes: {len(script.scenes)}\n")
    
    for scene in script.scenes:
        print(f"--- Scene {scene.scene_number} ---")
        print(f"Narration: {scene.narration}")
        print(f"Visual: {scene.visual_prompt}\n")

if __name__ == "__main__":
    main()
