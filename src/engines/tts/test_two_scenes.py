import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.content import VideoScript

load_dotenv()

client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("LLM_API_KEY")
)

def test_first_two_scenes():
    print("=========================================")
    print("🎙️ Testing TTS with Expressive Punctuation")
    print("=========================================")
    
    narrator_id = os.getenv("NARRATOR_VOICE_ID")
    if not narrator_id:
        print("❌ Error: You need to run 'python src/engines/tts/voice_cloner.py' first to clone the voice and save NARRATOR_VOICE_ID to .env!")
        return

    script_path = Path("data/expressive_script.json")
    with open(script_path, "r") as f:
        script_data = json.load(f)
    
    script = VideoScript(**script_data)
    
    audio_dir = Path("data/audio_test")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using cloned Voice ID: {narrator_id}\n")
    
    for scene in script.scenes[:2]: # ONLY THE FIRST 2 SCENES
        print(f"--- Generating Scene {scene.scene_number} ---")
        print(f"Narration Text: {scene.narration}\n")
        
        audio_file = audio_dir / f"scene_{scene.scene_number}_expressive.mp3"
        
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice=narrator_id,
                input=scene.narration
            )
            response.stream_to_file(audio_file)
            print(f"✅ Success! Saved to {audio_file}\n")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_first_two_scenes()
