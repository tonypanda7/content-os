import os
import sys
import json
import wave
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
    print("🎙️ Generating First 2 Scenes with Qwen 3 TTS")
    print("=========================================")
    
    script_path = Path("data/expressive_script.json")
    with open(script_path, "r") as f:
        script_data = json.load(f)
    
    script = VideoScript(**script_data)
    
    audio_dir = Path("data/audio_test")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    voice = os.getenv("NARRATOR_VOICE", "Ryan")
    print(f"Using Qwen 3 TTS (Voice: {voice})\n")
    
    for scene in script.scenes[:2]:
        print(f"--- Generating Scene {scene.scene_number} ---")
        print(f"Narration: {scene.narration}\n")
        
        audio_file = audio_dir / f"scene_{scene.scene_number}_qwen3.wav"
        
        try:
            response = client.audio.speech.create(
                model="Qwen/Qwen3-TTS",
                voice=voice,
                input=scene.narration,
                extra_body={
                    "instructions": (
                        "Speak with dramatic storytelling energy, lively inflections, "
                        "dynamic pauses, and animated cartoon narration cadence."
                    )
                }
            )
            response.stream_to_file(audio_file)
            
            with wave.open(str(audio_file), "rb") as wf:
                duration = round(wf.getnframes() / float(wf.getframerate()), 2)
                
            print(f"✅ Success! Saved to {audio_file} (Duration: {duration}s)\n")
        except Exception as e:
            print(f"❌ Failed: {e}\n")

if __name__ == "__main__":
    test_first_two_scenes()
