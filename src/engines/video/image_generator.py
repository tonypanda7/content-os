import os
import sys
import json
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX-1-schnell")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def generate_image(prompt: str, output_path: Path, width: int = 1280, height: int = 720) -> bool:
    """
    Generates a 16:9 illustration using DeepInfra's FLUX-1-schnell model via OpenAI SDK.
    """
    if not API_KEY:
        raise ValueError("LLM_API_KEY is not set in .env")

    size_str = f"{width}x{height}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, 4):
        try:
            res = client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=size_str,
                response_format="b64_json"
            )
            if res.data and res.data[0].b64_json:
                b64_data = res.data[0].b64_json
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                print(f"      ✅ Saved ({output_path.name}): {os.path.getsize(output_path)} bytes")
                return True
        except Exception as e:
            print(f"      ⚠️ Attempt {attempt} failed: {e}")

    print(f"      ❌ Failed to generate {output_path.name} after 3 attempts.")
    return False

def generate_storyboard_images(storyboard_path: Path, output_dir: Path, max_images: int = None):
    print("=========================================")
    print(f"🎨 Infographic Image Generation Engine ({IMAGE_MODEL})")
    print("=========================================")

    if not storyboard_path.exists():
        raise FileNotFoundError(f"Storyboard not found at {storyboard_path}")

    with open(storyboard_path, "r") as f:
        board = json.load(f)

    art_style = board.get("art_style_prompt", "Clean 2D cel-shaded comic infographic illustration, 4px crisp black linework, vibrant flat colors, graphic novel aesthetic")
    scenes = board.get("scenes", [])
    
    count = 0
    total_beats = sum(len(s.get("beats", [])) for s in scenes)
    target = max_images or total_beats
    print(f"[Image Engine]: Processing {target} image(s) from storyboard...\n")

    for s in scenes:
        scene_num = s.get("scene_number", 1)
        beats = s.get("beats", [])
        
        for b in beats:
            beat_idx = b.get("beat_index", 1)
            raw_prompt = b.get("background_prompt", "")
            
            # Combine global style anchor with beat-specific scene prompt
            full_prompt = f"{art_style}. {raw_prompt}. High detail, clear educational infographic elements, comic art style, clean composition, 16:9 widescreen."
            
            img_file = output_dir / f"scene_{scene_num}_beat_{beat_idx}.jpg"
            print(f"--- Scene {scene_num} | Beat {beat_idx} ---")
            print(f"Prompt: {full_prompt[:140]}...")
            
            success = generate_image(full_prompt, img_file, width=1280, height=720)
            if success:
                count += 1
                
            if max_images and count >= max_images:
                print(f"\n🎉 Successfully reached target limit of {max_images} image(s)!")
                return

    print(f"\n🎉 Complete! Generated {count} images in {output_dir}")

if __name__ == "__main__":
    board_file = Path("data/storyboard.json")
    out_folder = Path("data/images")
    generate_storyboard_images(board_file, out_folder, max_images=9)
