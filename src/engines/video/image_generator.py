import os
import sys
import json
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX-1-schnell")
DEEPINFRA_URL = f"https://api.deepinfra.com/v1/inference/{IMAGE_MODEL}"

def generate_image(prompt: str, output_path: Path, width: int = 1280, height: int = 720) -> bool:
    """
    Generates a 16:9 illustration using DeepInfra's FLUX-1-schnell model.
    """
    if not API_KEY:
        raise ValueError("LLM_API_KEY is not set in .env")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height
    }

    try:
        response = requests.post(DEEPINFRA_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            images = result.get("images", [])
            if images:
                b64_data = images[0].split(",")[-1]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                print(f"      ✅ Saved: {output_path} ({os.path.getsize(output_path)} bytes)")
                return True
        print(f"      ❌ Failed: HTTP {response.status_code} - {response.text[:200]}")
        return False
    except Exception as e:
        print(f"      ❌ Exception during image generation: {e}")
        return False

def generate_storyboard_images(storyboard_path: Path, output_dir: Path, max_images: int = None):
    print("=========================================")
    print(f"🎨 Image Generation Engine ({IMAGE_MODEL})")
    print("=========================================")

    if not storyboard_path.exists():
        raise FileNotFoundError(f"Storyboard not found at {storyboard_path}")

    with open(storyboard_path, "r") as f:
        board = json.load(f)

    art_style = board.get("art_style_prompt", "Clean 2D cel-shaded comic illustration, 4px crisp black linework, vibrant flat colors")
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
            full_prompt = f"{art_style}. {raw_prompt}"
            
            img_file = output_dir / f"scene_{scene_num}_beat_{beat_idx}.jpg"
            print(f"--- Scene {scene_num} | Beat {beat_idx} ---")
            print(f"Prompt: {full_prompt[:120]}...")
            
            success = generate_image(full_prompt, img_file)
            if success:
                count += 1
                
            if max_images and count >= max_images:
                print(f"\n🎉 Successfully reached target limit of {max_images} image(s)!")
                return

    print(f"\n🎉 Complete! Generated {count} images in {output_dir}")

if __name__ == "__main__":
    board_file = Path("data/storyboard.json")
    out_folder = Path("data/images")
    # Default to generating first 2 images
    generate_storyboard_images(board_file, out_folder, max_images=2)
