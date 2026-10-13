import os
import sys
import json
import base64
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.engines.video.qa_inspector import inspect_image, QAReport
from src.engines.video.prompt_refiner import refine_prompt_with_agent

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX-2-klein-9b")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def raw_generate_image(prompt: str, output_path: Path, width: int = 1280, height: int = 720) -> bool:
    """
    Generates a 16:9 illustration using DeepInfra's FLUX models via resilient curl.
    """
    if not API_KEY:
        raise ValueError("LLM_API_KEY is not set in .env")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"prompt": prompt, "width": width, "height": height})

    # DeepInfra Direct Inference API with curl retry
    for attempt in range(1, 4):
        try:
            cmd = [
                "curl", "-s", "--retry", "3", "--retry-all-errors", "--http1.1",
                f"https://api.deepinfra.com/v1/inference/{IMAGE_MODEL}",
                "-H", f"Authorization: Bearer {API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", payload
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                if "images" in data and len(data["images"]) > 0:
                    img_b64 = data["images"][0].split(",")[-1]
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(img_b64))
                    return True
                elif "error" in data:
                    print(f"      ⚠️ Attempt {attempt} API error: {data['error']}")
            time.sleep(1.0)
        except Exception as e:
            print(f"      ⚠️ Attempt {attempt} exception: {e}")
            time.sleep(1.0)

    # Fallback to OpenAI SDK images.generate
    try:
        size_str = f"{width}x{height}"
        res = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=size_str,
            response_format="b64_json"
        )
        if res.data and res.data[0].b64_json:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(res.data[0].b64_json))
            return True
    except Exception as e:
        print(f"      ⚠️ SDK Fallback failed: {e}")

    return False

def generate_and_verify_single_beat(
    beat_info: Dict[str, Any],
    output_dir: Path,
    enable_qa: bool = True,
    max_qa_retries: int = 2
) -> Dict[str, Any]:
    """
    Worker task: generates a single beat illustration and runs strict QA inspection.
    Uses PromptRefinerAgent to rewrite the prompt when defects are flagged.
    """
    img_filename = beat_info["image"]
    output_path = output_dir / img_filename
    prompt = beat_info["visual_prompt"]
    expected_context = beat_info.get("concept") or beat_info.get("narration_chunk", "")
    expected_text = beat_info.get("expected_text") or beat_info.get("text_overlay")

    # Check if a valid, QA-passing file already exists in THIS specific output directory
    if output_path.exists() and output_path.stat().st_size > 10000:
        if enable_qa:
            report = inspect_image(output_path, expected_context=expected_context, expected_text=expected_text)
            if report.status == "PASS" and report.anatomy_valid and report.physics_valid and report.text_valid:
                print(f"   ⏩ [Cached & Verified]: {img_filename} passed QA.")
                return {"image": img_filename, "success": True, "qa_report": report}
            else:
                print(f"   ⚠️ Existing {img_filename} failed QA ({report.defects}), regenerating...")
        else:
            print(f"   ⏩ [Cached]: Reusing existing {img_filename}.")
            return {"image": img_filename, "success": True, "qa_report": None}

    current_prompt = prompt
    last_report = None
    width = beat_info.get("width", 1280)
    height = beat_info.get("height", 720)

    for attempt in range(1, max_qa_retries + 2):
        print(f"   🎨 Generating {img_filename} ({width}x{height}) (Attempt {attempt}/{max_qa_retries + 1})...")
        ok = raw_generate_image(current_prompt, output_path, width=width, height=height)
        if not ok:
            print(f"      ❌ Raw image generation failed for {img_filename}")
            continue

        if not enable_qa:
            return {"image": img_filename, "success": True, "qa_report": None}

        report = inspect_image(output_path, expected_context=expected_context, expected_text=expected_text)
        last_report = report

        if report.status == "PASS" and report.anatomy_valid and report.physics_valid and report.text_valid and report.context_valid:
            print(f"      ✅ [QA PASS] {img_filename}: {report.details[:80]}...")
            return {"image": img_filename, "success": True, "qa_report": report}
        else:
            print(f"      ❌ [QA FAIL - Attempt {attempt}] {img_filename}: Defects={report.defects}")
            print(f"         Details: {report.details[:90]}...")

            if attempt <= max_qa_retries:
                # Agentic prompt restructuring to eliminate defects at the root cause
                refinement = refine_prompt_with_agent(
                    original_prompt=current_prompt,
                    concept=beat_info.get("concept", ""),
                    narration_chunk=beat_info.get("narration_chunk", ""),
                    qa_report=report,
                    expected_text=expected_text,
                    attempt=attempt + 1
                )
                current_prompt = refinement.get("refined_prompt", current_prompt)
                time.sleep(1.0)

    # If all attempts exhausted, strictly evaluate pass state
    is_passed = bool(last_report and last_report.status == "PASS" and last_report.anatomy_valid and last_report.physics_valid and last_report.text_valid)
    if is_passed:
        print(f"   ✅ {img_filename} accepted.")
    else:
        print(f"   ❌ {img_filename} failed strict QA after {max_qa_retries + 1} attempts. Defects: {last_report.defects if last_report else 'No report'}")

    return {"image": img_filename, "success": is_passed and output_path.exists(), "qa_report": last_report}

def generate_storyboard_images(
    storyboard_path: Path,
    output_dir: Path,
    max_images: Optional[int] = None,
    enable_qa: bool = True,
    max_workers: int = 3,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Generates all storyboard illustrations in parallel with real-time Multimodal QA inspection.
    """
    print("=========================================================================")
    print(f"🎨 Parallel Image Generation & QA Engine ({IMAGE_MODEL})")
    print(f"🛡️ QA Inspector: {'ENABLED (Qwen3-VL-30B Parallel)' if enable_qa else 'DISABLED'}")
    print(f"⚡ Concurrency: {max_workers} worker threads | Dry Run: {dry_run}")
    print("=========================================================================")

    if not storyboard_path.exists():
        raise FileNotFoundError(f"Storyboard not found at {storyboard_path}")

    with open(storyboard_path, "r") as f:
        board = json.load(f)

    art_style = board.get("art_style", "OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, simple cartoon stick characters with round white faces and expressive eyes, thin stick limbs, exactly two arms and two legs, clear visual metaphors with diagram arrows, zero text, zero pseudo-words")
    scenes = board.get("scenes", [])
    width = board.get("width", 1280)
    height = board.get("height", 720)
    aspect_ratio = board.get("aspect_ratio", "16:9")
    ratio_clause = "vertical 9:16 mobile portrait ratio, centered vertical composition" if aspect_ratio == "9:16" else "16:9 widescreen"

    tasks = []
    for s in scenes:
        s_num = s["scene_number"]
        for b in s.get("beats", []):
            b_idx = b["beat_index"]
            img_name = b.get("image") or f"shot_s{s_num}_b{b_idx}.jpg"
            raw_prompt = b.get("visual_prompt", "")
            text_clause = (
                "Completely clean surfaces with absolutely zero text, zero letters, zero words, "
                "zero numbers, zero signs, zero labels, and zero scribbles. "
                "Pure visual diagram storytelling with arrows, cutaways, and cross-sections."
            )

            full_prompt = (
                f"{art_style}. {raw_prompt}. {text_clause} "
                f"{ratio_clause}, clean vector art, vibrant cel-shading, bold black outlines, "
                f"clean composition, exactly two arms, two legs, normal cartoon stick hands, no extra limbs, "
                f"no hands sticking out of closed glass, no clipping through solid objects, "
                f"no pseudo-text, no unreadable letters, completely clean background."
            )

            tasks.append({
                "scene_number": s_num,
                "beat_index": b_idx,
                "image": img_name,
                "visual_prompt": full_prompt,
                "concept": b.get("concept", ""),
                "narration_chunk": b.get("narration_chunk", ""),
                "expected_text": None,
                "width": width,
                "height": height
            })

    if max_images:
        tasks = tasks[:max_images]

    print(f"📋 Total Tasks Prepared: {len(tasks)} images in target directory: {output_dir}\n")

    if dry_run:
        print("🔍 [DRY RUN ACTIVE] Pre-flight verification summary:")
        for t in tasks:
            print(f"   • Scene {t['scene_number']} Beat {t['beat_index']} -> {t['image']}")
            print(f"     Prompt: {t['visual_prompt'][:90]}...")
            print(f"     Concept: {t['concept']}")
        print(f"\n✅ Dry run complete. {len(tasks)} images planned. No API credits consumed.")
        return {"total": len(tasks), "dry_run": True}

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(generate_and_verify_single_beat, t, output_dir, enable_qa): t["image"]
            for t in tasks
        }
        for future in as_completed(future_map):
            img_name = future_map[future]
            try:
                res = future.result()
                results[img_name] = res
            except Exception as e:
                print(f"   ❌ Error generating {img_name}: {e}")
                results[img_name] = {"image": img_name, "success": False, "error": str(e)}

    successful = sum(1 for r in results.values() if r.get("success"))
    passed_qa = sum(1 for r in results.values() if r.get("qa_report") and r["qa_report"].status == "PASS")
    print(f"\n=========================================================================")
    print(f"🎉 Visual Generation Complete: {successful}/{len(tasks)} generated.")
    if enable_qa:
        print(f"🛡️ QA Verification: {passed_qa}/{successful} passed inspection.")
    print(f"📁 Output directory: {output_dir}")
    print("=========================================================================")

    return results

if __name__ == "__main__":
    board_p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/storyboard_longform.json")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/images")
    generate_storyboard_images(board_p, out_dir, dry_run=True)
