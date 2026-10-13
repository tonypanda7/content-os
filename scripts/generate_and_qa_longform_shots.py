import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engines.video.image_generator import generate_image, raw_generate_image
from src.engines.video.qa_inspector import inspect_image, QAReport

def main():
    storyboard_path = Path("data/storyboard_longform.json")
    images_dir = Path("data/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(storyboard_path, "r") as f:
        board = json.load(f)

    art_style = board.get("art_style", "OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, chibi characters embedded in scene, exactly two arms, two legs, no pseudo-text")
    scenes = board.get("scenes", [])

    total_shots = sum(len(s.get("beats", [])) for s in scenes)
    print("=================================================================")
    print("🎨 LONGFORM VISUAL GENERATION & QA INSPECTOR PIPELINE")
    print(f"🎬 Total Scenes: {len(scenes)} | Total Shots: {total_shots}")
    print("🛡️ QA Inspector: Qwen3-VL-30B (Zero Defects Enforcement)")
    print("=================================================================\n")

    qa_results = {}
    shot_index = 0

    for s in scenes:
        s_num = s["scene_number"]
        beats = s.get("beats", [])

        for b in beats:
            shot_index += 1
            b_idx = b["beat_index"]
            img_filename = b.get("image", f"shot_{s_num}_{b_idx}.jpg")
            img_path = images_dir / img_filename
            raw_prompt = b.get("visual_prompt", "")

            full_prompt = (
                f"{art_style}. {raw_prompt}. "
                f"16:9 widescreen, clean vector art, vibrant cel-shading, bold black outlines, "
                f"clean composition, exactly two arms, two legs, no extra limbs, no mutated hands, "
                f"no pseudo-text, no unreadable letters, completely clean background."
            )

            print(f"[{shot_index}/{total_shots}] Scene {s_num} Beat {b_idx} -> {img_filename}")

            # Check if file exists and passes QA already
            need_generation = True
            if img_path.exists() and img_path.stat().st_size > 10000:
                report = inspect_image(img_path)
                if report.status == "PASS" and report.anatomy_valid and report.text_valid:
                    print(f"   ✨ Existing image passed QA: {report.details[:80]}...")
                    qa_results[img_filename] = report
                    need_generation = False
                else:
                    print(f"   ⚠️ Existing image failed QA ({report.defects}), regenerating...")

            if need_generation:
                # Generate with automatic QA retry loop
                success = False
                current_prompt = full_prompt

                for qa_attempt in range(1, 4):
                    print(f"   🖌️ Generating (Attempt {qa_attempt}/3)...")
                    gen_ok = raw_generate_image(current_prompt, img_path, width=1280, height=720)
                    if not gen_ok:
                        print(f"      ❌ Raw generation failed on attempt {qa_attempt}")
                        time.sleep(1.0)
                        continue

                    # Inspect newly generated image
                    report = inspect_image(img_path)
                    print(f"      [QA Check]: Status={report.status} | Anatomy={report.anatomy_valid} | Text={report.text_valid}")
                    if report.status == "PASS" and report.anatomy_valid and report.text_valid:
                        print(f"      ✅ QA PASSED: {report.details[:80]}...")
                        qa_results[img_filename] = report
                        success = True
                        break
                    else:
                        print(f"      ❌ QA REJECTED: Defects={report.defects}")
                        print(f"         Details: {report.details[:100]}...")
                        remedy = ""
                        if not report.text_valid:
                            remedy += " Absolutely zero lettering, no signs, no pseudo-words, blank clean surfaces."
                        if not report.anatomy_valid:
                            remedy += " Perfect cartoon anatomy, exactly two arms, two legs, one head, no extra body parts."
                        current_prompt = f"{full_prompt}. {remedy}"
                        time.sleep(1.0)

                if not success:
                    # Final fallback inspection record
                    rep = inspect_image(img_path)
                    qa_results[img_filename] = rep
                    print(f"   ⚠️ Retained current generation for {img_filename} (Status: {rep.status})")

    # Final Audit Report
    print("\n=================================================================")
    print("📋 FINAL QA INSPECTOR AUDIT REPORT")
    print("=================================================================")
    passed = sum(1 for r in qa_results.values() if r.status == "PASS")
    failed = sum(1 for r in qa_results.values() if r.status != "PASS")
    print(f"Total Images: {len(qa_results)} | Passed: {passed} | Failed: {failed}")
    for fname, r in qa_results.items():
        status_mark = "✅ PASS" if r.status == "PASS" else "❌ FAIL"
        print(f"  [{status_mark}] {fname} -> {r.details[:70]}...")

    with open("data/qa_audit_report.json", "w") as f:
        json.dump({k: v.model_dump() for k, v in qa_results.items()}, f, indent=2)

    print("\n📄 Detailed QA Report saved to data/qa_audit_report.json")

if __name__ == "__main__":
    main()
