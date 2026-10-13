import json
from pathlib import Path
from mutagen.mp3 import MP3

def build_longform_storyboard():
    script_path = Path("data/fermi_paradox_longform.json")
    audio_dir = Path("data/audio")
    output_storyboard = Path("data/storyboard_longform.json")

    with open(script_path, "r") as f:
        script = json.load(f)

    storyboard = {
        "title": script.get("title", "The Cosmic Silence: Unraveling the Fermi Paradox"),
        "topic": "Fermi Paradox",
        "target_platform": "YouTube Longform (5 to 8 Minutes)",
        "art_style": "OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, chibi characters embedded in scene, exactly two arms, two legs, no pseudo-text",
        "scenes": []
    }

    total_duration = 0.0
    total_beats = 0

    for s in script["scenes"]:
        s_num = s["scene_number"]
        audio_file = audio_dir / f"scene_{s_num}.mp3"
        if not audio_file.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_file}")

        mp3 = MP3(audio_file)
        scene_dur = round(mp3.info.length, 2)
        total_duration += scene_dur

        raw_beats = s.get("beats", [])
        num_beats = len(raw_beats)
        total_beats += num_beats

        # Distribute scene duration evenly across beats, with rounding on last beat
        base_beat_dur = round(scene_dur / num_beats, 2)
        accumulated = 0.0

        scene_beats = []
        for i, b in enumerate(raw_beats):
            b_idx = b.get("beat_index", i + 1)
            start_t = round(accumulated, 2)
            if i == num_beats - 1:
                beat_dur = round(scene_dur - accumulated, 2)
            else:
                beat_dur = base_beat_dur
                accumulated += beat_dur
            end_t = round(start_t + beat_dur, 2)

            image_name = f"shot_{s_num}_{b_idx}.jpg"

            scene_beats.append({
                "beat_index": b_idx,
                "start_time": start_t,
                "end_time": end_t,
                "duration": beat_dur,
                "narration": b.get("narration", ""),
                "visual_prompt": b.get("visual_prompt", ""),
                "image": image_name,
                "camera_motion": b.get("camera_motion", "SLOW_ZOOM_IN")
            })

        storyboard["scenes"].append({
            "scene_number": s_num,
            "title": f"Scene {s_num}",
            "audio_file": f"scene_{s_num}.mp3",
            "duration": scene_dur,
            "beats": scene_beats
        })

    with open(output_storyboard, "w") as f:
        json.dump(storyboard, f, indent=2)

    print("==================================================")
    print("🎬 Longform Storyboard Generated Successfully!")
    print(f"📁 Saved to: {output_storyboard}")
    print(f"⏱️ Total Video Runtime: {total_duration:.2f}s ({total_duration/60:.2f} mins)")
    print(f"🎞️ Total Scenes: {len(storyboard['scenes'])} | Total Shots/Beats: {total_beats}")
    print("==================================================")

if __name__ == "__main__":
    build_longform_storyboard()
