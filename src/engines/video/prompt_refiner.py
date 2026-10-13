import os
import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Ensure project root in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.engines.video.qa_inspector import QAReport

load_dotenv()

REFINER_MODEL = os.getenv("DIRECTOR_MODEL", "Qwen/Qwen2.5-72B-Instruct")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
API_KEY = os.getenv("LLM_API_KEY")

REFINER_SYSTEM_PROMPT = """You are an elite Autonomous Prompt Engineering & Animation Director Agent.
Your duty is to reconstruct and edit Text-to-Image (FLUX) prompts when a Forensic QA Inspector flags visual defects.

You will receive:
1. Original visual prompt
2. Scene concept & narration chunk
3. List of defects caught by the QA Inspector (anatomy, physical impossibility, gibberish text, context)
4. QA inspector details and recommendations
5. Expected text (if any)
6. Attempt number

STRICT REMEDIATION DIRECTIVES:
1. FIX ANATOMY AT THE ROOT:
   - If 3 arms, mutated limbs, or extra hands occurred: DO NOT merely say "no third arm". Restructure the character's physical posture into a stable, canonical pose.
   - Example: Specify "Chibi character cleanly dressed in a green shirt, standing upright with both hands resting naturally at their sides. Normal cartoon proportions, exactly two arms, two legs."
   - Never leave characters naked or ambiguous in attire; specify clean, recognizable clothing.

2. FIX PHYSICAL IMPOSSIBILITIES & CLIPPING:
   - If objects clip through heads or windows (e.g. telescope through head, arm outside spaceship glass):
   - Spatially separate the objects. Example: "A telescope mounted securely on a wooden tripod on the grass, standing beside the character. The character stands comfortably next to the tripod."
   - If in a spaceship: "Character seated safely inside the cockpit with hands on the control console, viewed through the clear cockpit glass with deep space outside."

3. ZERO TOLERANCE FOR TEXT GIBBERISH:
   - If the QA Inspector flagged pseudo-words, unreadable scribbles, or misspellings (e.g. 'GULDITIOCS CN', 'PLABT'):
   - If text IS required: Provide ONLY the exact string in uppercase double quotes on a single clean banner (e.g., 'a clean wooden sign with the exact text "GOLDILOCKS ZONE" in bold white sans-serif letters').
   - ELIMINATE all secondary or confusing labels. Never ask the model to generate multiple labels on different objects in one scene.
   - If text is NOT strictly required, remove all text instructions and mandate: "Completely clean surfaces with zero letters, zero numbers, and zero text. Pure visual storytelling through color and shape."

4. PRESERVE THE OVERSIMPLIFIED ART STYLE & RICH CONTEXT:
   - Retain: "OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, cel-shaded flat colors, chibi characters embedded in scene, 16:9 widescreen".
   - Enrich the educational context, props, and setting so it directly matches the spoken narration.

5. OUTPUT FORMAT:
   Return ONLY a raw JSON object with:
   {
     "refined_prompt": "Complete, cohesive, production-ready FLUX visual prompt...",
     "changes_made": "Brief explanation of how the defect was fixed at the root"
   }
   DO NOT wrap in markdown fences.
"""

def call_refiner_llm(messages: list, max_tokens: int = 600) -> Optional[Dict[str, Any]]:
    """Resilient curl call to Qwen2.5-72B for intelligent prompt restructuring."""
    if not API_KEY:
        return None

    payload = {
        "model": REFINER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        tmp_p = tf.name

    try:
        cmd = [
            "curl", "-s", "--retry", "2", "--http1.1",
            f"{LLM_BASE_URL}/chat/completions",
            "-H", f"Authorization: Bearer {API_KEY}",
            "-H", "Content-Type: application/json",
            "-d", f"@{tmp_p}"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if "choices" in data and len(data["choices"]) > 0:
                raw = data["choices"][0]["message"]["content"].strip()
                if raw.startswith("```json"): raw = raw[7:]
                if raw.startswith("```"): raw = raw[3:]
                if raw.endswith("```"): raw = raw[:-3]
                return json.loads(raw.strip())
    except Exception:
        pass
    finally:
        if os.path.exists(tmp_p):
            try: os.unlink(tmp_p)
            except Exception: pass
    return None

def deterministic_semantic_refine(
    original_prompt: str,
    concept: str,
    narration_chunk: str,
    qa_report: QAReport,
    expected_text: Optional[str] = None,
    attempt: int = 1
) -> Dict[str, Any]:
    """
    High-intelligence deterministic fallback that reconstructs prompts to eliminate
    the root cause of defects even when remote LLM quotas are exhausted.
    """
    changes = []
    style_header = (
        "OverSimplified YouTube 2D cartoon explainer style, clean bold black outlines, "
        "cel-shaded flat colors, simple cartoon stick characters with round white faces and expressive eyes, "
        "thin stick limbs, exactly two arms and two legs, clear visual metaphors with diagram arrows, 16:9 widescreen"
    )

    # 1. Base clean description from concept and narration
    scene_core = concept or narration_chunk or original_prompt

    # 2. Fix Anatomy & Posture
    anatomy_clause = (
        "Character is cleanly dressed in casual clothing (green shirt, blue jeans), "
        "standing upright with hands resting naturally at side. Clean cartoon anatomy, "
        "exactly two arms, two legs, normal hands, one head, zero extra limbs, zero mutations."
    )
    if not qa_report.anatomy_valid or any("arm" in d.lower() or "hand" in d.lower() for d in qa_report.defects):
        changes.append("Enforced stable anatomical pose with hands at side and complete clothing to prevent 3-hand/extra-limb mutations")

    # 3. Fix Physical Impossibility / Clipping
    physics_clause = (
        "Physically realistic layout: character stands firmly on the floor or ground beside objects. "
        "No objects clipping through heads or bodies. No hands protruding through sealed glass or walls."
    )
    if not qa_report.physics_valid or any("clipping" in d.lower() or "protruding" in d.lower() or "glass" in d.lower() for d in qa_report.defects):
        changes.append("Enforced strict physical separation between characters and surrounding equipment/canopies")

    # 4. Fix Text / Typography
    text_clause = ""
    if not qa_report.text_valid or any("gibberish" in d.lower() or "misspell" in d.lower() or "pseudo" in d.lower() for d in qa_report.defects):
        if expected_text and attempt <= 2:
            clean_word = expected_text.strip().upper()
            text_clause = (
                f'A single clean signboard cleanly displays the exact uppercase text "{clean_word}". '
                f'Absolutely zero other text, zero numbers, zero unreadable scribbles anywhere in the scene. '
                f'All other surfaces, planets, and backgrounds are completely blank.'
            )
            changes.append(f'Isolated single clean text "{clean_word}" and purged all secondary pseudo-words')
        else:
            text_clause = (
                "Completely blank surfaces with absolutely zero text, zero letters, zero pseudo-words, "
                "and zero scribbles. Pure visual educational infographic using clean shapes and vibrant color coding."
            )
            changes.append("Eliminated all text labels to guarantee 100% text hygiene without AI pseudo-words")
    elif expected_text:
        text_clause = f'A single clean signboard cleanly displays the exact uppercase text "{expected_text.upper()}". No other text.'

    # 5. Assemble refined prompt
    refined_parts = [
        style_header,
        f"Educational scenario: {scene_core}",
        anatomy_clause,
        physics_clause
    ]
    if text_clause:
        refined_parts.append(text_clause)
    refined_parts.append("Vibrant lighting, crisp vector outlines, masterfully composed 2D explainer scene.")

    refined_prompt = ". ".join(refined_parts)
    return {
        "refined_prompt": refined_prompt,
        "changes_made": "; ".join(changes) if changes else "Reinforced clean anatomical and compositional constraints"
    }

def refine_prompt_with_agent(
    original_prompt: str,
    concept: str,
    narration_chunk: str,
    qa_report: QAReport,
    expected_text: Optional[str] = None,
    attempt: int = 1
) -> Dict[str, Any]:
    """
    Autonomous Prompt Refiner Agent:
    Takes the flawed prompt and the QA Inspector forensic findings, and rewrites the prompt
    to fix the root cause while preserving the original educational narrative.
    """
    print(f"\n🧠 [Prompt Refiner Agent]: Analyzing QA defects on Attempt {attempt}...")
    print(f"   ⚠️ Defects flagged: {qa_report.defects}")
    print(f"   📋 Inspector details: {qa_report.details[:100]}...")

    # Formulate message for LLM
    user_msg = (
        f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
        f"SCENE CONCEPT:\n{concept}\n\n"
        f"NARRATION CHUNK:\n\"{narration_chunk}\"\n\n"
        f"QA DEFECTS FLAGGED:\n{json.dumps(qa_report.defects)}\n\n"
        f"QA DETAILS:\n{qa_report.details}\n\n"
        f"QA FIX RECOMMENDATIONS:\n{qa_report.fix_recommendations}\n\n"
        f"EXPECTED TEXT IN IMAGE (IF ANY):\n{expected_text or 'NONE'}\n\n"
        f"ATTEMPT NUMBER: {attempt}\n\n"
        f"Please rewrite this prompt to eliminate every single defect at the root cause. "
        f"Ensure characters have clean 2-arm anatomy, no objects clipping through heads or glass, "
        f"and absolutely ZERO gibberish or misspelled text."
    )

    messages = [
        {"role": "system", "content": REFINER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]

    llm_res = call_refiner_llm(messages)
    if llm_res and "refined_prompt" in llm_res:
        print(f"   ✨ LLM Prompt Refiner Restructured Prompt Successfully:")
        print(f"      Root Fix: {llm_res.get('changes_made', 'Remediated')}")
        return llm_res

    # Fallback to deterministic semantic refiner
    print("   🛡️ Applying Deterministic Semantic Prompt Refinement...")
    det_res = deterministic_semantic_refine(
        original_prompt=original_prompt,
        concept=concept,
        narration_chunk=narration_chunk,
        qa_report=qa_report,
        expected_text=expected_text,
        attempt=attempt
    )
    print(f"      Root Fix: {det_res['changes_made']}")
    return det_res
