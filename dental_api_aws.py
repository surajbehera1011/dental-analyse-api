#!/usr/bin/env python3
"""
DIAL - Dental Image Analysis API
Optimized for AWS g4dn.xlarge (single T4 GPU)
"""
import os
import io
import re
from typing import List, Dict

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =============================================================
# Configuration
# =============================================================
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER = "hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo"

# =============================================================
# Load Model (runs once at startup)
# =============================================================
print("Loading model...")
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"Applying LoRA adapter: {ADAPTER}...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)

print(f"✓ Model loaded! VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# =============================================================
# FastAPI App
# =============================================================
app = FastAPI(title="DIAL — Dental Image Analysis API", version="8.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEFAULT_PROMPT = "Which FDI tooth numbers have issues as seen in this image? For each tooth, describe the condition and severity."


def run_inference(img: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.9)
    
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": ADAPTER,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), question: str = Form(default="")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {file.content_type}")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {str(e)}")

    # Resize large images
    max_dim = 1280
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.LANCZOS)

    prompt = question.strip() if question.strip() else DEFAULT_PROMPT
    raw_output = run_inference(image, prompt, max_new_tokens=800)
    torch.cuda.empty_cache()

    findings = parse_model_output(raw_output)
    sevs = [f["severity"] for f in findings]
    overall = "severe" if "severe" in sevs else "moderate" if "moderate" in sevs else "mild" if sevs else "unknown"

    return {
        "model": ADAPTER,
        "image_size": f"{image.size[0]}x{image.size[1]}",
        "overall_severity": overall,
        "findings": findings,
        "total_findings": len(findings),
        "note": "Non-diagnostic. Educational purposes only.",
        "_raw": raw_output,
    }


# =============================================================
# Parser Functions
# =============================================================
def is_valid_fdi(num: int) -> bool:
    return isinstance(num, int) and (num // 10) in (1, 2, 3, 4) and 1 <= (num % 10) <= 8


def compute_fdi(description: str) -> int:
    d = description.lower()
    quadrant = 0
    if ("upper" in d or "maxillar" in d) and "right" in d: quadrant = 1
    elif ("upper" in d or "maxillar" in d) and "left" in d: quadrant = 2
    elif ("lower" in d or "mandibul" in d) and "left" in d: quadrant = 3
    elif ("lower" in d or "mandibul" in d) and "right" in d: quadrant = 4
    pos = 0
    if "central" in d and "incisor" in d: pos = 1
    elif "lateral" in d and "incisor" in d: pos = 2
    elif "canine" in d: pos = 3
    elif "first" in d and "premolar" in d: pos = 4
    elif "second" in d and "premolar" in d: pos = 5
    elif "first" in d and "molar" in d: pos = 6
    elif "second" in d and "molar" in d: pos = 7
    elif "third" in d or "wisdom" in d: pos = 8
    elif "molar" in d and "pre" not in d: pos = 6
    elif "premolar" in d: pos = 4
    elif "incisor" in d: pos = 1
    return quadrant * 10 + pos if quadrant and pos else 0


def extract_condition(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["periapical", "apical"]) and any(w in t for w in ["lesion", "radiolucen", "abscess"]):
        return "periapical lesion"
    conditions = [
        (["necrotic", "necrosis", "pulp necrosis", "non-vital"], "pulp necrosis"),
        (["cyst", "cystic"], "periapical cyst"),
        (["abscess"], "abscess"),
        (["granuloma"], "periapical granuloma"),
        (["bone loss", "bone destruction"], "bone loss"),
        (["periodontal"], "periodontal disease"),
        (["deep caries", "deep cavity", "deep decay", "severe caries"], "deep caries"),
        (["caries", "carious", "cavity", "decay"], "caries"),
        (["impacted", "unerupted"], "impacted"),
        (["fracture", "crack", "broken"], "fracture"),
        (["missing", "absent", "extracted"], "missing"),
        (["radiolucen"], "radiolucency"),
        (["resorption"], "resorption"),
        (["no visible decay", "intact", "not affected", "no abnormal", "healthy", "normal"], "healthy"),
    ]
    for keywords, condition in conditions:
        if any(w in t for w in keywords):
            return condition
    return "abnormality"


def extract_severity(text: str) -> str:
    t = text.lower()
    if "severe" in t and "moderate" not in t: return "severe"
    if "moderate to severe" in t or "moderate-to-severe" in t: return "severe"
    if "moderate" in t: return "moderate"
    if any(w in t for w in ["mild", "minor", "slight", "early"]): return "mild"
    if any(w in t for w in ["large", "significant", "necrotic", "necrosis", "non-vital", "pulp", "abscess", "destruction", "extensive"]): return "severe"
    if any(w in t for w in ["small", "minimal", "incipient", "pit", "tiny", "early stages", "confined to the enamel"]): return "mild"
    return "moderate"


def extract_tooth_sections(raw: str) -> List[Dict]:
    sections = []
    parts = re.split(r'(?=\*\*FDI\s*#?\s*\d{2}\s*\()', raw)
    for part in parts:
        header = re.match(r'\*\*FDI\s*#?\s*(\d{2})\s*\(([^)]+)\)[:\s]*\*\*:?', part)
        if not header: continue
        model_num = int(header.group(1))
        description = header.group(2).strip()
        section_text = part[header.end():]
        cond_match = re.search(r'\*?\*?Condition\*?\*?\s*:\s*(.+?)(?:\n|$)', section_text, re.IGNORECASE)
        condition_text = cond_match.group(1).strip() if cond_match else section_text.split('\n')[0].strip()
        sev_match = re.search(r'\*?\*?Severity\*?\*?\s*:\s*(.+?)(?:\*\*|\.|\n|$)', section_text, re.IGNORECASE)
        severity_text = sev_match.group(1).strip() if sev_match else condition_text
        sections.append({"model_number": model_num, "description": description, "condition_text": condition_text, "severity_text": severity_text})
    if sections: return sections
    
    # Fallback patterns
    for match in re.finditer(r'FDI\s*#?\s*(\d{2})(?:\s*\(([^)]+)\))?', raw, re.IGNORECASE):
        num = int(match.group(1))
        if not is_valid_fdi(num): continue
        sections.append({"model_number": num, "description": match.group(2) or "", "condition_text": raw[match.end():match.end()+300], "severity_text": raw[match.end():match.end()+300]})
    return sections


def parse_model_output(raw: str) -> List[Dict]:
    findings = []
    for section in extract_tooth_sections(raw):
        fdi = compute_fdi(section["description"]) or (section["model_number"] if is_valid_fdi(section["model_number"]) else 0)
        if not is_valid_fdi(fdi): continue
        condition = extract_condition(section["condition_text"])
        if condition == "healthy": continue
        findings.append({"tooth_number": fdi, "condition": condition, "severity": extract_severity(section["severity_text"]), "description": re.sub(r'\*\*', '', section["condition_text"]).strip()[:300]})
    seen = {}
    for f in findings:
        if f["tooth_number"] not in seen: seen[f["tooth_number"]] = f
    return sorted(seen.values(), key=lambda x: x["tooth_number"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
