!pip install -q --no-warn-conflicts \
    transformers>=4.51.0 \
    accelerate>=0.28.0 \
    bitsandbytes>=0.46.1 \
    peft>=0.18.0 \
    fastapi>=0.110.0 \
    uvicorn>=0.27.0 \
    pyngrok>=7.0.0 \
    qwen-vl-utils \
    Pillow>=10.0.0

from huggingface_hub import login
from pyngrok import ngrok, conf

# ⚠️ REPLACE THESE with your actual tokens
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"
NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"

login(token=HF_TOKEN)
conf.get_default().auth_token = NGROK_TOKEN

print("✓ HuggingFace authenticated")
print("✓ ngrok configured")

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# Clear any leftover GPU memory from previous attempts
torch.cuda.empty_cache()

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER = "hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Explicitly balance across both T4 GPUs + allow CPU overflow
print(f"Loading base model: {BASE_MODEL}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    max_memory={0: "12GiB", 1: "12GiB", "cpu": "20GiB"},
)

print(f"Applying LoRA adapter: {ADAPTER}...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,  # Reduced from 768 to save more VRAM for inference
)

print(f"✓ Model + adapter loaded")

for i in range(torch.cuda.device_count()):
    mem = torch.cuda.memory_allocated(i) / 1e9
    print(f"  GPU {i}: ~{mem:.1f} GB used")

from PIL import Image
from qwen_vl_utils import process_vision_info

def run_inference(img: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]

print("✓ run_inference() defined — ready to build API")

# =============================================================
# DIAL — Dental Image Analysis API
# Model: hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo
# =============================================================
import io
import re
from typing import List, Dict
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DIAL — Dental Image Analysis API", version="8.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEFAULT_PROMPT = "Which FDI tooth numbers have issues as seen in this image? For each tooth, describe the condition and severity."


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": ADAPTER,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(default="", description="Leave empty for default dental analysis"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {file.content_type}")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {str(e)}")

    # Resize large images to speed up inference
    max_dim = 1280
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    prompt = question.strip() if question and question.strip() else DEFAULT_PROMPT

    # --- Inference ---
    raw_output = run_inference(image, prompt, max_new_tokens=800)
    torch.cuda.empty_cache()

    # --- Parse ---
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
# PARSER
# =============================================================
def parse_model_output(raw: str) -> List[Dict]:
    findings = []
    tooth_sections = extract_tooth_sections(raw)

    for section in tooth_sections:
        fdi = compute_fdi(section["description"])
        if not fdi:
            fdi = section["model_number"] if is_valid_fdi(section["model_number"]) else 0
        if not is_valid_fdi(fdi):
            continue

        condition = extract_condition(section["condition_text"])
        severity = extract_severity(section["severity_text"])

        if condition == "healthy":
            continue

        findings.append({
            "tooth_number": fdi,
            "condition": condition,
            "severity": severity,
            "description": clean_text(section["condition_text"])[:300],
        })

    # Deduplicate by tooth number
    seen = {}
    for f in findings:
        if f["tooth_number"] not in seen:
            seen[f["tooth_number"]] = f
    result = list(seen.values())
    result.sort(key=lambda x: x["tooth_number"])
    return result


def extract_tooth_sections(raw: str) -> List[Dict]:
    sections = []

    parts = re.split(r'(?=\*\*FDI\s*#?\s*\d{2}\s*\()', raw)
    for part in parts:
        # Match header — handles all variations
        header = re.match(r'\*\*FDI\s*#?\s*(\d{2})\s*\(([^)]+)\)[:\s]*\*\*:?', part)
        if not header:
            continue

        model_num = int(header.group(1))
        description = header.group(2).strip()
        section_text = part[header.end():]

        # Extract condition text
        cond_match = re.search(r'\*?\*?Condition\*?\*?\s*:\s*(.+?)(?:\n|$)', section_text, re.IGNORECASE)
        if cond_match:
            condition_text = cond_match.group(1).strip()
        else:
            # Inline format — grab text before **Severity:
            sev_pos = re.search(r'\*\*Severity', section_text, re.IGNORECASE)
            condition_text = section_text[:sev_pos.start()].strip() if sev_pos else section_text.split('\n')[0].strip()

        # Extract severity text
        sev_match = re.search(r'\*?\*?Severity\*?\*?\s*:\s*(.+?)(?:\*\*|\.|\n|$)', section_text, re.IGNORECASE)
        severity_text = sev_match.group(1).strip() if sev_match else condition_text

        sections.append({
            "model_number": model_num,
            "description": description,
            "condition_text": condition_text,
            "severity_text": severity_text,
        })

    if sections:
        return sections

    # ==========================================================
    # FORMAT B: **Tooth #XX (Description)** – condition text
    # ==========================================================
    for match in re.finditer(
        r'\*?\*?Tooth\s*#?\s*(\d+)\s*\(([^)]+)\)\*?\*?\s*[–\-—:]+\s*(.+?)(?:\n|$)',
        raw, re.IGNORECASE
    ):
        sections.append({
            "model_number": int(match.group(1)),
            "description": match.group(2).strip(),
            "condition_text": match.group(3).strip(),
            "severity_text": match.group(3).strip(),
        })

    if sections:
        return sections

    # ==========================================================
    # FORMAT C: **Description** → **FDI XX**
    # ==========================================================
    fdi_map = {}
    for match in re.finditer(
        r'\*?\*?((?:Maxillary|Mandibular|Upper|Lower)\s+(?:right|left)\s+[\w\s]+?)\*?\*?\s*[→]+\s*\*?\*?FDI\s*(\d{2})\*?\*?',
        raw, re.IGNORECASE
    ):
        fdi_map[int(match.group(2))] = match.group(1).strip()

    if fdi_map:
        for num, desc in fdi_map.items():
            search = re.search(re.escape(desc[:20]), raw, re.IGNORECASE)
            ctx = raw[search.end():search.end()+300] if search else raw
            sections.append({
                "model_number": num,
                "description": desc,
                "condition_text": ctx,
                "severity_text": ctx,
            })
        return sections

    # ==========================================================
    # FALLBACK: Any FDI XX mention with context
    # ==========================================================
    for match in re.finditer(r'FDI\s*#?\s*(\d{2})(?:\s*\(([^)]+)\))?', raw, re.IGNORECASE):
        num = int(match.group(1))
        if not is_valid_fdi(num):
            continue
        # Skip quadrant explanation lines
        line_start = raw.rfind('\n', 0, match.start()) + 1
        line_end = raw.find('\n', match.end())
        line = raw[line_start:line_end if line_end != -1 else len(raw)]
        if "quadrant" in line.lower() and "=" in line:
            continue
        if "summary" in raw[max(0, line_start-50):line_start].lower():
            continue

        description = match.group(2).strip() if match.group(2) else ""
        # Context until next FDI or ---
        ctx_start = match.end()
        next_fdi = re.search(r'(?:\n---|\*\*FDI\s*#?\s*\d{2})', raw[ctx_start:])
        ctx_end = ctx_start + next_fdi.start() if next_fdi else min(ctx_start + 400, len(raw))
        context = raw[ctx_start:ctx_end]

        if re.search(r'not\s+affected|not\s+visible|appears?\s+intact', context, re.IGNORECASE):
            continue

        sections.append({
            "model_number": num,
            "description": description,
            "condition_text": context,
            "severity_text": context,
        })

    return sections


# =============================================================
# HELPERS
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
    if any(w in t for w in ["necrotic", "necrosis", "pulp necrosis", "non-vital"]):
        return "pulp necrosis"
    if any(w in t for w in ["cyst", "cystic"]):
        return "periapical cyst"
    if any(w in t for w in ["periapical", "apical"]) and any(w in t for w in ["lesion", "radiolucen", "abscess"]):
        return "periapical lesion"
    if "abscess" in t:
        return "abscess"
    if "granuloma" in t:
        return "periapical granuloma"
    if any(w in t for w in ["bone loss", "bone destruction"]):
        return "bone loss"
    if "periodontal" in t:
        return "periodontal disease"
    if any(w in t for w in ["deep caries", "deep cavity", "deep decay", "severe caries"]):
        return "deep caries"
    if any(w in t for w in ["caries", "carious", "cavity", "decay"]):
        return "caries"
    if any(w in t for w in ["impacted", "unerupted"]):
        return "impacted"
    if any(w in t for w in ["fracture", "crack", "broken"]):
        return "fracture"
    if any(w in t for w in ["missing", "absent", "extracted"]):
        return "missing"
    if any(w in t for w in ["radiolucen"]):
        return "radiolucency"
    if any(w in t for w in ["resorption"]):
        return "resorption"
    if any(w in t for w in ["no visible decay", "intact", "not affected", "no abnormal", "healthy", "normal"]):
        return "healthy"
    return "abnormality"


def extract_severity(text: str) -> str:
    t = text.lower()
    if "severe" in t and "moderate" not in t:
        return "severe"
    if "moderate to severe" in t or "moderate-to-severe" in t:
        return "severe"
    if "moderate" in t:
        return "moderate"
    if "mild" in t or "minor" in t or "slight" in t or "early" in t:
        return "mild"
    if any(w in t for w in ["large", "significant", "necrotic", "necrosis", "non-vital", "pulp", "abscess", "destruction", "extensive"]):
        return "severe"
    if any(w in t for w in ["small", "minimal", "incipient", "pit", "tiny", "early stages", "confined to the enamel"]):
        return "mild"
    if any(w in t for w in ["visible", "carious", "lesion", "progressing", "dentin"]):
        return "moderate"
    return "moderate"


def clean_text(text: str) -> str:
    return re.sub(r'\*\*', '', text).strip()


print("DIAL API v8 ready")
print("  Endpoints: GET /health, POST /analyze")

import uvicorn
import threading
import time

PORT = 8000

# Start server in background thread
def run():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

thread = threading.Thread(target=run, daemon=True)
thread.start()
time.sleep(3)

# Create public tunnel
public_url = ngrok.connect(PORT, "http").public_url

print("=" * 60)
print("🟢 SERVER IS LIVE")
print("=" * 60)
print(f"  Public URL:  {public_url}")
print(f"  Health:      {public_url}/health")
print(f"  Analyze:     POST {public_url}/analyze")
print(f"  Swagger UI:  {public_url}/docs")
print("=" * 60)
print()
print("Share the Public URL with your team!")
print("Note: URL changes each time you restart. Update team when it changes.")