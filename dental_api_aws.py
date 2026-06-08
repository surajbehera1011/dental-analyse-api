#!/usr/bin/env python3
"""
DIAL - Dental Image Analysis API v10
Optimized for AWS g4dn.xlarge (single T4 GPU)
With bounding box detection and confidence scoring
"""
import os
import io
import json
import re
from typing import List, Dict
from collections import Counter, defaultdict

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
app = FastAPI(title="DIAL — Dental Image Analysis API", version="10.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# =============================================================
# PROMPT — Single pass, optimized for Qwen3-VL RLVR grounding
# =============================================================
DEFAULT_PROMPT = (
    "locate every instance that belongs to the following categories: "
    "caries, deep caries, periapical lesion, impacted tooth. "
    "For each abnormal tooth, report bbox coordinates, in JSON format like this: "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "category"}'
)


def run_inference(img: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": ADAPTER,
        "version": "10.0.0",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(default="", description="Leave empty for default dental detection"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Got: {file.content_type}")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {str(e)}")

    original_width, original_height = image.size

    max_dim = 1280
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    proc_width, proc_height = image.size
    prompt = question.strip() if question and question.strip() else DEFAULT_PROMPT

    raw_output = run_inference(image, prompt, max_new_tokens=1024)
    torch.cuda.empty_cache()

    findings = parse_bbox_output(raw_output, proc_width, proc_height, original_width, original_height)

    # Separate actionable findings from informational
    actionable = [f for f in findings if f["condition"] != "impacted"]
    informational = [f for f in findings if f["condition"] == "impacted"]

    sevs = [f["severity"] for f in actionable]
    overall = "severe" if "severe" in sevs else "moderate" if "moderate" in sevs else "mild" if sevs else "normal"

    return {
        "model": ADAPTER,
        "image_dimensions": {"width": original_width, "height": original_height},
        "overall_severity": overall,
        "findings": actionable,
        "informational": informational,
        "total_findings": len(actionable),
        "note": "Non-diagnostic. Educational purposes only.",
        "_raw": raw_output,
    }


# =============================================================
# PARSER
# =============================================================
def parse_bbox_output(raw: str, proc_w: int, proc_h: int, orig_w: int, orig_h: int) -> List[Dict]:
    findings = []

    # Strategy 1: JSON
    json_findings = try_parse_json_bboxes(raw)
    if json_findings:
        findings = json_findings

    # Strategy 2: Individual bbox JSON objects
    if not findings:
        for match in re.finditer(r'\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}', raw):
            try:
                obj = json.loads(match.group())
                if "bbox_2d" in obj:
                    findings.append(obj)
            except json.JSONDecodeError:
                continue

    # Strategy 3: <box>x1,y1,x2,y2</box>
    if not findings:
        for match in re.finditer(r'<box>(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)</box>', raw):
            x1, y1, x2, y2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            findings.append({"bbox_2d": [x1, y1, x2, y2], "label": "finding", "normalized": True})

    # Strategy 4: <ref>label</ref><box>(x1,y1),(x2,y2)</box>
    if not findings:
        for match in re.finditer(r'<ref>([^<]+)</ref>\s*<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', raw):
            label = match.group(1).strip()
            x1, y1, x2, y2 = int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))
            findings.append({"bbox_2d": [x1, y1, x2, y2], "label": label, "normalized": True})

    # Strategy 5: Raw [x1, y1, x2, y2] with nearby label
    if not findings:
        for match in re.finditer(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', raw):
            x1, y1, x2, y2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            context_start = max(0, match.start() - 100)
            context_end = min(len(raw), match.end() + 100)
            context = raw[context_start:context_end].lower()
            label = detect_label(context)
            findings.append({"bbox_2d": [x1, y1, x2, y2], "label": label or "finding"})

    # Strategy 6: Text fallback
    if not findings:
        findings = parse_text_fallback(raw)

    # Build final results
    result = []
    for f in findings:
        bbox = f.get("bbox_2d", f.get("bbox", []))
        label = f.get("label", "abnormality")
        is_normalized = f.get("normalized", False)
        condition = normalize_label(label)

        if condition == "healthy":
            continue

        # Handle normalized 0-1000 coords
        if bbox and len(bbox) == 4:
            coords_in_thousand = all(0 <= c <= 1000 for c in bbox)
            bbox_exceeds_image = bbox[2] > proc_w or bbox[3] > proc_h
            image_larger_than_thousand = proc_w > 1000 or proc_h > 1000
            if is_normalized or (coords_in_thousand and (bbox_exceeds_image or image_larger_than_thousand)):
                bbox = [
                    int(bbox[0] * proc_w / 1000),
                    int(bbox[1] * proc_h / 1000),
                    int(bbox[2] * proc_w / 1000),
                    int(bbox[3] * proc_h / 1000),
                ]

        confidence = compute_confidence(bbox, proc_w, proc_h)

        fdi = f.get("tooth_number_override", 0)
        if not fdi and bbox and len(bbox) == 4:
            fdi = bbox_to_fdi(bbox, proc_w, proc_h)

        # Scale to original dimensions
        if bbox and len(bbox) == 4:
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h
            bbox = [round(bbox[0]*scale_x), round(bbox[1]*scale_y), round(bbox[2]*scale_x), round(bbox[3]*scale_y)]

        # Compute severity
        condition, severity = label_to_severity(condition, bbox, orig_w, orig_h)

        # Expand tight bboxes
        if bbox and len(bbox) == 4:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            area_ratio = (bw * bh) / (orig_w * orig_h) if (orig_w * orig_h) > 0 else 0

            if area_ratio < 0.01:
                pad_pct = 0.4
            elif area_ratio < 0.04:
                pad_pct = 0.2
            else:
                pad_pct = 0.1

            pad_x = max(int(bw * pad_pct), 10)
            pad_y = max(int(bh * pad_pct), 10)
            bbox = [
                max(0, bbox[0] - pad_x),
                max(0, bbox[1] - pad_y),
                min(orig_w, bbox[2] + pad_x),
                min(orig_h, bbox[3] + pad_y),
            ]

        entry = {"tooth_number": fdi, "condition": condition, "severity": severity, "confidence": round(confidence, 2)}
        if bbox:
            entry["bbox"] = bbox
        result.append(entry)

    result = [f for f in result if f["confidence"] >= 0.35]
    result = filter_hallucinations(result)
    result = filter_tiling_hallucination(result)
    result = deduplicate_same_bbox(result)

    seen = set()
    unique = []
    for f in result:
        key = (f["tooth_number"], f["condition"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    unique.sort(key=lambda x: (-x["confidence"], x.get("tooth_number", 0)))
    return unique


# =============================================================
# HELPERS
# =============================================================
def try_parse_json_bboxes(raw: str) -> list:
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and ("bbox_2d" in d or "bbox" in d)]
        if isinstance(data, dict) and ("bbox_2d" in data or "bbox" in data):
            return [data]
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            text = re.sub(r',\s*([}\]])', r'\1', match.group())
            data = json.loads(text)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and ("bbox_2d" in d or "bbox" in d)]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def parse_text_fallback(raw: str) -> list:
    findings = []
    parts = re.split(r'(?=\*\*FDI\s*#?\s*\d{2}\s*\()', raw)
    for part in parts:
        header = re.match(r'\*\*FDI\s*#?\s*(\d{2})\s*\(([^)]+)\)\s*:?\s*\*\*:?', part)
        if not header:
            continue
        model_num = int(header.group(1))
        description = header.group(2).strip()
        section_text = part[header.end():]
        fdi = compute_fdi(description)
        if not fdi:
            fdi = model_num if is_valid_fdi(model_num) else 0
        if "no visible" in section_text.lower() or "appears healthy" in section_text.lower():
            continue
        if "severity: none" in section_text.lower():
            continue
        cond_match = re.search(r'\*?\*?Condition\*?\*?\s*:\s*(.+?)(?:\n|$)', section_text, re.IGNORECASE)
        condition_text = cond_match.group(1).strip() if cond_match else section_text
        condition = normalize_label(condition_text)
        if condition == "healthy":
            continue
        findings.append({"bbox_2d": [], "label": condition, "tooth_number_override": fdi})
    return findings


def compute_confidence(bbox: list, img_w: int, img_h: int) -> float:
    if not bbox or len(bbox) < 4:
        return 0.5
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return 0.0
    area_ratio = (box_w * box_h) / (img_w * img_h) if (img_w * img_h) > 0 else 0
    if area_ratio < 0.001: return 0.2
    if area_ratio > 0.20: return 0.25
    if 0.003 <= area_ratio <= 0.06: return 0.85
    if 0.001 <= area_ratio < 0.003: return 0.55
    if 0.06 < area_ratio <= 0.20: return 0.45
    return 0.7


def filter_hallucinations(findings: list) -> list:
    max_per_condition = 4
    result = []
    condition_added = Counter()
    sorted_findings = sorted(findings, key=lambda x: -x["confidence"])
    for f in sorted_findings:
        cond = f["condition"]
        if condition_added[cond] < max_per_condition:
            result.append(f)
            condition_added[cond] += 1
    return result


def filter_tiling_hallucination(findings: list) -> list:
    groups = defaultdict(list)
    for f in findings:
        groups[f["condition"]].append(f)

    keep = []
    for condition, items in groups.items():
        if len(items) < 4:
            keep.extend(items)
            continue

        bboxes_with_y = []
        for item in items:
            bbox = item.get("bbox", [])
            if len(bbox) == 4:
                bboxes_with_y.append((bbox[1], bbox[3], item))

        if not bboxes_with_y:
            keep.extend(items)
            continue

        y_tops = [b[0] for b in bboxes_with_y]
        y_spread = max(y_tops) - min(y_tops) if len(y_tops) > 1 else 0
        if y_spread < 50 and len(items) >= 4:
            continue
        else:
            keep.extend(items)

    return keep


def deduplicate_same_bbox(findings: list) -> list:
    severity_rank = {"severe": 3, "moderate": 2, "mild": 1, "unknown": 0}
    sorted_f = sorted(findings, key=lambda x: -severity_rank.get(x.get("severity", "unknown"), 0))
    keep = []
    for candidate in sorted_f:
        c_bbox = candidate.get("bbox", [])
        is_dup = False
        if c_bbox and len(c_bbox) == 4:
            for kept in keep:
                k_bbox = kept.get("bbox", [])
                if k_bbox and len(k_bbox) == 4:
                    iou = compute_iou(c_bbox, k_bbox)
                    if iou > 0.85:
                        is_dup = True
                        break
        if not is_dup:
            keep.append(candidate)
    return keep


def compute_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def bbox_to_fdi(bbox: list, img_w: int, img_h: int) -> int:
    if not bbox or len(bbox) < 4:
        return 0
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    nx, ny = cx / img_w, cy / img_h
    is_upper = ny < 0.5
    is_left_side = nx < 0.5

    if is_upper and is_left_side: quadrant = 1
    elif is_upper and not is_left_side: quadrant = 2
    elif not is_upper and not is_left_side: quadrant = 3
    else: quadrant = 4

    dist = abs(nx - 0.5) * 2
    aspect_ratio = img_w / img_h if img_h > 0 else 1.0

    if aspect_ratio >= 2.0:
        if dist < 0.13: pos = 1
        elif dist < 0.25: pos = 2
        elif dist < 0.38: pos = 3
        elif dist < 0.50: pos = 4
        elif dist < 0.63: pos = 5
        elif dist < 0.75: pos = 6
        elif dist < 0.88: pos = 7
        else: pos = 8
    else:
        if dist < 0.15: pos = 1
        elif dist < 0.30: pos = 2
        elif dist < 0.58: pos = 3
        elif dist < 0.74: pos = 4
        elif dist < 0.87: pos = 5
        else: pos = 6

    return quadrant * 10 + pos


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


def normalize_label(label: str) -> str:
    t = label.lower().strip()
    if any(w in t for w in ["deep caries", "deep cavity", "severe caries", "pulp"]): return "deep caries"
    if any(w in t for w in ["caries", "carious", "cavity", "decay"]): return "caries"
    if any(w in t for w in ["periapical", "apical", "lesion", "abscess", "radiolucen"]): return "periapical lesion"
    if any(w in t for w in ["impacted", "unerupted", "embedded"]): return "impacted"
    if any(w in t for w in ["healthy", "normal", "no visible", "none", "intact"]): return "healthy"
    return t[:50] if t else "abnormality"


def label_to_severity(condition: str, bbox: list = None, img_w: int = 0, img_h: int = 0) -> tuple:
    base_severity = {"deep caries": "severe", "periapical lesion": "severe", "impacted": "moderate", "caries": "moderate"}.get(condition, "moderate")

    if bbox and len(bbox) == 4 and img_w > 0 and img_h > 0:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        area_ratio = (bw * bh) / (img_w * img_h)

        aspect = img_w / img_h
        threshold = 0.003 if aspect >= 2.0 else 0.01

        if area_ratio < threshold:
            if condition == "deep caries":
                return ("caries", "moderate")
            elif base_severity == "moderate":
                return (condition, "mild")

    return (condition, base_severity)


def detect_label(text: str) -> str:
    t = text.lower()
    if "deep caries" in t or "deep cavity" in t: return "deep caries"
    if "periapical" in t or "apical lesion" in t or "radiolucen" in t: return "periapical lesion"
    if "impacted" in t or "unerupted" in t: return "impacted"
    if "caries" in t or "cavity" in t or "decay" in t: return "caries"
    return ""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
