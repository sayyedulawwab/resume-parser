import re
import os
import json
import spacy
import pdfplumber
import pytesseract
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from itertools import chain
import difflib

# Load spaCy model and sentence transformer
nlp = spacy.load("en_core_web_lg")
embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Expanded skills DB (can be external JSON file)
SKILLS_FILE = "./data/skills.json"
if os.path.exists(SKILLS_FILE):
    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        SKILLS_DB = json.load(f)
else:
    SKILLS_DB = []
    print(f"Warning: {SKILLS_FILE} not found. SKILLS_DB is empty.")

    
SKILL_EMBEDDINGS = embedder.encode(SKILLS_DB, convert_to_tensor=True)

# ------------------------------------------------
# Helpers
# ------------------------------------------------

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                # fallback to OCR
                page_image = page.to_image(resolution=300).original
                text += pytesseract.image_to_string(page_image)
    return text.strip()

def extract_text_from_image(path):
    return pytesseract.image_to_string(path)

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

# ------------------------------------------------
# Core Extractors
# ------------------------------------------------

def extract_name(text, lines_to_check=15):
    lines = text.split("\n")[:lines_to_check]
    for line in lines:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
    return None

def extract_contact_info(text):
    email = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = re.findall(r"\+?\d[\d\-\s]{9,}\d", text)
    return {"email": email, "phone": phone}

def extract_links(text):
    links = re.findall(r"(https?://[^\s]+)", text)
    linkedin = [l for l in links if "linkedin.com" in l]
    github = [l for l in links if "github.com" in l]
    return {"linkedin": linkedin, "github": github, "other": list(set(links) - set(linkedin) - set(github))}

def extract_skills(text):
    if not SKILLS_DB:  # <-- check the list, not the tensor
        return []

    if SKILL_EMBEDDINGS is None or SKILL_EMBEDDINGS.numel() == 0:
        return []

    doc = nlp(text)
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
    ngrams = list(chain.from_iterable(
        [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        for n in range(1, 4)
    ))

    token_embeddings = embedder.encode(ngrams, convert_to_tensor=True, batch_size=32)
    cosine_scores = util.pytorch_cos_sim(token_embeddings, SKILL_EMBEDDINGS)

    matched_skills = set()
    for i, token in enumerate(ngrams):
        for j, score in enumerate(cosine_scores[i]):
            if score > 0.7:
                matched_skills.add(SKILLS_DB[j])
    return list(matched_skills)

def extract_section(text, keywords):
    """Improved section extractor: regex + fuzzy matching"""
    lines = text.split("\n")
    start, section_lines = None, []
    for i, line in enumerate(lines):
        norm = line.strip().lower()
        if any(difflib.get_close_matches(k, [norm], cutoff=0.8) for k in keywords):
            start = i
            continue
        if start is not None:
            if line.strip() == "" or re.match(r"^[A-Z][A-Z\s]+$", line):
                break
            section_lines.append(line)
    return " ".join(section_lines).strip()

def extract_experience(text):
    exp_text = extract_section(text, ["experience", "employment", "work history"])
    exp_blocks = re.split(r"\n\s*\n", exp_text)
    experiences = []
    for block in exp_blocks:
        dates = re.findall(r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{4}\b|\d{2}/\d{4}|\d{4})", block)
        experiences.append({"text": block.strip(), "dates": dates})
    return experiences

def extract_education(text):
    edu_text = extract_section(text, ["education", "academics", "qualification"])
    degrees = re.findall(r"(Bachelor|Master|PhD|B\.Sc|M\.Sc|B\.Eng|MBA)[^,\n]*", edu_text, re.IGNORECASE)
    return {"raw": edu_text, "degrees": degrees}

# ------------------------------------------------
# Main Parser
# ------------------------------------------------

def parse_resume(path, is_image=False):
    text = extract_text_from_image(path) if is_image else extract_text_from_pdf(path)
    text = clean_text(text)

    return {
        "name": extract_name(text),
        "contacts": extract_contact_info(text),
        "links": extract_links(text),
        "skills": extract_skills(text),
        "experience": extract_experience(text),
        "education": extract_education(text),
        "raw_text": text  # optional, for debugging
    }

def parse_resumes_in_folder(folder_path):
    parsed_data = {}
    for file_name in os.listdir(folder_path):
        ext = file_name.lower().split(".")[-1]
        if ext in ("pdf", "png", "jpg", "jpeg"):
            path = os.path.join(folder_path, file_name)
            parsed_data[file_name] = parse_resume(path, is_image=(ext != "pdf"))
    return parsed_data

# Example usage
if __name__ == "__main__":
    folder = "./resumes"  # replace with your folder path
    all_resumes = parse_resumes_in_folder(folder)
    with open("parsed_resumes.json", "w", encoding="utf-8") as f:
        json.dump(all_resumes, f, indent=4, ensure_ascii=False)
    print(f"Parsed {len(all_resumes)} resumes. Saved to parsed_resumes.json")
