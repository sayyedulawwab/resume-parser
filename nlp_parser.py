import re
import os
import json
import spacy
import pdfplumber
import pytesseract
from sentence_transformers import SentenceTransformer, util
from itertools import chain
import dateparser

# ----------------------------
# Setup
# ----------------------------
nlp = spacy.load("en_core_web_lg")
embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load skills DB
SKILLS_FILE = "./data/skills.json"
if os.path.exists(SKILLS_FILE):
    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        SKILLS_DB = json.load(f)
else:
    SKILLS_DB = []
    print(f"Warning: {SKILLS_FILE} not found. SKILLS_DB is empty.")

SKILL_EMBEDDINGS = embedder.encode(SKILLS_DB, convert_to_tensor=True)

# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # fallback to OCR
                page_image = page.to_image(resolution=300).original
                text += pytesseract.image_to_string(page_image) + "\n"
    return text.strip()

def extract_text_from_image(path):
    return pytesseract.image_to_string(path)

def clean_text(text):
    return re.sub(r"[ \t]+", " ", text).strip()

# ----------------------------
# Section splitter
# ----------------------------
def split_sections(text):
    headings = re.findall(r"^(?:[A-Z][A-Z\s/&]+)$", text, flags=re.MULTILINE)
    sections = {}
    lines = text.split("\n")
    current_heading = None
    buffer = []
    
    for line in lines + ["END_OF_TEXT"]:
        norm = line.strip().upper()
        if norm in headings or line == "END_OF_TEXT":
            if current_heading:
                sections[current_heading] = "\n".join(buffer).strip()
                buffer = []
            current_heading = norm if line != "END_OF_TEXT" else None
        else:
            buffer.append(line)
    return sections

# ----------------------------
# Extractors
# ----------------------------
def extract_name(text, lines_to_check=20):
    lines = text.split("\n")[:lines_to_check]
    for line in lines:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
    return lines[0] if lines else None

def extract_contact_info(text):
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phones = re.findall(r"\+?\d[\d\-\s]{9,}\d", text)
    phones = [re.sub(r"\s+", "", p) for p in phones]
    return {"email": emails, "phone": phones}

def extract_links(text):
    links = re.findall(r"(https?://[^\s|]+|www\.[^\s|]+)", text)
    linkedin = [l for l in links if "linkedin.com" in l.lower()]
    github = [l for l in links if "github.com" in l.lower()]
    others = list(set(links) - set(linkedin) - set(github))
    return {"linkedin": linkedin, "github": github, "other": others}

def extract_skills(text):
    if not SKILLS_DB or SKILL_EMBEDDINGS is None or SKILL_EMBEDDINGS.numel() == 0:
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

    return sorted(set([s.strip().title() for s in matched_skills]))

# ----------------------------
# Experience & Education
# ----------------------------
def parse_experience_block(block):
    dates = re.findall(r"([A-Za-z]{3}\s\d{4}|\d{4}|Present)\s*[-–]\s*([A-Za-z]{3}\s\d{4}|\d{4}|Present)", block)
    start_date, end_date = (None, None)
    if dates:
        start_date = str(dateparser.parse(dates[0][0]).date().year) if dates[0][0] != "Present" else "Present"
        end_date = str(dateparser.parse(dates[0][1]).date().year) if dates[0][1] != "Present" else "Present"
    lines = block.split("\n")
    role_line = lines[0] if lines else ""
    role, company = None, None
    if "," in role_line:
        parts = role_line.split(",")
        role = parts[0].strip()
        company = parts[1].strip() if len(parts) > 1 else None
    description = [l.strip("-*• ").strip() for l in lines[1:] if l.strip()]
    return {"start_date": start_date, "end_date": end_date, "role": role, "company": company, "description": description}

def extract_experience(text):
    sections = split_sections(text)
    exp_text = sections.get("EXPERIENCE", "")
    blocks = re.split(r"\n\s*\n", exp_text)
    experiences = [parse_experience_block(b) for b in blocks if b.strip()]
    return experiences

def extract_education(text):
    sections = split_sections(text)
    edu_text = sections.get("EDUCATION", "")
    education = []
    for line in edu_text.split("\n"):
        degree_match = re.findall(r"(B\.?Sc|M\.?Sc|B\.?Eng|M\.?Eng|Bachelor|Master|PhD|MBA)[^,\n]*", line, re.IGNORECASE)
        if degree_match:
            education.append({"degree": degree_match[0].strip(), "institution": line.strip()})
    return education

# ----------------------------
# Main Parser
# ----------------------------
def parse_resume(path, is_image=False):
    text = extract_text_from_image(path) if is_image else extract_text_from_pdf(path)
    text = "\n".join([l.strip() for l in text.split("\n") if l.strip() != ""])
    return {
        "name": extract_name(text),
        "contacts": extract_contact_info(text),
        "links": extract_links(text),
        "skills": extract_skills(text),
        "experience": extract_experience(text),
        "education": extract_education(text),
        "raw_text": text
    }

def parse_resumes_in_folder(folder_path):
    parsed_data = {}
    for file_name in os.listdir(folder_path):
        ext = file_name.lower().split(".")[-1]
        if ext in ("pdf", "png", "jpg", "jpeg"):
            path = os.path.join(folder_path, file_name)
            parsed_data[file_name] = parse_resume(path, is_image=(ext != "pdf"))
    return parsed_data

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    folder = "./resumes"
    all_resumes = parse_resumes_in_folder(folder)
    with open("parsed_resumes.json", "w", encoding="utf-8") as f:
        json.dump(all_resumes, f, indent=4, ensure_ascii=False)
    print(f"Parsed {len(all_resumes)} resumes. Saved to parsed_resumes.json")
