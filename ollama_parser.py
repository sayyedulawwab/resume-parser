import ollama
import pytesseract
import pdfplumber
import re
import os
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p: str | None) -> str | None:
    """Return an absolute, normalized path. If p is relative, resolve from PROJECT_ROOT."""
    if not p:
        return p
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p))

# === ENV CONFIG ===
RESUMES_DIR = os.getenv("RESUMES_DIR", "resumes")
RESULT_JSON = os.getenv("OLLAMA_RESPONSE_JSON_PATH", "ollama_response.json")
INSTRUCTION_FILE = resolve_path(os.getenv("PROMPT_PATH", "prompt.txt"))
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-v3.1:671b-cloud")

# ==== Pick first resume file ====
def get_first_resume(resume_dir: str) -> str:
    files = sorted(os.listdir(resume_dir))
    for f in files:
        if f.lower().endswith((".pdf", ".docx")):
            return os.path.join(resume_dir, f)
    raise FileNotFoundError(f"No .pdf or .docx resumes found in {resume_dir}")


def get_all_resumes(resume_dir: str) -> list[str]:
    files = sorted(os.listdir(resume_dir))
    resumes = [os.path.join(resume_dir, f) for f in files if f.lower().endswith((".pdf", ".docx"))]
    if not resumes:
        raise FileNotFoundError(f"No .pdf or .docx resumes found in {resume_dir}")
    return resumes

RESUME_FILES = get_all_resumes(RESUMES_DIR)
print(f"📂 Found {len(RESUME_FILES)} resumes.")
print(f"📄 Instruction file: {INSTRUCTION_FILE}")

# ==== Helper: ensure logs dir ====
LOG_DIR = os.path.join(os.getcwd(), "logs", "reviews")
os.makedirs(LOG_DIR, exist_ok=True)

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


def parse_resume(path, is_image=False):
    text = extract_text_from_image(path) if is_image else extract_text_from_pdf(path)
    text = "\n".join([l.strip() for l in text.split("\n") if l.strip() != ""])
    return text

def parse_resumes_in_folder(folder_path):
    parsed_data = {}
    for file_name in os.listdir(folder_path):
        ext = file_name.lower().split(".")[-1]
        if ext in ("pdf", "png", "jpg", "jpeg"):
            path = os.path.join(folder_path, file_name)
            parsed_data[file_name] = parse_resume(path, is_image=(ext != "pdf"))
    return parsed_data

with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
    instructions = f.read()


all_resumes = parse_resumes_in_folder(RESUMES_DIR)

all_resumes_json = json.dumps(all_resumes, indent=4)


prompt = f"""--INSTRUCTION--
{instructions}
--RESUMES--
{all_resumes_json}"""


client = ollama.Client()

messages = [
  {
    'role': 'user',
    'content': prompt,
  },
]

response_text = ""

ollama.pull(MODEL_NAME)

print("Start processing ...")
# Collect streamed output
for part in client.chat(MODEL_NAME, messages=messages, stream=True):
    content = part['message']['content']
    response_text += content

print("✅ Finished collecting response.")

# Try to capture any json block inside triple backticks
match = re.search(r"```json\s*([\s\S]*?)```", response_text, re.MULTILINE)

if not match:
    raise ValueError("No JSON block found in response")

json_str = match.group(1).strip()

# Parse JSON safely
try:
    parsed_json = json.loads(json_str)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in response: {e}")

with open(RESULT_JSON, "w", encoding="utf-8") as f:
    json.dump(parsed_json, f, indent=4, ensure_ascii=False)

print(f"💾 Extracted JSON saved to {RESULT_JSON}")