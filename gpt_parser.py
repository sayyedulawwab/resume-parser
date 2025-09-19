import os
import platform
import time
import subprocess
import json
import re
import pyautogui
from datetime import datetime
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    NoSuchElementException,
    ElementClickInterceptedException,
)
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p: str | None) -> str | None:
    """Return an absolute, normalized path. If p is relative, resolve from PROJECT_ROOT."""
    if not p:
        return p
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p))

# === ENV CONFIG ===
RESUMES_DIR = os.getenv("RESUMES_DIR", "resumes")
RESULT_JSON = os.getenv("RESPONSE_JSON_PATH", "ai_response.json")
INSTRUCTION_FILE = resolve_path(os.getenv("PROMPT_PATH", "prompt.txt"))
TARGET_URL = os.getenv("TARGET_URL", "https://chatgpt.com/")
REMOTE_DEBUG_PORT = int(os.getenv("REMOTE_DEBUG_PORT", 9223))
CHROME_CMD = os.getenv("CHROME_CMD")

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

RESUME_FILE = resolve_path(get_first_resume(RESUMES_DIR))
print(f"📂 Selected resume: {RESUME_FILE}")
print(f"📄 Instruction file: {INSTRUCTION_FILE}")

# ==== Helper: ensure logs dir ====
LOG_DIR = os.path.join(os.getcwd(), "logs", "reviews")
os.makedirs(LOG_DIR, exist_ok=True)

def get_next_result_filename(base_filename: str) -> tuple[str, str, int, str]:
    """Generate '<n>_<stem>_<timestamp>.<ext>' auto-increment result file."""
    base = os.path.basename(base_filename).strip() or "result.json"
    stem, ext = os.path.splitext(base)
    stem = stem or "result"
    ext = ext.lstrip(".") or "json"

    pattern = re.compile(rf"^(\d+)_({re.escape(stem)})_\d{{8}}_\d{{6}}\.{ext}$", re.IGNORECASE)

    max_index = 0
    for folder in [os.getcwd(), LOG_DIR]:
        for f in os.listdir(folder):
            m = pattern.match(f)
            if m:
                max_index = max(max_index, int(m.group(1)))

    next_index = max_index + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    next_name = f"{next_index}_{stem}_{timestamp}.{ext}"
    return stem, next_name, next_index, ext

# ==== STEP 1: Launch Chrome if specified ====
chrome_proc = None
if CHROME_CMD:
    args = [
        CHROME_CMD,
        f"--remote-debugging-port={REMOTE_DEBUG_PORT}",
        # "--profile-directory=\"Profile 1\""
        "--user-data-dir=C:\ChromeDebug",
    ]
    print(f"🚀 Launching Chrome: {args}")
    chrome_proc = subprocess.Popen(args)
    time.sleep(5)

# ==== STEP 2: Connect to existing Chrome ====
options = webdriver.ChromeOptions()
options.debugger_address = f"127.0.0.1:{REMOTE_DEBUG_PORT}"
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

print(f"✅ Connecting to {TARGET_URL}")
driver.get(TARGET_URL)
print(f"✅ Connected to {TARGET_URL}")

# ==== STEP 3: Upload Resume File ====
def safe_click(xpath, retries=3, delay=1):
    last_exception = None
    for attempt in range(retries):
        try:
            element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            element.click()
            return element
        except Exception as e:
            last_exception = e
            time.sleep(delay)
    raise last_exception

# Open composer
safe_click("//button[@data-testid='composer-plus-btn']")
safe_click("//div[contains(text(),'Add photos & files')]")

file_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='file']")))
driver.execute_script("arguments[0].style.display='block';", file_input)

all_files_str = "\n".join([resolve_path(f) for f in RESUME_FILES])
file_input.send_keys(all_files_str)

print(f"📤 Uploaded resumes: {all_files_str}")
time.sleep(1)
pyautogui.press("esc")

# ==== STEP 4: Enter Instructions ====
textarea = wait.until(EC.presence_of_element_located((By.XPATH, "//div[@id='prompt-textarea']")))

with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
    instructions = f.read()

try:
    textarea.clear()
except Exception:
    pass

for line in instructions.splitlines():
    textarea.send_keys(line)
    textarea.send_keys(Keys.SHIFT, Keys.ENTER)

print("📝 Instructions entered.")

# ==== STEP 5: Submit Prompt ====
submit_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@id='composer-submit-button']")))
submit_btn.click()
print("🚀 Submitted request, waiting for response...")

# ==== STEP 6: Capture GPT Response ====
result_text = ""
MAX_WAIT = 500
POLL_INTERVAL = 5
STREAMING_BTN_XPATH = ("//button[@id='composer-submit-button' and @aria-label='Stop streaming']")

elapsed = 0
while elapsed < MAX_WAIT:
    try:
        result_box = driver.find_element(By.XPATH, "(//div[@data-message-author-role='assistant'])[last()]")
        text = result_box.get_attribute("innerText").strip()
        if text and "…" not in text:
            result_text = text

        try:
            driver.find_element(By.XPATH, STREAMING_BTN_XPATH)
        except NoSuchElementException:
            break
    except (StaleElementReferenceException, NoSuchElementException):
        pass

    time.sleep(POLL_INTERVAL)
    elapsed += POLL_INTERVAL

json_match = re.search(r"(\[.*\])", result_text, re.DOTALL)

if json_match:
    json_array_text = json_match.group(1)
    try:
        data = json.loads(json_array_text)  # Validate it's proper JSON
        # Overwrite result_text with clean JSON string
        result_text = json.dumps(data, indent=2)
    except json.JSONDecodeError:
        print("⚠️ JSON parsing failed, saving raw result.")
else:
    print("⚠️ No JSON array found, saving raw result.")

# ==== STEP 7: Save Results ====
with open(RESULT_JSON, "w", encoding="utf-8") as f:
    f.write(result_text)

stem, next_file, idx, ext = get_next_result_filename(RESULT_JSON)
review_log_path = os.path.join(LOG_DIR, next_file)
with open(review_log_path, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"✅ Response saved to {RESULT_JSON}")
print(f"📝 Log copy saved to {review_log_path}")

# ==== STEP 8: Close ====
driver.quit()
if chrome_proc:
    chrome_proc.terminate()
    try:
        chrome_proc.wait(timeout=5)
    except Exception:
        pass
    if chrome_proc.poll() is None:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", str(chrome_proc.pid), "/T", "/F"], check=False)
        else:
            chrome_proc.kill()
print("🎉 Done.")
