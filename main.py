
from fastapi import FastAPI, UploadFile, File
import uvicorn
import re
import shutil
import os
import tempfile
import zipfile
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import whisper

app = FastAPI()

# -------- إعداد النموذج: تحميل من Google Drive إذا لم يكن موجود --------
MODEL_DIR = "vishing_model/FINAL_vishing_detection_model_2025-03-19_22-18-11"
GDRIVE_ZIP_URL = "https://drive.google.com/uc?id=1HDDSV6abkXuLXZSjHvWsh4hsaVtnCGLz"

if not os.path.exists(MODEL_DIR):
    print("🔄 تحميل النموذج من Google Drive...")
    gdown.download(GDRIVE_ZIP_URL, output="model.zip", quiet=False)
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("✅ تم فك ضغط النموذج!")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
whisper_model = whisper.load_model("base")

# -------- قائمة التصفية للنصوص --------
STOPWORDS = set(["في", "على", "من", "إلى", "عن", "أن", "مع", "كان", "التي", "هذا", "ذلك", "ما", "لم", 
                 "لن", "قد", "ثم", "إذ", "بعد", "قبل", "حتى", "كل", "أي", "به", "إلي", "بين", "مثل", "عند", 
                 "و", "ب", "ف", "كما", "إن", "إذا", "لكن", "أو", "بل", "أيضا", "حيث", "كما", "لأن", "إلا", 
                 "بسبب", "عبر", "داخل", "خلال", "دون", "حول", "حسب", "بما", "إضافة", "عليه", "لذلك", 
                 "ذلك", "بهذا", "منذ"])

KEYWORDS_TO_REMOVE = set(["السلام عليكم", "وعليكم السلام", "صباح الخير", "مساء الخير", "أهلًا وسهلًا",
                          "مرحبًا", "كيف حالك", "كيف حالكم", "تحياتي", "يوم سعيد", "تحية طيبة وبعد",
                          "مع خالص الشكر والتقدير", "أرجو أن تكون بخير", "نأمل تفاعلكم الكريم",
                          "في انتظار ردكم الكريم", "شاكرين لكم حسن تعاونكم", "دمتم بخير", "كيف الأمور",
                          "كل عام وأنتم بخير", "بالتوفيق إن شاء الله", "ما شاء الله", "جزاك الله خيرًا",
                          "بارك الله فيك", "الله يعطيك العافية", "الله يسعدك", "عيدكم مبارك"])

# -------- دالة تنظيف النصوص --------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^؀-ۿ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[0-9٠-٩]", "", text)
    text = re.sub(r"[\u064B-\u065F]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ء", "", text)

    for phrase in KEYWORDS_TO_REMOVE:
        text = text.replace(phrase, "")
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# -------- تحليل الصوت --------
def transcribe_audio(file_path: str) -> str:
    result = whisper_model.transcribe(file_path, language="ar")
    return result["text"]

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return "احتيالية" if predicted_class == 1 else "ليست احتيالية"

# -------- Endpoint --------
@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    raw_text = transcribe_audio(temp_file_path)
    cleaned_text = clean_text(raw_text)
    label = predict(cleaned_text)

    os.remove(temp_file_path)
    return {"text": raw_text, "cleaned": cleaned_text, "label": label}
