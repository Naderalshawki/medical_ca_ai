from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import os
import shutil

# ======================================================
# إعداد المسارات (تصحيح مهم جداً)
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best3.pt")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UI_DIR = os.path.join(BASE_DIR, "ui")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ======================================================
# تحميل الموديل مرة واحدة فقط
# ======================================================

print("🔄 Loading YOLO Segmentation Model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("✅ Model Loaded Successfully!")

# ======================================================
# إنشاء التطبيق
# ======================================================

app = FastAPI(title="Medical AI Platform")

# جعل مجلد outputs قابل للعرض في المتصفح
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# ======================================================
# الصفحة الرئيسية
# ======================================================

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(UI_DIR, "index.html")
    if not os.path.exists(index_path):
        return "<h2>index.html not found inside ui folder</h2>"

    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# ======================================================
# Segmentation Endpoint
# ======================================================

@app.post("/segment-image")
async def segment_image(file: UploadFile = File(...)):

    input_path = os.path.join(OUTPUTS_DIR, file.filename)

    # حفظ الصورة المرفوعة
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # تنفيذ التنبؤ بدون save=True (نرسم يدويًا)
    results = model.predict(
        source=input_path,
        conf=0.25,
        save=False
    )

    result = results[0]
    plotted = result.plot()

    output_name = "result_" + file.filename
    output_path = os.path.join(OUTPUTS_DIR, output_name)

    Image.fromarray(plotted).save(output_path)

    return {
        "message": "تم تحليل الصورة بنجاح",
        "result_image": output_name
    }