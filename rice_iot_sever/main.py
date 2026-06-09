import os
import uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="Rice Disease Classification Server")

UPLOAD_DIR = "uploads"
MODEL_PATH = "model/best_rice_disease.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory="giaodien")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO(MODEL_PATH)

message_map = {
    "bacterial_leaf_blight": "Bệnh bạc lá",
    "brown_spot": "ệnh đốm nâu",
    "healthy": "Lá khỏe mạnh",
    "leaf_blast": "Bệnh đạo ôn lá",
    "leaf_scald": "Bệnh cháy bờ lá",
    "sheath_blight": "Bệnh kho van"
}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "File khong phai anh"}
            )

        ext = os.path.splitext(file.filename)[1]
        if ext == "":
            ext = ".jpg"

        filename = f"{uuid.uuid4()}{ext}"
        image_path = os.path.join(UPLOAD_DIR, filename)

        contents = await file.read()
        with open(image_path, "wb") as f:
            f.write(contents)

        results = model(image_path)

        result = results[0]
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = result.names[class_id]

        message = message_map.get(class_name, class_name)

        return {
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "message": message
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )