import os
import uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="Rice Disease Classification Server")

# ==================== CẤU HÌNH HỆ THỐNG ====================
UPLOAD_DIR = "uploads"
MODEL_PATH = "model/best_rice_disease.pt"
TEMPLATE_DIR = "giaodien"
STATIC_DIR = "static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ==================== LOAD MODEL YOLO ====================
model = YOLO(MODEL_PATH)

# ==================== KẾT QUẢ MỚI NHẤT CHO ESP32 ====================
latest_result = {
    "class_name": "",
    "message": "Chua co ket qua",
    "web_message": "Chưa có kết quả",
    "confidence": 0,
    "confidence_percent": 0
}

# Chế độ điều khiển LED:
# auto = tự động theo AI
# on   = bật cảnh báo đỏ thủ công
# off  = tắt tất cả LED
led_control = {
    "mode": "auto"
}

# ==================== ÁNH XẠ TÊN BỆNH ====================
# Hiển thị trên web/app: tiếng Việt có dấu
message_map = {
    "bacterial_leaf_blight": "Bệnh bạc lá",
    "brown_spot": "Bệnh đốm nâu",
    "healthy": "Lá khỏe mạnh",
    "leaf_blast": "Bệnh đạo ôn lá",
    "leaf_scald": "Bệnh cháy bờ lá",
    "sheath_blight": "Bệnh khô vằn"
}

# Hiển thị trên OLED: không dấu để tránh lỗi font
oled_message_map = {
    "bacterial_leaf_blight": "Bac La",
    "brown_spot": "Dom Nau",
    "healthy": "Khoe Manh",
    "leaf_blast": "Dao On",
    "leaf_scald": "Chay Bo La",
    "sheath_blight": "Kho Van"
}


# ==================== WEB GIAO DIỆN CŨ ====================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ==================== API ESP32 ĐỌC KẾT QUẢ ====================
@app.get("/latest")
def get_latest_result():
    data = latest_result.copy()
    data["led_mode"] = led_control["mode"]
    return data


# ==================== API ĐIỀU KHIỂN LED ====================
@app.post("/led/{mode}")
def set_led_mode(mode: str):
    if mode not in ["auto", "on", "off"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Mode phai la auto, on hoac off"}
        )

    led_control["mode"] = mode

    return {
        "status": "ok",
        "led_mode": mode
    }


# ==================== API DỰ ĐOÁN AI ====================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global latest_result

    try:
        if file.content_type is None or not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "File khong phai anh"}
            )

        ext = os.path.splitext(file.filename or "")[1]
        if ext == "":
            ext = ".jpg"

        filename = f"{uuid.uuid4()}{ext}"
        image_path = os.path.join(UPLOAD_DIR, filename)

        contents = await file.read()
        with open(image_path, "wb") as f:
            f.write(contents)

        results = model(image_path)
        result = results[0]

        if result.probs is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Model khong tra ve ket qua phan loai"}
            )

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = result.names[class_id]

        confidence_percent = round(confidence * 100, 2)

        web_message = message_map.get(class_name, class_name)
        oled_message = oled_message_map.get(class_name, "Chua Xac Dinh")

        latest_result = {
            "class_name": class_name,
            "message": oled_message,
            "web_message": web_message,
            "confidence": round(confidence, 4),
            "confidence_percent": confidence_percent
        }

        print("========== KET QUA AI ==========")
        print(f"Class name: {class_name}")
        print(f"Web message: {web_message}")
        print(f"OLED message: {oled_message}")
        print(f"Confidence: {confidence_percent}%")
        print(f"LED mode: {led_control['mode']}")
        print("================================")

        return {
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "confidence_percent": confidence_percent,
            "message": web_message,
            "oled_message": oled_message,
            "led_mode": led_control["mode"]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ==================== APP WEB ĐIỆN THOẠI / LAPTOP ====================
@app.get("/app", response_class=HTMLResponse)
def mobile_app():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice AI IoT App</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: white;
        }

        .container {
            max-width: 480px;
            margin: auto;
            padding: 20px;
        }

        .card {
            background: #1e293b;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }

        h1 {
            text-align: center;
            font-size: 26px;
            margin-bottom: 6px;
        }

        .subtitle {
            text-align: center;
            color: #cbd5e1;
            margin-bottom: 20px;
        }

        .result {
            font-size: 18px;
            line-height: 1.8;
        }

        .label {
            color: #94a3b8;
        }

        .value {
            font-weight: bold;
            color: #facc15;
        }

        button {
            width: 100%;
            padding: 16px;
            margin-top: 10px;
            border: none;
            border-radius: 14px;
            color: white;
            font-size: 17px;
            font-weight: bold;
        }

        .btn-auto {
            background: #2563eb;
        }

        .btn-on {
            background: #dc2626;
        }

        .btn-off {
            background: #16a34a;
        }

        .btn-upload {
            background: #9333ea;
        }

        input[type="file"] {
            width: 100%;
            margin-top: 12px;
            padding: 12px;
            background: #334155;
            border-radius: 12px;
            color: white;
        }

        .status {
            margin-top: 12px;
            color: #cbd5e1;
            font-size: 15px;
            text-align: center;
        }

        .ok {
            color: #22c55e;
        }

        .warning {
            color: #facc15;
        }

        .danger {
            color: #ef4444;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Rice AI IoT App</h1>
        <div class="subtitle">Nhận diện bệnh cây lúa + điều khiển LED</div>

        <div class="card">
            <h2>Kết quả AI</h2>
            <div class="result">
                <div><span class="label">Bệnh:</span> <span class="value" id="disease">---</span></div>
                <div><span class="label">Class:</span> <span id="className">---</span></div>
                <div><span class="label">Độ tin cậy:</span> <span id="confidence">0%</span></div>
                <div><span class="label">Chế độ LED:</span> <span id="ledMode">auto</span></div>
            </div>
            <div class="status" id="statusText">Đang tải dữ liệu...</div>
        </div>

        <div class="card">
            <h2>Điều khiển LED</h2>
            <button class="btn-off" onclick="setLed('off')">TẮT TẤT CẢ LED</button>
            <button class="btn-on" onclick="setLed('on')">BẬT CẢNH BÁO ĐỎ</button>
            <button class="btn-auto" onclick="setLed('auto')">AUTO THEO AI</button>
        </div>

        <div class="card">
            <h2>Gửi ảnh lên AI</h2>
            <input id="imageInput" type="file" accept="image/*" capture="environment">
            <button class="btn-upload" onclick="uploadImage()">PHÂN TÍCH ẢNH</button>
            <div class="status" id="uploadStatus">Chọn hoặc chụp ảnh lá lúa để phân tích.</div>
        </div>
    </div>

    <script>
        async function loadStatus() {
            try {
                const res = await fetch('/latest');
                const data = await res.json();

                document.getElementById('disease').innerText = data.web_message || data.message || 'Chưa có kết quả';
                document.getElementById('className').innerText = data.class_name || '---';
                document.getElementById('confidence').innerText = (data.confidence_percent || 0) + '%';
                document.getElementById('ledMode').innerText = data.led_mode || 'auto';

                document.getElementById('statusText').innerText = 'Đã kết nối server';
                document.getElementById('statusText').className = 'status ok';
            } catch (err) {
                document.getElementById('statusText').innerText = 'Không kết nối được server';
                document.getElementById('statusText').className = 'status danger';
            }
        }

        async function setLed(mode) {
            try {
                await fetch('/led/' + mode, { method: 'POST' });
                await loadStatus();
            } catch (err) {
                alert('Không gửi được lệnh LED');
            }
        }

        async function uploadImage() {
            const input = document.getElementById('imageInput');
            const file = input.files[0];

            if (!file) {
                alert('Vui lòng chọn hoặc chụp ảnh trước');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('uploadStatus').innerText = 'Đang gửi ảnh lên AI...';
            document.getElementById('uploadStatus').className = 'status warning';

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();

                if (res.ok) {
                    document.getElementById('uploadStatus').innerText =
                        'Đã phân tích: ' + data.message + ' - ' + data.confidence_percent + '%';
                    document.getElementById('uploadStatus').className = 'status ok';
                    await loadStatus();
                } else {
                    document.getElementById('uploadStatus').innerText =
                        'Lỗi: ' + JSON.stringify(data);
                    document.getElementById('uploadStatus').className = 'status danger';
                }
            } catch (err) {
                document.getElementById('uploadStatus').innerText = 'Không gửi được ảnh';
                document.getElementById('uploadStatus').className = 'status danger';
            }
        }

        loadStatus();
        setInterval(loadStatus, 1000);
    </script>
</body>
</html>
    """