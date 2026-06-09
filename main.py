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
    "bacterial_leaf_blight": "Bạc lá",
    "brown_spot": "Đốm nâu",
    "healthy": "Khỏe mạnh",
    "leaf_blast": "Đạo ôn lá",
    "leaf_scald": "Cháy bờ lá",
    "sheath_blight": "Khô vằn"
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
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background:
                linear-gradient(rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.9)),
                url('/static/images/rice-bg.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }

        .container {
            width: 100%;
            max-width: 560px;
            margin: auto;
            padding: 24px 16px 40px;
        }

        .header {
            text-align: center;
            margin-bottom: 24px;
        }

        .header h1 {
            margin: 0;
            font-size: 32px;
            font-weight: 800;
        }

        .header p {
            margin: 10px 0 0;
            color: #dbeafe;
            font-size: 16px;
        }

        .badge {
            display: inline-block;
            margin-top: 14px;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.16);
            color: #86efac;
            font-size: 14px;
            font-weight: bold;
            border: 1px solid rgba(134, 239, 172, 0.35);
        }

        .card {
            background: rgba(30, 41, 59, 0.94);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.28);
            backdrop-filter: blur(8px);
        }

        .card h2 {
            margin: 0 0 18px;
            font-size: 24px;
        }

        .result-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            font-size: 17px;
        }

        .result-row:last-child {
            border-bottom: none;
        }

        .label {
            color: #cbd5e1;
        }

        .value {
            text-align: right;
            font-weight: bold;
            color: #facc15;
        }

        .healthy {
            color: #22c55e;
        }

        .danger {
            color: #ef4444;
        }

        .neutral {
            color: #facc15;
        }

        .upload-box {
            border: 2px dashed rgba(147, 197, 253, 0.35);
            border-radius: 18px;
            padding: 18px;
            background: rgba(15, 23, 42, 0.42);
        }

        input[type="file"] {
            width: 100%;
            padding: 14px;
            border-radius: 14px;
            border: none;
            background: #334155;
            color: white;
            font-size: 15px;
        }

        .preview {
            width: 100%;
            margin-top: 16px;
            display: none;
            border-radius: 16px;
            overflow: hidden;
            background: #020617;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .preview img {
            width: 100%;
            display: block;
            max-height: 360px;
            object-fit: contain;
        }

        button {
            width: 100%;
            padding: 16px;
            margin-top: 16px;
            border: none;
            border-radius: 16px;
            color: white;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            background: linear-gradient(135deg, #22c55e, #2563eb);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.25);
        }

        button:disabled {
            opacity: 0.65;
            cursor: not-allowed;
        }

        .status {
            margin-top: 14px;
            text-align: center;
            color: #cbd5e1;
            font-size: 15px;
            min-height: 22px;
        }

        .status.ok {
            color: #22c55e;
        }

        .status.warning {
            color: #facc15;
        }

        .status.error {
            color: #ef4444;
        }

        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            margin-top: 20px;
        }

        @media (max-width: 480px) {
            .header h1 {
                font-size: 27px;
            }

            .card {
                padding: 18px;
                border-radius: 18px;
            }

            .result-row {
                font-size: 15px;
            }

            button {
                font-size: 16px;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <h1>Rice AI IoT App</h1>
            <p>Nhận diện bệnh cây lúa bằng AI và hiển thị kết quả lên ESP32 OLED</p>
            <div class="badge" id="serverBadge">Đang kết nối Cloud...</div>
        </div>

        <div class="card">
            <h2>Kết quả AI</h2>

            <div class="result-row">
                <span class="label">Tên bệnh</span>
                <span class="value" id="disease">---</span>
            </div>

            <div class="result-row">
                <span class="label">Class AI</span>
                <span class="value" id="className">---</span>
            </div>

            <div class="result-row">
                <span class="label">Độ tin cậy</span>
                <span class="value" id="confidence">0%</span>
            </div>

            <div class="result-row">
                <span class="label">Trạng thái</span>
                <span class="value" id="aiStatus">Chưa phân tích</span>
            </div>

            <div class="status" id="statusText">Đang tải dữ liệu...</div>
        </div>

        <div class="card">
            <h2>Gửi ảnh lá lúa</h2>

            <div class="upload-box">
                <input id="imageInput" type="file" accept="image/*" capture="environment">

                <div class="preview" id="previewBox">
                    <img id="previewImage" src="" alt="Ảnh xem trước">
                </div>

                <button id="analyzeButton" onclick="uploadImage()">PHÂN TÍCH ẢNH</button>

                <div class="status" id="uploadStatus">
                    Chọn hoặc chụp ảnh lá lúa để gửi lên Cloud AI.
                </div>
            </div>
        </div>

        <div class="footer">
            Cloud API: Railway · ESP32 đọc kết quả tại /latest
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const previewBox = document.getElementById('previewBox');
        const previewImage = document.getElementById('previewImage');
        const analyzeButton = document.getElementById('analyzeButton');

        imageInput.addEventListener('change', function () {
            const file = imageInput.files[0];

            if (!file) {
                previewBox.style.display = 'none';
                previewImage.src = '';
                return;
            }

            previewImage.src = URL.createObjectURL(file);
            previewBox.style.display = 'block';
        });

        function updateResultUI(data) {
            const disease = data.web_message || data.message || 'Chưa có kết quả';
            const className = data.class_name || '---';
            const confidence = data.confidence_percent || 0;

            document.getElementById('disease').innerText = disease;
            document.getElementById('className').innerText = className;
            document.getElementById('confidence').innerText = confidence + '%';

            const aiStatus = document.getElementById('aiStatus');

            if (!className) {
                aiStatus.innerText = 'Chưa phân tích';
                aiStatus.className = 'value neutral';
            } else if (className === 'healthy') {
                aiStatus.innerText = 'Lá khỏe mạnh';
                aiStatus.className = 'value healthy';
            } else {
                aiStatus.innerText = 'Cảnh báo bệnh';
                aiStatus.className = 'value danger';
            }
        }

        async function loadStatus() {
            try {
                const res = await fetch('/latest');
                const data = await res.json();

                updateResultUI(data);

                document.getElementById('statusText').innerText = 'Đã kết nối Cloud';
                document.getElementById('statusText').className = 'status ok';

                document.getElementById('serverBadge').innerText = 'Cloud đang hoạt động';
            } catch (err) {
                document.getElementById('statusText').innerText = 'Không kết nối được Cloud';
                document.getElementById('statusText').className = 'status error';

                document.getElementById('serverBadge').innerText = 'Cloud lỗi kết nối';
            }
        }

        async function uploadImage() {
            const file = imageInput.files[0];

            if (!file) {
                alert('Vui lòng chọn hoặc chụp ảnh trước');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            analyzeButton.disabled = true;
            analyzeButton.innerText = 'ĐANG PHÂN TÍCH...';

            document.getElementById('uploadStatus').innerText = 'Đang gửi ảnh lên Cloud AI...';
            document.getElementById('uploadStatus').className = 'status warning';

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();

                if (res.ok) {
                    updateResultUI(data);

                    document.getElementById('uploadStatus').innerText =
                        'Phân tích xong: ' + data.message + ' - ' + data.confidence_percent + '%';
                    document.getElementById('uploadStatus').className = 'status ok';

                    await loadStatus();
                } else {
                    document.getElementById('uploadStatus').innerText =
                        'Lỗi server: ' + JSON.stringify(data);
                    document.getElementById('uploadStatus').className = 'status error';
                }
            } catch (err) {
                document.getElementById('uploadStatus').innerText =
                    'Không gửi được ảnh lên Cloud';
                document.getElementById('uploadStatus').className = 'status error';
            }

            analyzeButton.disabled = false;
            analyzeButton.innerText = 'PHÂN TÍCH ẢNH';
        }

        loadStatus();
        setInterval(loadStatus, 3000);
    </script>
</body>
</html>
    """