import os
import glob
import io
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np
import cv2
from PIL import Image
import torch
from models.restoration_model import SemiconRestorationNet
from utils.metrics import calculate_psnr, calculate_ssim

# Global Model & Data Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = SemiconRestorationNet().to(DEVICE)
WEIGHTS_PATH = 'weights/best_model.pth'
if os.path.exists(WEIGHTS_PATH):
    MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
MODEL.eval()

NOISY_FILES = sorted(glob.glob('dataset/train/NoisyLR/*.npy'))
GT_FILES = sorted(glob.glob('dataset/train/GT/*.npy'))

def ndarray_to_b64png(arr):
    """Normalizes float array [0,1] or raw intensity and converts to base64 PNG string."""
    arr_norm = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr_norm * 255.0).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KLA Semiconductor Image Restoration Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0F172A;
            --card-bg: #1E293B;
            --cyan-accent: #06B6D4;
            --emerald-accent: #10B981;
            --white-text: #F8FAFC;
            --muted-text: #94A3B8;
            --border-color: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--white-text);
            padding: 24px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }
        h1 { font-size: 24px; font-weight: 800; color: var(--cyan-accent); display: flex; align-items: center; gap: 10px; }
        .badge { font-size: 12px; background: rgba(6, 182, 212, 0.15); color: var(--cyan-accent); padding: 4px 10px; borderRadius: 20px; border: 1px solid var(--cyan-accent); }
        .main-layout { display: grid; grid-template-columns: 340px 1fr; gap: 24px; }
        .sidebar { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }
        .sidebar h2 { font-size: 16px; margin-bottom: 16px; color: var(--emerald-accent); }
        label { font-size: 13px; color: var(--muted-text); margin-bottom: 8px; display: block; font-weight: 600; }
        select, input[type="file"], button {
            width: 100%; padding: 12px; background: #0F172A; color: var(--white-text); border: 1px solid var(--border-color);
            border-radius: 8px; font-size: 14px; font-family: inherit; margin-bottom: 16px; cursor: pointer; outline: none;
        }
        select:focus, input[type="file"]:focus { border-color: var(--cyan-accent); }
        button { background: var(--cyan-accent); color: #0F172A; font-weight: 700; transition: all 0.2s; border: none; }
        button:hover { background: #22d3ee; box-shadow: 0 0 15px rgba(6, 182, 212, 0.4); }
        .upload-box {
            border: 2px dashed var(--border-color); border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 16px; background: rgba(15, 23, 42, 0.5);
        }
        .upload-box:hover { border-color: var(--cyan-accent); }
        .viewer-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .image-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; text-align: center; }
        .image-card h3 { font-size: 15px; margin-bottom: 12px; }
        .image-card.degraded h3 { color: var(--cyan-accent); }
        .image-card.restored h3 { color: var(--emerald-accent); }
        .image-card.gt h3 { color: var(--white-text); }
        .img-wrapper { background: #000; border-radius: 8px; overflow: hidden; position: relative; height: 320px; display: flex; align-items: center; justify-content: center; }
        .img-wrapper img { max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: pixelated; }
        .metrics-hud { margin-top: 24px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
        .stat-box { text-align: center; border-right: 1px solid var(--border-color); padding-right: 16px; }
        .stat-box:last-child { border-right: none; }
        .stat-label { font-size: 12px; color: var(--muted-text); margin-bottom: 4px; }
        .stat-value { font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        .stat-value.green { color: var(--emerald-accent); }
        .stat-value.cyan { color: var(--cyan-accent); }
    </style>
</head>
<body>
    <header>
        <h1>⚡ KLA Semiconductor AI Restoration Studio <span class="badge">SINGLE & BATCH INFERENCE ENGINE</span></h1>
        <div style="color: var(--muted-text); font-size: 13px;">SemiconRestorationNet v1.0 | PyTorch</div>
    </header>

    <div class="main-layout">
        <div class="sidebar">
            <h2>Option A: Select Dataset Image</h2>
            <label for="sampleSelect">Dataset Degraded Sample:</label>
            <select id="sampleSelect" onchange="runInference()">
                <!-- Populated dynamically -->
            </select>
            <button onclick="runInference()">⚡ Restore Dataset Image</button>

            <h2 style="margin-top: 24px;">Option B: Single Custom File</h2>
            <div class="upload-box">
                <label for="customFileInput">Upload Single (.npy, .png, .jpg):</label>
                <input type="file" id="customFileInput" accept=".npy,.png,.jpg,.jpeg" onchange="uploadSingleFile()">
            </div>
            
            <div style="margin-top: 20px; font-size: 13px; color: var(--muted-text); line-height: 1.6;">
                <strong style="color: var(--white-text);">Model Specification:</strong><br>
                • Gated Residual UNet (NAFNet)<br>
                • PixelShuffle 2x Super Resolution<br>
                • Input: 128x128 Degraded (Noise + Blur)<br>
                • Output: 256x256 Clean Reconstructed
            </div>
        </div>

        <div class="right-panel">
            <div class="viewer-container">
                <div class="image-card degraded">
                    <h3>1. Degraded Input (128×128)</h3>
                    <div class="img-wrapper"><img id="imgDegraded" src="" alt="Degraded Input"></div>
                    <p id="degradedRange" style="font-size: 12px; color: var(--muted-text); margin-top: 8px;">Range: --</p>
                </div>
                <div class="image-card restored">
                    <h3>2. AI Restored Output (256×256)</h3>
                    <div class="img-wrapper"><img id="imgRestored" src="" alt="AI Restored"></div>
                    <p style="font-size: 12px; color: var(--emerald-accent); margin-top: 8px;">Target Bounded [0.0, 1.0]</p>
                </div>
                <div class="image-card gt">
                    <h3>3. Ground Truth Reference</h3>
                    <div class="img-wrapper"><img id="imgGT" src="" alt="Ground Truth"></div>
                    <p id="gtLabel" style="font-size: 12px; color: var(--muted-text); margin-top: 8px;">Full Res Clean Reference</p>
                </div>
            </div>

            <div class="metrics-hud">
                <div class="stat-box">
                    <div class="stat-label">PSNR IMPROVEMENT</div>
                    <div class="stat-value green" id="valPSNR">-- dB</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">STRUCTURAL SSIM SCORE</div>
                    <div class="stat-value green" id="valSSIM">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">INFERENCE LATENCY</div>
                    <div class="stat-value cyan" id="valLatency">-- ms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">BICUBIC BASELINE PSNR</div>
                    <div class="stat-value" id="valBicubic">-- dB</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function loadSampleList() {
            const res = await fetch('/api/samples');
            const samples = await res.json();
            const select = document.getElementById('sampleSelect');
            select.innerHTML = '';
            samples.forEach((s, idx) => {
                const opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = `Sample #${idx+1} (${s})`;
                select.appendChild(opt);
            });
            runInference();
        }

        async function runInference() {
            const idx = document.getElementById('sampleSelect').value || 0;
            const res = await fetch(`/api/restore?idx=${idx}`);
            const data = await res.json();

            document.getElementById('imgDegraded').src = 'data:image/png;base64,' + data.b64_degraded;
            document.getElementById('imgRestored').src = 'data:image/png;base64,' + data.b64_restored;
            document.getElementById('imgGT').src = 'data:image/png;base64,' + data.b64_gt;
            document.getElementById('gtLabel').textContent = "Full Res Clean Reference";

            document.getElementById('degradedRange').textContent = `Range: [${data.min_val.toFixed(2)}, ${data.max_val.toFixed(2)}]`;
            document.getElementById('valPSNR').textContent = `${data.psnr_model.toFixed(2)} dB`;
            document.getElementById('valSSIM').textContent = `${data.ssim_model.toFixed(4)}`;
            document.getElementById('valLatency').textContent = `${data.latency_ms.toFixed(1)} ms`;
            document.getElementById('valBicubic').textContent = `${data.psnr_bicubic.toFixed(2)} dB`;
        }

        async function uploadSingleFile() {
            const fileInput = document.getElementById('customFileInput');
            if (!fileInput.files || fileInput.files.length === 0) return;

            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/api/restore_single', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            document.getElementById('imgDegraded').src = 'data:image/png;base64,' + data.b64_degraded;
            document.getElementById('imgRestored').src = 'data:image/png;base64,' + data.b64_restored;
            document.getElementById('imgGT').src = 'data:image/png;base64,' + data.b64_restored;
            document.getElementById('gtLabel').textContent = "Custom Single Image Restored";

            document.getElementById('degradedRange').textContent = `Range: [${data.min_val.toFixed(2)}, ${data.max_val.toFixed(2)}]`;
            document.getElementById('valPSNR').textContent = `N/A (Single Custom Input)`;
            document.getElementById('valSSIM').textContent = `N/A`;
            document.getElementById('valLatency').textContent = `${data.latency_ms.toFixed(1)} ms`;
            document.getElementById('valBicubic').textContent = `N/A`;
        }

        window.onload = loadSampleList;
    </script>
</body>
</html>
"""

class VisualizerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif path == '/api/samples':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            sample_names = [os.path.basename(f) for f in NOISY_FILES]
            self.wfile.write(json.dumps(sample_names).encode('utf-8'))
        elif path == '/api/restore':
            idx = int(query.get('idx', [0])[0])
            if idx < 0 or idx >= len(NOISY_FILES):
                idx = 0

            noisy_path = NOISY_FILES[idx]
            gt_path = GT_FILES[idx]

            noisy_img = np.load(noisy_path).astype(np.float32)
            gt_img = np.load(gt_path).astype(np.float32)

            import time
            start = time.time()
            input_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                restored_tensor = MODEL(input_tensor)
            latency_ms = (time.time() - start) * 1000

            restored_img = restored_tensor.cpu().numpy().squeeze(0).squeeze(0)

            # Bicubic
            bicubic_img = cv2.resize(noisy_img, (256, 256), interpolation=cv2.INTER_CUBIC)
            bicubic_img = np.clip(bicubic_img, 0.0, 1.0)

            psnr_model = calculate_psnr(restored_img, gt_img)
            ssim_model = calculate_ssim(restored_img, gt_img)
            psnr_bicubic = calculate_psnr(bicubic_img, gt_img)

            b64_degraded = ndarray_to_b64png(noisy_img)
            b64_restored = ndarray_to_b64png(restored_img)
            b64_gt = ndarray_to_b64png(gt_img)

            resp = {
                'b64_degraded': b64_degraded,
                'b64_restored': b64_restored,
                'b64_gt': b64_gt,
                'min_val': float(noisy_img.min()),
                'max_val': float(noisy_img.max()),
                'psnr_model': float(psnr_model),
                'ssim_model': float(ssim_model),
                'psnr_bicubic': float(psnr_bicubic),
                'latency_ms': float(latency_ms)
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/restore_single':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # Extract raw file bytes from multipart form data or raw payload
            # Search for file bytes in boundary
            try:
                # Basic parsing for single file upload
                file_bytes = body
                if b'\r\n\r\n' in body:
                    file_bytes = body.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n--', 1)[0]

                # Try loading as numpy array first, else image
                try:
                    buf = io.BytesIO(file_bytes)
                    noisy_img = np.load(buf).astype(np.float32)
                except Exception:
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = np.zeros((128, 128), dtype=np.float32)
                    noisy_img = img.astype(np.float32) / 255.0

                import time
                start = time.time()
                input_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    restored_tensor = MODEL(input_tensor)
                latency_ms = (time.time() - start) * 1000

                restored_img = restored_tensor.cpu().numpy().squeeze(0).squeeze(0)

                b64_degraded = ndarray_to_b64png(noisy_img)
                b64_restored = ndarray_to_b64png(restored_img)

                resp = {
                    'b64_degraded': b64_degraded,
                    'b64_restored': b64_restored,
                    'min_val': float(noisy_img.min()),
                    'max_val': float(noisy_img.max()),
                    'latency_ms': float(latency_ms)
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))

def run_server(port=8050):
    server = HTTPServer(('127.0.0.1', port), VisualizerHandler)
    print(f"============================================================")
    print(f" KLA AI Restoration Interactive Studio Server Running")
    print(f" URL: http://127.0.0.1:{port}")
    print(f"============================================================")
    server.serve_forever()

if __name__ == '__main__':
    run_server()
