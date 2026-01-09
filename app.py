from flask import Flask, request, render_template_string, url_for, send_from_directory
import os, cv2, torch, numpy as np
from werkzeug.utils import secure_filename
from unet_model import UNet   # import your UNet model class

# ----------------------- SETTINGS -----------------------
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
GT_FOLDER = "static/ground_truth"
MODEL_PATH = "model/best_unet_salt.pth"
IMG_SIZE = 128

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(GT_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Metrics From Evaluation Script --------
DICE = 0.6750
IOU  = 0.6374
EPOCHS_TRAINED = 150

# ----------------------- LOAD MODEL -----------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model Loaded on {DEVICE}")

# ----------------------- FUNCTIONS -----------------------
def predict_mask(img_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = (img - 0.5) / 0.5
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(t)).cpu().numpy()[0,0]

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_path, mask)

def load_ground_truth(fname):
    path = os.path.join(GT_FOLDER, fname)
    return path if os.path.exists(path) else None

# ----------------------- HTML TEMPLATE -----------------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<title>Salt Segmentation using UNet | Deep Learning Project</title>
<style>
body { background:#f4f6fc; font-family:Arial; margin:0; }
.nav { background:#0B3D91; padding:15px; color:white; font-size:22px; font-weight:bold; }
.container { width:90%; margin:auto; padding:20px; }
.section-title { font-size:26px; font-weight:bold; color:#0B3D91; margin-top:30px; }
.card {
  background:white; padding:18px; border-radius:8px;
  box-shadow:0 4px 8px rgba(0,0,0,0.1); margin:12px 0;
}
ul, ol { line-height:1.7; }
button, input {
  padding:10px 15px; font-size:16px; margin-top:10px;
  border:none; border-radius:6px;
}
button { background:#0B3D91; color:white; cursor:pointer; }
button:hover { background:#082d6b; }
img {
  width:300px; border-radius:8px; border:3px solid #333; margin:5px;
}
.footer { margin-top:40px; background:#0B3D91; color:white; padding:12px; text-align:center; }
.kbd { background:#eef0f6; padding:3px 6px; border-radius:4px; font-family:monospace; }
.note { background:#FFF7CC; padding:10px; border-left:4px solid #F4C300; margin-top:10px; }
</style>
</head>

<body>
<div class="nav">Salt Segmentation using UNet – Deep Learning & Flask</div>
<div class="container">

<div class="card">
  <h2 class="section-title">📘 Introduction</h2>
  <p>
  Salt structures beneath earth surface help geologists locate oil & gas reservoirs. 
  This project uses a <b>UNet deep learning model</b> for automatic salt segmentation from seismic images.
  It is deployed using Flask for real-time prediction.
  </p>
</div>

<div class="card">
  <h2 class="section-title">⚙️ High-Level Workflow</h2>
  <ol>
    <li>Input grayscale seismic image</li>
    <li>Normalize & resize to 128x128</li>
    <li>UNet model predicts salt region mask</li>
    <li>Ground truth mask (if available) compared</li>
    <li>Output mask displayed on UI</li>
  </ol>
</div>

<div class="card">
  <h2 class="section-title">🧠 Deep System Explanation (Step-by-Step)</h2>
  <ol>
    <li>Dataset loaded (images + masks)</li>
    <li>Resize, normalize to [-1,1]</li>
    <li>PyTorch <b>DataLoader</b> feeds model</li>
    <li>UNet encoder extracts spatial features</li>
    <li>Decoder reconstructs segmentation mask</li>
    <li>Sigmoid → probability → threshold to mask</li>
    <li>Best model saved as <span class="kbd">.pth</span></li>
    <li>Flask loads model once on startup</li>
    <li>User uploads → model predicts mask</li>
    <li>Show: original + ground truth + prediction</li>
  </ol>
  <div class="note">Trained for <b>{{epochs}}</b> epochs using BCE + Dice + Focal Loss.</div>
</div>

<div class="card">
<h2 class="section-title">📤 Upload Image for Segmentation</h2>
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <button type="submit">Run Segmentation</button>
</form>
</div>

{% if input_img %}
<div class="card">
<h2 class="section-title">🔎 Results</h2>

<b>Input Image</b><br>
<img src="{{ input_img }}">

{% if ground_truth_img %}
<b>Ground Truth Mask</b><br>
<img src="{{ ground_truth_img }}">
{% else %}
<p style="color:red;"><b>No Ground Truth Found</b></p>
{% endif %}

<b>Predicted Mask</b><br>
<img src="{{ output_img }}">
</div>
{% endif %}

<a href="/show_all"><button>📁 View All Dataset Predictions</button></a>

<div class="card">
<h2 class="section-title">📊 Model Performance</h2>
<ul>
<li>Training Epochs: <b>{{epochs}}</b></li>
<li>Dice Score: <b>{{dice}}</b></li>
<li>IoU Score: <b>{{iou}}</b></li>
<li>Architecture: UNet CNN</li>
<li>Framework: PyTorch + Flask</li>
</ul>
</div>

<div class="card">
<h2 class="section-title">✅ Applications</h2>
<ul>
<li>Oil & Gas Reservoir Detection</li>
<li>Geological Subsurface Analysis</li>
<li>Earth Science & Mining Research</li>
<li>Remote Sensing & Resource Mapping</li>
</ul>
</div>

<div class="card">
<h2 class="section-title">🚀 Future Enhancements</h2>
<ul>
<li>Train 256×256 or 512×512 resolution</li>
<li>Use UNet++ / Attention-UNet for higher accuracy</li>
<li>Overlay prediction heatmaps</li>
<li>Cloud GPU deployment (AWS/GCP)</li>
</ul>
</div>

</div>
<div class="footer">© Semester 3 Project | UNet Salt Segmentation</div>
</body>
</html>
"""

# ----------------------- FLASK ROUTES -----------------------
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():
    input_img = output_img = ground_truth_img = None

    if request.method == "POST":
        file = request.files["file"]
        fname = secure_filename(file.filename)

        in_path = os.path.join(UPLOAD_FOLDER, fname)
        out_path = os.path.join(RESULT_FOLDER, fname)
        file.save(in_path)

        predict_mask(in_path, out_path)

        input_img = url_for("static_files", filename=f"uploads/{fname}")
        output_img = url_for("static_files", filename=f"results/{fname}")
        
        gt = load_ground_truth(fname)
        if gt:
            ground_truth_img = url_for("static_files", filename=f"ground_truth/{fname}")

    return render_template_string(HTML,
        input_img=input_img,
        output_img=output_img,
        ground_truth_img=ground_truth_img,
        dice=DICE, iou=IOU, epochs=EPOCHS_TRAINED)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/show_all")
def show_all():
    files = os.listdir(RESULT_FOLDER)
    html = "<h2>All Predictions</h2>"
    for f in files:
        html += f"<p><b>{f}</b></p>"
        html += f"<img src='/static/uploads/{f}' width='250'>"
        html += f"<img src='/static/results/{f}' width='250'><hr>"
    return html

if __name__ == "__main__":
    app.run(debug=True)
