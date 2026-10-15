import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from tensorflow.keras.models import load_model

# Import custom modular scripts
from src.preprocessing import preprocess_image
from src.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from src.report_generator import generate_pdf_report

app = Flask(__name__)

# Configure upload and output folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'
app.config['REPORTS_FOLDER'] = 'static/reports'

# Load the trained model at startup (ensure you have trained and saved it first)
MODEL_PATH = 'models/pneumonia_resnet50.keras'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print("Warning: Trained model not found in models/. Run train.py first.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and model:
            # 1. Save uploaded image
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 2. Preprocess and predict
            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)[0][0]
            
            # Binary classification logic (Normal vs Pneumonia)
            result = "Pneumonia Detected" if prediction > 0.5 else "Normal"
            confidence = float(prediction if prediction > 0.5 else 1 - prediction) * 100
            confidence = round(confidence, 2)
            
            # 3. Generate Grad-CAM Heatmap
            heatmap = make_gradcam_heatmap(filepath, model, last_conv_layer_name="conv5_block3_out")
            heatmap_filename = f"heatmap_{filename}"
            heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
            save_and_display_gradcam(filepath, heatmap, cam_path=heatmap_path)
            
            # 4. Generate PDF Report
            pdf_filename = f"report_{filename.rsplit('.', 1)[0]}.pdf"
            pdf_path = os.path.join(app.config['REPORTS_FOLDER'], pdf_filename)
            generate_pdf_report(pdf_path, filepath, heatmap_path, result, confidence)
            
            return render_template(
                'index.html',
                result=result,
                confidence=confidence,
                image_path=url_for('static', filename=f'uploads/{filename}'),
                heatmap_path=url_for('static', filename=f'heatmaps/{heatmap_filename}'),
                pdf_download=url_for('download_report', filename=pdf_filename)
            )
            
    return render_template('index.html', result=None)

@app.route('/download/<filename>')
def download_report(filename):
    pdf_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    # Ensure necessary folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
    
    app.run(debug=True)