from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['GENERATED_FOLDER'] = 'generated/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 Megabytes limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
LOGO_STYLES = [
    "Minimalistic", "Futuristic", "Vintage or Retro", "Hand-Drawn or Artistic",
    "Corporate", "Eco-Friendly or Natural", "Luxury or Elegant", "Bold and Colorful",
    "Geometric", "Abstract", "Typography-Based", "Cultural or Ethnic",
    "Sporty or Athletic", "Mascot", "Tech or Digital"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_sketch', methods=['POST'])
def upload_sketch():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({"message": "Upload successful. Please choose a logo style.", "styles": LOGO_STYLES, "sketch_filename": filename})

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/choose_style', methods=['POST'])
def choose_style():
    data = request.json
    style = data.get('style')
    sketch_filename = data.get('sketch_filename')

    if not style or style not in LOGO_STYLES:
        return jsonify({"error": "Invalid style chosen."}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], sketch_filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Sketch file not found."}), 404

    return jsonify({"message": "Style chosen. Please provide additional details.", "sketch_filename": sketch_filename, "style": style})

@app.route('/finalize_logo', methods=['POST'])
def finalize_logo():
    data = request.json
    sketch_filename = data.get('sketch_filename')
    style = data.get('style')
    business_name = data.get('business_name', '')
    background_color = data.get('background_color', '#FFFFFF')  # Default to white

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], sketch_filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Sketch file not found."}), 404

    logo_filename = generate_logo(filepath, style, business_name, background_color)
    logo_path = os.path.join(app.config['GENERATED_FOLDER'], logo_filename)

    return send_from_directory(app.config['GENERATED_FOLDER'], logo_filename, as_attachment=True)

def generate_logo(sketch_filepath, style, business_name, background_color):
    # Placeholder for AI logo generation
    # In a real application, this should be replaced with actual AI service integration
    filename = os.path.basename(sketch_filepath)
    generated_filename = f"generated_{filename}"
    generated_filepath = os.path.join(app.config['GENERATED_FOLDER'], generated_filename)
    os.rename(sketch_filepath, generated_filepath)

    return generated_filename

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
    app.run(debug=True)
