import os
import requests
import json
import sqlite3
from flask import Flask, request, jsonify, abort, send_from_directory
from werkzeug.utils import secure_filename
from websocket import create_connection
from datetime import datetime

# Server initialization
app = Flask(__name__)

# Database connection
conn = sqlite3.connect('server_data.db', check_same_thread=False)
c = conn.cursor()

# Create tables if not exist
def initialize_database():
    c.execute('''CREATE TABLE IF NOT EXISTS files (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 filename TEXT,
                 uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    conn.commit()

# Define routes for file handling and operations
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        abort(400, 'No file part')
    
    file = request.files['file']
    if file.filename == '':
        abort(400, 'No selected file')

    filename = secure_filename(file.filename)
    file_path = os.path.join('/path/to/the/uploads', filename)
    file.save(file_path)

    # Insert file information into the database
    c.execute('INSERT INTO files (filename) VALUES (?)', (filename,))
    conn.commit()

    return jsonify({'message': 'File uploaded successfully', 'filename': filename})

@app.route('/files/<int:file_id>', methods=['GET'])
def download_file(file_id):
    c.execute('SELECT filename FROM files WHERE id = ?', (file_id,))
    file_record = c.fetchone()
    if file_record:
        filename = file_record[0]
        return send_from_directory('/path/to/the/uploads', filename)
    else:
        abort(404, 'File not found')

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    # Example processing logic
    processed_data = {'status': 'processed', 'data': data}
    return jsonify(processed_data)

# Function to interact with external API
def call_external_api():
    try:
        response = requests.get('https://api.external.com/data')
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        abort(500, f'External API call failed: {e}')

# WebSocket connection for real-time updates
def websocket_connection():
    ws = create_connection("ws://example.com/websocket")
    try:
        ws.send(json.dumps({"request": "data", "timestamp": str(datetime.now())}))
        result = ws.recv()
        return json.loads(result)
    except Exception as e:
        abort(500, f'WebSocket error: {e}')
    finally:
        ws.close()

@app.route('/websocket-test', methods=['GET'])
def test_websocket():
    result = websocket_connection()
    return jsonify({'websocket_response': result})

@app.route('/external-api-test', methods=['GET'])
def test_external_api():
    api_data = call_external_api()
    return jsonify({'api_data': api_data})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

# Run server
if __name__ == '__main__':
    initialize_database()
    app.run(debug=True, port=5000)
