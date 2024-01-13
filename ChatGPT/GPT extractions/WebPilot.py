from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Endpoint for webPageReader
@app.route('/webPageReader', methods=['POST'])
def web_page_reader():
    data = request.json
    # Call to WebPilot webPageReader API
    # You need to replace 'webpilot_api_endpoint' with the actual API endpoint and handle authentication as required.
    response = requests.post('webpilot_api_endpoint/webPageReader', json=data)
    return jsonify(response.json())

# Endpoint for longContentWriter
@app.route('/longContentWriter', methods=['POST'])
def long_content_writer():
    data = request.json
    # Call to WebPilot longContentWriter API
    # Handle API endpoint and authentication
    response = requests.post('webpilot_api_endpoint/longContentWriter', json=data)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
