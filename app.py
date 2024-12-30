from flask import Flask, request, jsonify, make_response
import easyocr
import os
import cv2
import numpy as np
import logging
import sys

# Initialize Flask app and logging
app = Flask(__name__)
application = app  # Required for cPanel and WSGI
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("flask_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads directory exists

def is_image_quality_low(image_path):
    """
    Check the quality of the image.
    Criteria: Low brightness or low sharpness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error("Failed to load image.")
        return "Failed to load image."

    # Check brightness
    brightness = np.mean(image)
    if brightness < 50:
        logging.warning("Image is too dark.")
        return "Image is too dark."

    # Check sharpness
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    if laplacian_var < 100:
        logging.warning("Image is too blurry.")
        return "Image is too blurry."

    return None


@app.route('/', methods=['GET'])
def home():
    """
    Home route that returns a welcome message.
    """
    logging.debug("Accessed the home route.")
    return jsonify({
        'message': 'Welcome to the NID OCR API!',
        'endpoints': {
            'POST /image_to_text': 'Upload an image to extract text.',
        },
        'status': 'Running',
    }), 200


@app.route('/image_to_text', methods=['POST'])
def nid_ocr():
    """
    Endpoint to process an image and extract text using OCR.
    """
    try:
        # Debug incoming request
        logging.debug(f"Headers: {request.headers}")
        logging.debug(f"Form Data: {request.form}")
        logging.debug(f"Files: {request.files}")

        if 'file' not in request.files:
            logging.error("No file part in the request.")
            return jsonify({'error': 'No file part in the request'}), 400

        image_file = request.files['file']
        if image_file.filename == '':
            logging.error("No file selected.")
            return jsonify({'error': 'No file selected'}), 400

        # Save file
        temp_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(temp_path)

        # Check image quality
        quality_issue = is_image_quality_low(temp_path)
        if quality_issue:
            os.remove(temp_path)
            return jsonify({'error': quality_issue}), 400

        # Initialize EasyOCR Reader
        try:
            reader = easyocr.Reader(['en', 'bn'], gpu=True)
        except Exception as gpu_error:
            logging.warning(f"GPU not available ({gpu_error}), falling back to CPU.")
            reader = easyocr.Reader(['en', 'bn'], gpu=False)

        # Perform OCR
        result = reader.readtext(temp_path)
        os.remove(temp_path)

        # Extract text and parse it
        raw_text = [detection[1] for detection in result]
        corrected_data = parse_nid_data(raw_text)

        # Prepare response
        response = jsonify(corrected_data)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response, 200

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        response = jsonify({'error': str(e)})
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response, 400


def parse_nid_data(raw_text):
    """
    Parse the text extracted from an ID and handle both temporary and permanent ID formats.
    """
    correct_fields = {
        "নাম": "Name_bn",
        "Name": "Name_en",
        "পিতা": "Father's Name",
        "মাতা": "Mother's Name",
        "Date of Birth": "Date of Birth",
        "NID No": "NID Number",
        "ID NO": "NID Number",
    }

    corrections = {
        "মাম": "নাম",
        "মাঢা": "মাতা",
        "Dare of Birth": "Date of Birth",
        "ID NO": "NID No",
    }

    # Initialize dictionary to store corrected data
    data = {}
    temp_key = None  # Temporarily hold the key while processing

    # Correct OCR mistakes using the corrections dictionary
    raw_text = [corrections.get(text, text) for text in raw_text]

    # Process raw text and extract fields
    for i, text in enumerate(raw_text):
        logging.debug(f'raw_text[{i}]: {text}')
        
        # Match fields based on exact matches
        if text in correct_fields:
            temp_key = correct_fields[text]
        elif temp_key:
            # Assign the value to the identified key
            if temp_key not in data:
                data[temp_key] = text
            temp_key = None

    return data


if __name__ == '__main__':
    # Ensure UTF-8 is supported
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    app.run(host='0.0.0.0', port=5000, debug=True)
