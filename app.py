from flask import Flask, request, jsonify
import easyocr
import os
import cv2
import numpy as np
import logging
import sys
import re

# Initialize Flask app
app = Flask(__name__)
application = app  # For WSGI servers

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("flask_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Directory for uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    if brightness < 75:
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
        'status': 'Running',
    }), 200

@app.route('/image_to_text', methods=['POST'])
def nid_ocr():
    """
    Endpoint to process images (front and back parts of an ID) and extract text using OCR.
    """
    try:
        # Debug incoming request
        #logging.debug(f"Headers: {request.headers}")
        #logging.debug(f"Form Data: {request.form}")
        logging.debug(f"Files: {request.files.to_dict()}")

        # Validate file presence
        if 'front_part' not in request.files or 'back_part' not in request.files:
            received_files = list(request.files.keys())
            logging.error(f"Missing files. Received: {received_files}")
            return jsonify({
                'error': "Both 'front_part' and 'back_part' files are required",
                'received_files': received_files
            }), 400

        front_image = request.files['front_part']
        back_image = request.files['back_part']

        if front_image.filename == '' or back_image.filename == '':
            logging.error("Both files must be provided.")
            return jsonify({'error': "Both files must be provided"}), 400

        # Save files
        front_path = os.path.join(UPLOAD_FOLDER, front_image.filename)
        back_path = os.path.join(UPLOAD_FOLDER, back_image.filename)
        front_image.save(front_path)
        back_image.save(back_path)

        # Check image quality
        for file_path in [front_path, back_path]:
            quality_issue = is_image_quality_low(file_path)
            if quality_issue:
                os.remove(file_path)
                return jsonify({'error': quality_issue}), 400

        # Initialize EasyOCR Reader
        try:
            reader = easyocr.Reader(['en', 'bn'], gpu=True)
        except Exception as gpu_error:
            logging.warning(f"GPU not available ({gpu_error}), falling back to CPU.")
            reader = easyocr.Reader(['en', 'bn'], gpu=False)

        # Perform OCR on both images
        front_result = reader.readtext(front_path)
        back_result = reader.readtext(back_path)
        
        
        # Clean up temporary files
        os.remove(front_path)
        os.remove(back_path)

        # Extract and parse text from both results
        front_text = [detection[1] for detection in front_result]
        back_text = [detection[1] for detection in back_result]

        front_data = parse_nid_data(front_text)
        back_data = parse_back_data(back_text)
        

        # Combine data
        combined_data = {**front_data, **back_data}

        # Prepare response
        response = jsonify(combined_data)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response, 200

    except Exception as e:
        logging.error(f"Error processing images: {e}")
        response = jsonify({'error': str(e)})
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response, 400

def parse_nid_data(front_text):
    
    print('front part', front_text)
    
    """
    Parse the text extracted from the front part of the ID.
    """
    correct_fields = {
        "নাম": "name_bn",
        "Name": "name_en",
        "পিতা": "father_name",
        "মাতা": "mother_name",
        "Date of Birth": "date_of_birth",
        "NID No": "nid_number",
        "ID NO": "nid_number",
    }

    corrections = {
        "মাম": "নাম",
        "মাহা": "মাতা",
        "Nlare": "Name",
        "মাঢা": "মাতা",
        "এম, এম": "নাম",
        "Dare of Birth": "Date of Birth",
        "ID NO": "NID No",
        "WID ^0": "NID No"
    }

    data = {}
    temp_key = None

    # Correct OCR mistakes using the corrections dictionary
    raw_text = [corrections.get(text, text) for text in front_text]

    # Process raw text and extract fields
    for text in raw_text:
        if text in correct_fields:
            temp_key = correct_fields[text]
        elif temp_key:
            data[temp_key] = text
            temp_key = None

    return data


def parse_back_data(raw_text):
    """
    Parse the text extracted from the back part of the ID, ensuring all address lines are merged correctly and misinterpreted fields are handled.
    """
    correct_fields = {
        "ঠিকানা": "address",
        "Address": "address",
        "রক্তের গ্রুপ": "blood_group",
        "Blood Group": "blood_group",
        "Eiond Group": "Blood Group", 
    }

    corrections = {
        "Eiond Group": "Blood Group",  # Fix for OCR misinterpretation
        "Group": "Blood Group",  # Handle common OCR mistakes
        "রক্তের গ্রপ": "রক্তের গ্রুপ",  # Bengali correction for 'Blood Group'
    }

    irrelevant_keywords = [
        "Place of Birtn", "KHULNA", "558 1==", "12 #a-2017", "1<BG03754", 
        "43426<86", "BG0<<<<<<<<<<<8", "HOSSAIN<<M<M<ENAMUL", "Date of Birth"
    ]

    valid_blood_groups = ["A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]

    data = {}
    temp_key = None
    address_lines = []  # To collect address-related text
    blood_group_found = False

    # Preprocess text: Apply corrections and remove unnecessary characters
    raw_text = [corrections.get(text.strip().replace(":", ""), text.strip().replace(":", "")) for text in raw_text if text.strip()]

    # Debug: Print raw text before processing
    logging.debug(f"Raw text (back side): {raw_text}")

    for text in raw_text:
        if text in correct_fields:
            temp_key = correct_fields[text]
            if temp_key == "address":
                address_lines = []  # Start fresh collection for address
        elif temp_key:
            if temp_key == "address":
                # Stop adding address lines if we encounter irrelevant data
                if any(keyword in text for keyword in irrelevant_keywords):
                    logging.debug(f"Stopping address collection due to irrelevant text: {text}")
                    break  # Stop collecting address
                else:
                    address_lines.append(text)
            elif temp_key == "blood_group":
                # Ensure that we are capturing only valid blood group values
                if not blood_group_found and text in valid_blood_groups:
                    data[temp_key] = text
                    blood_group_found = True
                temp_key = None  # Reset temp_key after processing blood group
            elif temp_key == "place_of_birth":
                data[temp_key] = text
                temp_key = None
            elif temp_key == "date_of_birth":
                data[temp_key] = text
                temp_key = None
            else:
                data[temp_key] = text
                temp_key = None  # Reset temp_key after processing

    # Merge and clean up the address field
    if address_lines:
        address_text = " ".join(address_lines).strip()
        # Remove 'Eiond Group H+' from the address if present
        address_text = address_text.replace("Eiond Group H+", "").strip()
        data["address"] = address_text

    # Ensure that Blood Group is correctly separated and identified
    if "address" in data:
        # Check if 'Eiond Group H+' is part of the address and set blood_group accordingly
        if "Eiond Group H+" in data["address"]:
            data["blood_group"] = "H+"
    
    # Debug: Check if address and blood group are extracted correctly
    logging.debug(f"Extracted data: {data}")

    return data




if __name__ == '__main__':
    # Ensure UTF-8 is supported
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    app.run(host='0.0.0.0', port=5000, debug=True)
