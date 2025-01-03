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
    Parse the text extracted from the back part of the ID, ensuring all address lines are merged correctly
    and misinterpreted fields are handled.
    """
    # Field mapping to standard labels
    correct_fields = {
        "ঠিকানা": "address",
        "Address": "address",
        "Blood Group": "blood_group",
        "Eiond Group": "blood_group",
    }

    # OCR corrections for common misinterpretations
    corrections = {
        "Eiond Group": "Blood Group",
        "Group": "Blood Group",
        "ঠিকানা": "Address",
    }

    # Keywords to exclude from processing
    irrelevant_keywords = [
        "Place of Birtn", "KHULNA", "558 1=:=<", "12 #a:-2017", "1<BG03754",
        "43426<86", "BG0<<<<<<<<<<<8", "HOSSAIN<<M<M<ENAMUL", "Date of Birth"
    ]

    # List of valid blood groups
    valid_blood_groups = ["A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-", "H+"]

    # Initialize results and temporary variables
    data = {}
    temp_key = None
    address_lines = []
    blood_group_found = False

    # Correct OCR mistakes
    corrected_text = [corrections.get(text, text) for text in raw_text]
    
    print('corrected_text:',corrected_text)

    # Process raw text and extract fields
    for text in corrected_text:
        # Skip irrelevant lines
        if any(keyword in text for keyword in irrelevant_keywords):
            continue

        # Match recognized fields
        if text.rstrip(":") in correct_fields:
            temp_key = correct_fields[text.rstrip(":")]
            print('text:',text)
        elif temp_key == "address":  # Collect multiline address data
            if "Eiond Group" in text or text.strip() in valid_blood_groups:
                # Extract blood group if present in the address block
                blood_group = text.split(":")[-1].strip()
                if blood_group in valid_blood_groups:
                    data["blood_group"] = blood_group
                    blood_group_found = True
                    logging.debug(f"Blood Group Found: {blood_group}")
            else:
                address_lines.append(text)
                logging.debug(f"Address Line: {text}")
        elif temp_key == "blood_group" and not blood_group_found:
            # Validate and set blood group
            if text in valid_blood_groups:
                data["blood_group"] = text
                logging.debug(f"Blood Group Set: {text}")
            temp_key = None  
        else:
            temp_key = None

    # Combine address lines into a single string
    if address_lines:
        # Clean up the combined address by removing irrelevant substrings
        address = ", ".join(address_lines).replace(";", ",").strip(", ")
        for keyword in irrelevant_keywords + ["Eiond Group: H+", "558 1=:="]:
            address = address.replace(keyword, "").strip(", ")
        data["address"] = address

    return data


  



if __name__ == '__main__':
    # Ensure UTF-8 is supported
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    app.run(host='0.0.0.0', port=5000, debug=True)
