import os
import re 
import json
import base64
import tempfile
import google.generativeai as genai
import pdfplumber
import docx2txt
import easyocr
from PIL import Image
from flask import Flask, request, jsonify


API_KEY = os.getenv("GEMINI_API_KEY", None) 
if not API_KEY: 
    raise RuntimeError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")  
else: 
    genai.configure(api_key=API_KEY) 
model = genai.GenerativeModel("gemini-2.5-flash")


# OCR Reader
reader = easyocr.Reader(['en', 'hi'])  # English + Hindi support

def extract_text_from_file(upload_file):
    text = ""
    ext = upload_file.lower()
    if ext.endswith(".txt"):
        with open(upload_file, "r", encoding="utf-8") as f:
            text = f.read()

    elif ext.endswith(".pdf"):
        with pdfplumber.open(upload_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    elif ext.endswith(".docx"):
        text = docx2txt.process(upload_file)

    return text.strip()    

# Extract text from image (OCR)
def extract_text_from_image(uploaded_image):
    image = Image.open(uploaded_image)
    results = reader.readtext(image)
    extracted_text = " ".join([res[1] for res in results])
    return extracted_text.strip()


# Gemini-based Image Classification
def classify_image_with_gemini(uploaded_image):
    image = Image.open(uploaded_image)
    prompt = """
    You are an AI that classifies grievance-related images.
    Possible Categories:
    1. infrastructure hazard / road damage
    2. flood / water logging
    3. water leakage / pipe burst
    4. electricity outage / fallen poles
    5. sanitation / garbage / drainage issues

    Output only the best fitting category from the list above.
    """
    response = model.generate_content([prompt, image])
    return response.text.strip()

# Analyze grievance with Gemini API
def analyze_grievance(input_data, input_type="text", category_override=None):
    category_str = ""
    if category_override:
        category_str = f"""
        NOTE: The image was classified as **{category_override}**. 
        Consider this as the complaint category even if no text is available.
        """

    prompt = f"""
    You are an AI assistant for government grievance management. 
    Analyze the citizen complaint below and return a structured JSON **exactly** in the following format:
    
{{
  "risk_level": "low/medium/high",
  "urgency_score": 1-10,
  "category": "short descriptive category of the issue like public safety / health issue / water supply disruption / electricity outage / sanitation problem / other short category ",
  "recommended_action": "clear, immediate, and strong action focusing on protecting citizens and restoring essential services",
  "escalation_needed": true/false
}}

    
    Rules:
    1. Use numeric values (1-10) for urgency_score, where 10 is most urgent.
    2. risk_level should be a short description: "low risk", "medium risk", or "high risk".
    3. category should describe the problem clearly in a few words (e.g., "water supply disruption").
    4. recommended_action should be actionable and concise.
    5. escalation_needed is true if the issue requires immediate attention from higher authorities, otherwise false.
    6. Output **only JSON** — no explanations, no extra text.
    
    {category_str}

    Complaint:
    \"\"\"{input_data}\"\"\"

    """

    if input_type == "image":
        # For image input (Gemini API can take image object)
        response = model.generate_content([prompt, input_data])
    else:
        response = model.generate_content(prompt + f"\nGrievance: {input_data}")

    return response.text

# ===== Firebase Cloud Function =====
def redtape_reducer(request):
    """
    Cloud Function Entry Point
    Input (POST):
    {
      "text": "complaint text",
      "file": "base64 encoded file",
      "file_type": "pdf/docx/jpg/png"
    }
    """
    if request.method != "POST":
        return jsonify({"error": "Only POST allowed"}), 405

    try:
        grievance_text = ""
        category_override = None

        # Case 1: JSON Input
        if request.is_json:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Invalid JSON"}), 400

            if "text" in data and data["text"].strip():
                grievance_text = data["text"]

             # Base64-encoded file
            elif "file" in data and "file_type" in data:
                file_bytes = base64.b64decode(data["file"])
                file_ext = data["file_type"].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                if file_ext in ["pdf", "docx", "doc", "txt"]:
                    grievance_text = extract_text_from_file(tmp_path)
                elif file_ext in ["jpg", "jpeg", "png"]:
                    grievance_text = extract_text_from_image(tmp_path)
                    if not grievance_text.strip():
                        category_override = classify_image_with_gemini(tmp_path)
                        grievance_text = "No text provided"
                os.remove(tmp_path)

        # ===== Case 2: Multipart/form-data file upload =====
        elif request.files:
            uploaded_file = request.files.get("file")
            if uploaded_file:
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]).name
                uploaded_file.save(tmp_path)
                ext = uploaded_file.filename.split(".")[-1].lower()
                if ext in ["pdf", "docx", "txt"]:
                    grievance_text = extract_text_from_file(tmp_path)
                elif ext in ["jpg", "jpeg", "png"]:
                    grievance_text = extract_text_from_image(tmp_path)
                    if not grievance_text.strip():
                        category_override = classify_image_with_gemini(tmp_path)
                        grievance_text = "No text provided"
                os.remove(tmp_path)        

        # Case 3: Plain text body
        if not grievance_text:
            try:
                raw_text = request.data.decode("utf-8").strip()
                if raw_text:
                    grievance_text = raw_text
            except Exception:
                pass

        if not grievance_text:
            return jsonify({"error": "No text or file provided"}), 400

        # Analyze grievance
        result = analyze_grievance(grievance_text, "text", category_override)

         # ===== Clean Markdown code blocks from Gemini output =====
        clean_result = re.sub(r"```(json)?\s*([\s\S]*?)```", r"\2", result).strip()

        try:
            parsed = json.loads(clean_result)
            return jsonify(parsed)
        except Exception as e:
            return jsonify({"error": "Failed to parse JSON", "raw_response": clean_result, "exception": str(e)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

