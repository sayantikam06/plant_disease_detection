import os
import uuid
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from predict import get_predictor

app = Flask(__name__)

UPLOAD_FOLDER   = os.path.join("static", "uploads")
MAX_FILE_SIZE   = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route("/", methods=["GET"])
def health_check():

    return jsonify({
        "status" : "running",
        "message": "Plant Disease Detection API is live",
        "version": "1.0"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Expects:
        - Multipart form data with key "image" containing the leaf image file

    Returns (JSON):
        Success (200):
            {
                "success"     : true,
                "disease"     : "Tomato_Leaf_Blight",
                "confidence"  : 91.5,
                "all_predictions": [
                    {"disease": "Leaf_Blight", "confidence": 91.5},
                    {"disease": "Healthy",     "confidence": 5.2},
                    ...
                ],
                "filename"    : "abc123.jpg"
            }

        Error (400/500):
            {
                "success": false,
                "error"  : "No image file provided"
            }

    ERROR HANDLING covers:
        - No file in request
        - Empty filename (user submitted without choosing a file)
        - Invalid file extension (not jpg/png/etc)
        - File too large (handled by Flask MAX_CONTENT_LENGTH)
        - Model prediction failure (corrupt image, etc.)
    """


    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error"  : "No image file provided. Send the file with key 'image'."
        }), 400

    file = request.files["image"]


    if file.filename == "":
        return jsonify({
            "success": False,
            "error"  : "No file selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error"  : f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit(".", 1)[1].lower()

    unique_filename = f"{uuid.uuid4().hex}.{extension}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    file.save(save_path)

    try:
        predictor = get_predictor() 
        start_time = time.time()
        result = predictor.predict(save_path)
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        result["success"]      = True
        result["filename"]     = unique_filename
        result["inference_ms"] = elapsed_ms

        return jsonify(result), 200

    except ValueError as ve:

        return jsonify({
            "success": False,
            "error"  : f"Image processing failed: {str(ve)}"
        }), 400

    except Exception as e:

        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error"  : "Internal server error during prediction. Check server logs."
        }), 500

    finally:

        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/classes", methods=["GET"])
def get_classes():
    """
    Returns all disease classes the model can detect.
    Useful for the frontend to display the list of supported diseases.

    Returns (JSON):
        {
            "success": true,
            "classes": ["Healthy", "Leaf_Blight", "Powdery_Mildew", ...],
            "count"  : 3
        }
    """
    try:
        predictor = get_predictor()
        classes = list(predictor.class_labels.values())
        return jsonify({
            "success": True,
            "classes": sorted(classes),
            "count"  : len(classes)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error"  : str(e)
        }), 500

if __name__ == "__main__":
    print("=" * 60)
    print("  Plant Disease Detection — Flask Backend")
    print("=" * 60)
    print("  Initializing ML model (first load may take ~10 seconds)...")

    try:
        get_predictor()
        print("  Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"\n  WARNING: {e}")
        print("  Run 'python train_model.py' to train the model first.")
        print("  Server will start anyway (predictions will fail until model is ready).\n")

    print("\n  Server starting at: http://localhost:5000")
    print("  Press CTRL+C to stop\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
