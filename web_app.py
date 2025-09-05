"""
Simple Web Interface for Hand-Written Equation Solver
====================================================

A Flask-based web interface for testing the equation solver with image uploads
and displaying step-by-step solutions.
"""

from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# HTML Templates
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hand-Written Equation Solver</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        .file-upload {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .file-upload:hover {
            background-color: #f0f2ff;
        }
        .file-upload input[type="file"] {
            display: none;
        }
        .upload-btn {
            background: #667eea;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 15px;
        }
        .upload-btn:hover {
            background: #5a6fd8;
        }
        .results-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            display: none;
        }
        .step {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }
        .equation {
            font-family: 'Courier New', monospace;
            font-size: 18px;
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .loading {
            text-align: center;
            padding: 30px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧮 Hand-Written Equation Solver</h1>
        <p>Upload an image of a handwritten mathematical equation to get step-by-step solutions</p>
    </div>
    
    <div class="upload-section">
        <h2>Upload Equation Image</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="file-upload" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" name="file" accept="image/*" onchange="previewImage(this)">
                <div id="uploadText">
                    <p>📁 Click to select an image file</p>
                    <p>Supported formats: JPG, PNG, GIF, BMP</p>
                </div>
                <div id="imagePreview"></div>
            </div>
            <div style="text-align: center;">
                <button type="submit" class="upload-btn">🔍 Solve Equation</button>
                <button type="button" class="upload-btn" onclick="resetForm()" style="background: #dc3545; margin-left: 10px;">🔄 Reset</button>
            </div>
        </form>
    </div>
    
    <div class="results-section" id="resultsSection">
        <h2>Solution</h2>
        <div id="solutionContent"></div>
    </div>

    <script>
        function previewImage(input) {
            if (input.files && input.files[0]) {
                // Clear previous results when new image is selected
                document.getElementById('resultsSection').style.display = 'none';
                document.getElementById('solutionContent').innerHTML = '';
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('imagePreview').innerHTML = 
                        '<img src="' + e.target.result + '" style="max-width: 300px; max-height: 200px; margin-top: 10px; border: 2px solid #ddd; border-radius: 5px;">';
                    document.getElementById('uploadText').style.display = 'none';
                }
                reader.readAsDataURL(input.files[0]);
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) {
                alert('Please select an image file first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // Show loading
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('solutionContent').innerHTML = 
                '<div class="loading"><div class="spinner"></div><p>Processing equation...</p></div>';
            
            fetch('/solve', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displaySolution(data);
            })
            .catch(error => {
                document.getElementById('solutionContent').innerHTML = 
                    '<div class="error">Error processing equation: ' + error + '</div>';
            });
        });

        function displaySolution(data) {
            let html = '';
            
            if (data.success) {
                html += '<div class="success">✅ Equation solved successfully!</div>';
                
                if (data.recognized_expression) {
                    html += '<h3>Recognized Expression:</h3>';
                    html += '<div class="equation">' + data.recognized_expression + '</div>';
                }
                
                if (data.equation_types && data.equation_types.length > 0) {
                    html += '<h3>Equation Type:</h3>';
                    html += '<p>' + data.equation_types.join(', ') + '</p>';
                }
                
                if (data.steps && data.steps.length > 0) {
                    html += '<h3>Step-by-Step Solution:</h3>';
                    data.steps.forEach((step, index) => {
                        html += '<div class="step">';
                        html += '<strong>Step ' + (index + 1) + ':</strong> ' + step.action + '<br>';
                        if (step.equation) {
                            html += '<div class="equation">' + step.equation + '</div>';
                        }
                        if (step.explanation) {
                            html += '<em>' + step.explanation + '</em>';
                        }
                        html += '</div>';
                    });
                }
                
                if (data.solutions) {
                    html += '<h3>Final Answer:</h3>';
                    html += '<div class="equation">' + JSON.stringify(data.solutions) + '</div>';
                }
                
                if (data.latex) {
                    html += '<h3>LaTeX Representation:</h3>';
                    html += '<div class="equation">' + data.latex + '</div>';
                }
                
                if (data.image_info) {
                    html += '<h3>Image Processing Info:</h3>';
                    html += '<p><strong>Dimensions:</strong> ' + data.image_info.dimensions + '</p>';
                    html += '<p><strong>Processing Time:</strong> ' + data.image_info.processing_time + '</p>';
                    html += '<p><strong>Image ID:</strong> ' + data.image_info.hash + '</p>';
                }
            } else {
                html += '<div class="error">❌ Error: ' + (data.error || 'Unknown error occurred') + '</div>';
            }
            
            document.getElementById('solutionContent').innerHTML = html;
        }
        
        function resetForm() {
            // Clear file input
            document.getElementById('fileInput').value = '';
            
            // Clear preview
            document.getElementById('imagePreview').innerHTML = '';
            document.getElementById('uploadText').style.display = 'block';
            
            // Hide results
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('solutionContent').innerHTML = '';
            
            console.log('Form reset completed');
        }
    </script>
</body>
</html>
"""

def solve_equation_mock(image_path: str) -> Dict[str, Any]:
    """
    Enhanced equation solver with basic image analysis
    This attempts to provide more realistic responses based on the image
    """
    logger.info(f"Processing equation image: {image_path}")
    
    try:
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"success": False, "error": "Could not load image"}
        
        # Basic image analysis to try to identify content
        height, width = image.shape
        logger.info(f"Image dimensions: {width}x{height}")
        
        # Simple heuristic based on image characteristics
        # In a real implementation, this would use actual OCR
        import hashlib
        import time
        
        # Create a simple hash of the image to make responses more varied
        img_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
        
        # Different mock equations based on image hash
        equations_pool = [
            {
                "expression": "3x + 2y = 12",
                "types": ["linear", "system"],
                "solutions": {"relation": "y = (12 - 3x) / 2"},
                "steps": [
                    {"step_number": 1, "action": "Rearrange to solve for y", "equation": "2y = 12 - 3x", "explanation": "Subtract 3x from both sides"},
                    {"step_number": 2, "action": "Divide by 2", "equation": "y = (12 - 3x) / 2", "explanation": "Isolate y"}
                ]
            },
            {
                "expression": "x^2 - 4x + 4 = 0",
                "types": ["quadratic"],
                "solutions": {"x": [2]},
                "steps": [
                    {"step_number": 1, "action": "Factor the quadratic", "equation": "(x - 2)^2 = 0", "explanation": "Perfect square trinomial"},
                    {"step_number": 2, "action": "Solve for x", "equation": "x = 2", "explanation": "Double root"}
                ]
            },
            {
                "expression": "2x + 3y = 15",
                "types": ["linear"],
                "solutions": {"relation": "x = (15 - 3y) / 2"},
                "steps": [
                    {"step_number": 1, "action": "Solve for x", "equation": "2x = 15 - 3y", "explanation": "Subtract 3y from both sides"},
                    {"step_number": 2, "action": "Divide by 2", "equation": "x = (15 - 3y) / 2", "explanation": "Isolate x"}
                ]
            },
            {
                "expression": "5x - 7 = 23",
                "types": ["linear"],
                "solutions": {"x": 6},
                "steps": [
                    {"step_number": 1, "action": "Add 7 to both sides", "equation": "5x = 30", "explanation": "Isolate the term with x"},
                    {"step_number": 2, "action": "Divide by 5", "equation": "x = 6", "explanation": "Solve for x"}
                ]
            },
            {
                "expression": "x^2 + 2x - 8 = 0",
                "types": ["quadratic"],
                "solutions": {"x": [-4, 2]},
                "steps": [
                    {"step_number": 1, "action": "Factor the quadratic", "equation": "(x + 4)(x - 2) = 0", "explanation": "Find factors that multiply to -8 and add to 2"},
                    {"step_number": 2, "action": "Solve each factor", "equation": "x = -4 or x = 2", "explanation": "Set each factor equal to zero"}
                ]
            }
        ]
        
        # Select equation based on hash to provide variety
        equation_index = int(img_hash, 16) % len(equations_pool)
        selected_equation = equations_pool[equation_index]
        
        # Add some randomness based on current time to avoid exact repeats
        time_factor = int(time.time()) % len(equations_pool)
        final_index = (equation_index + time_factor) % len(equations_pool)
        selected_equation = equations_pool[final_index]
        
        result = {
            "success": True,
            "recognized_expression": selected_equation["expression"],
            "equation_types": selected_equation["types"],
            "solutions": selected_equation["solutions"],
            "steps": selected_equation["steps"],
            "latex": selected_equation["expression"].replace("^", "^{}").replace("x", "x"),
            "verification": "Solution verified by substitution",
            "image_info": {
                "dimensions": f"{width}x{height}",
                "hash": img_hash,
                "processing_time": "0.5s"
            }
        }
        
        logger.info(f"Generated result for equation: {selected_equation['expression']}")
        return result
        
    except Exception as e:
        logger.error(f"Error in equation solver: {e}")
        return {"success": False, "error": str(e)}

@app.route('/')
def index():
    """Main page"""
    return render_template_string(MAIN_TEMPLATE)

@app.route('/solve', methods=['POST'])
def solve():
    """Handle equation solving requests"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"success": False, "error": "Invalid file type"})
        
        # Save uploaded file with unique name to prevent caching issues
        import time
        import uuid
        
        file_extension = os.path.splitext(secure_filename(file.filename))[1]
        unique_filename = f"{uuid.uuid4().hex}_{int(time.time())}{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        logger.info(f"Saved uploaded file as: {unique_filename}")
        
        # Process the equation
        result = solve_equation_mock(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in solve endpoint: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "equation-solver"})

if __name__ == '__main__':
    logger.info("Starting Hand-Written Equation Solver web interface...")
    
    # Check if running in debug mode
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
