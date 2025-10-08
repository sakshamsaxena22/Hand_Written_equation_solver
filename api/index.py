"""
Vercel Serverless Function Entry Point
=======================================

This is the single entry point for the Vercel serverless deployment.
It provides a lightweight FastAPI app with the equation solver.

Endpoints:
- GET  /api     - Health check and API info
- POST /api/solve - Solve handwritten equation from uploaded image
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import Optional
import logging
import sys
import os

# Import the lightweight solver
from ._lightweight_solver import LightweightEquationSolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Handwritten Equation Solver API",
    description="Lightweight equation solver for Vercel deployment",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize solver (lightweight version)
try:
    solver = LightweightEquationSolver()
    logger.info("✅ Lightweight solver initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize solver: {e}")
    solver = None


@app.get("/api")
async def root():
    """Root endpoint - API information and health check"""
    return {
        "name": "Handwritten Equation Solver API",
        "version": "1.0.0",
        "status": "healthy" if solver else "degraded",
        "endpoints": {
            "solve": "/api/solve (POST)",
            "health": "/api/health (GET)",
            "docs": "/api/docs",
        },
        "note": "This is a lightweight deployment. For full ML-based OCR, consider integrating external services.",
        "github": "https://github.com/sakshamsaxena22/Hand_Written_equation_solver"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if solver else "unhealthy",
        "solver": "lightweight" if solver else "not initialized",
        "version": "1.0.0"
    }


@app.post("/api/solve")
async def solve_equation(
    file: UploadFile = File(...),
    show_steps: Optional[bool] = Form(True)
):
    """
    Solve handwritten mathematical equation from uploaded image
    
    Args:
        file: Image file (PNG, JPG, etc.) containing handwritten equation
        show_steps: Whether to include step-by-step solution (default: True)
    
    Returns:
        JSON response with:
        - success: bool
        - recognized_expression: str
        - equation_types: list
        - solutions: dict
        - steps: list (if show_steps=True)
        - latex: str
    """
    
    if not solver:
        raise HTTPException(
            status_code=503,
            detail="Solver not initialized. Please try again later."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Read image data
        logger.info(f"Processing uploaded file: {file.filename}")
        image_data = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please upload a valid image file."
            )
        
        logger.info(f"Image decoded successfully: {image.shape}")
        
        # Solve equation
        result = solver.solve_handwritten_equation(
            image,
            show_steps=show_steps
        )
        
        logger.info(f"Equation solved: {result.get('success', False)}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/demo")
async def demo_page():
    """Simple demo page for testing the API"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Equation Solver Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-form {
                margin: 20px 0;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            button {
                background: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background: #45a049;
            }
            #result {
                margin-top: 20px;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 5px;
                display: none;
            }
            .success {
                color: #4CAF50;
            }
            .error {
                color: #f44336;
            }
            pre {
                background: #eee;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧮 Equation Solver Demo</h1>
            <p>Upload an image of a handwritten mathematical equation to solve it.</p>
            
            <div class="upload-form">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" name="file" accept="image/*" required>
                    <br>
                    <label>
                        <input type="checkbox" id="showSteps" checked> Show step-by-step solution
                    </label>
                    <br><br>
                    <button type="submit">Solve Equation</button>
                </form>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                const showSteps = document.getElementById('showSteps').checked;
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image file');
                    return;
                }
                
                // Show loading
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>Processing...</p>';
                
                // Prepare form data
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('show_steps', showSteps);
                
                try {
                    const response = await fetch('/api/solve', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        let html = '<h3 class="success">✅ Success!</h3>';
                        html += '<p><strong>Expression:</strong> ' + result.recognized_expression + '</p>';
                        html += '<p><strong>Type:</strong> ' + result.equation_types.join(', ') + '</p>';
                        html += '<p><strong>Solutions:</strong></p>';
                        html += '<pre>' + JSON.stringify(result.solutions, null, 2) + '</pre>';
                        
                        if (result.steps) {
                            html += '<p><strong>Steps:</strong></p><ol>';
                            result.steps.forEach(step => {
                                html += '<li>' + step + '</li>';
                            });
                            html += '</ol>';
                        }
                        
                        if (result.note) {
                            html += '<p><em>' + result.note + '</em></p>';
                        }
                        
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = '<h3 class="error">❌ Error</h3><p>' + 
                            (result.error || 'Unknown error') + '</p>';
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<h3 class="error">❌ Error</h3><p>' + error.message + '</p>';
                }
            });
        </script>
    </body>
    </html>
    """)


# Vercel serverless function handler
# This is the entry point that Vercel calls
handler = app

