# File: api/app.py
"""
FastAPI Web Application
"""

import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from typing import List
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import UniversalEquationSolver

logger = logging.getLogger(__name__)

# Initialize the solver
solver = UniversalEquationSolver()

# FastAPI application
app = FastAPI(
    title="Universal Mathematical Equation Solver",
    description="Advanced handwritten equation solver with computer vision and symbolic math",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Universal Math Equation Solver</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-area { border: 2px dashed #ccc; padding: 50px; margin: 20px 0; border-radius: 10px; }
            .result { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; text-align: left; }
            .latex { font-size: 18px; color: #333; }
            .steps { margin: 10px 0; }
            .step { padding: 5px 0; border-bottom: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧮 Universal Mathematical Equation Solver</h1>
            <p>Upload an image of a handwritten mathematical equation to solve it!</p>
            
            <form id="upload-form" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" id="image-input" name="file" accept="image/*" required>
                    <p>Choose an image file or drag and drop here</p>
                </div>
                
                <div>
                    <label>
                        <input type="checkbox" id="show-steps" checked> Show step-by-step solution
                    </label>
                </div>
                
                <button type="submit">Solve Equation</button>
            </form>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('upload-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('image-input');
                const showSteps = document.getElementById('show-steps').checked;
                
                formData.append('file', fileInput.files[0]);
                formData.append('show_steps', showSteps);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Processing... Please wait.</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/solve', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            });
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                
                if (result.success) {
                    let html = `
                        <h3>Recognition Result:</h3>
                        <p><strong>Recognized Expression:</strong> ${result.recognized_expression}</p>
                        <p><strong>Equation Type(s):</strong> ${result.equation_types.join(', ')}</p>
                    `;
                    
                    if (result.latex && result.latex.equation) {
                        html += `<div class="latex"><strong>LaTeX:</strong> ${result.latex.equation}</div>`;
                    }
                    
                    if (result.solutions) {
                        html += '<h3>Solutions:</h3>';
                        for (const [type, solution] of Object.entries(result.solutions)) {
                            html += `<p><strong>${type}:</strong> ${JSON.stringify(solution)}</p>`;
                        }
                    }
                    
                    if (result.steps) {
                        html += '<h3>Solution Steps:</h3><div class="steps">';
                        result.steps.forEach(step => {
                            html += `<div class="step">${step}</div>`;
                        });
                        html += '</div>';
                    }
                    
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/solve")
async def solve_equation(file: UploadFile = File(...), show_steps: bool = True):
    """Solve handwritten equation from uploaded image"""
    
    try:
        # Read and decode image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Solve equation
        result = solver.solve_handwritten_equation(image, show_steps=show_steps)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in solve endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-solve")
async def batch_solve_equations(files: List[UploadFile] = File(...)):
    """Solve multiple equations in batch"""
    
    results = []
    
    for file in files:
        try:
            image_data = await file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                result = solver.solve_handwritten_equation(image)
                result['filename'] = file.filename
                results.append(result)
            else:
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid image file',
                    'success': False
                })
                
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'success': False
            })
    
    return JSONResponse(content={'results': results})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}