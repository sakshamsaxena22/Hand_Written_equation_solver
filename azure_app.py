"""
Azure App Service Entry Point
==============================

Full-featured FastAPI application with PyTorch ML model support.
This version uses the complete UniversalEquationSolver with all ML capabilities.

Endpoints:
- GET  /           - Root info
- GET  /health     - Health check
- POST /solve      - Solve handwritten equation from uploaded image
- GET  /demo       - Interactive demo page
- GET  /docs       - OpenAPI documentation
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the full solver with ML models
try:
    from main import UniversalEquationSolver
    USE_FULL_SOLVER = True
except Exception as e:
    print(f"Warning: Could not import full solver: {e}")
    print("Falling back to lightweight solver")
    from api._lightweight_solver import LightweightEquationSolver
    USE_FULL_SOLVER = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log') if os.path.exists('logs') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Handwritten Equation Solver API - Azure",
    description="Full-featured equation solver with PyTorch ML models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize solver
solver = None
solver_type = "none"

try:
    if USE_FULL_SOLVER:
        logger.info("Initializing full ML solver with PyTorch...")
        solver = UniversalEquationSolver()
        solver_type = "full_ml"
        logger.info("✅ Full ML solver initialized successfully")
    else:
        logger.info("Initializing lightweight solver...")
        solver = LightweightEquationSolver()
        solver_type = "lightweight"
        logger.info("✅ Lightweight solver initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize solver: {e}", exc_info=True)
    solver = None
    solver_type = "failed"


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Handwritten Equation Solver API",
        "version": "2.0.0",
        "platform": "Azure App Service",
        "status": "healthy" if solver else "degraded",
        "solver_type": solver_type,
        "features": {
            "ml_ocr": USE_FULL_SOLVER,
            "pytorch": USE_FULL_SOLVER,
            "symbolic_solving": True,
            "step_by_step": True,
            "latex_output": True
        },
        "endpoints": {
            "solve": "/solve (POST)",
            "health": "/health (GET)",
            "demo": "/demo (GET)",
            "docs": "/docs (GET)",
        },
        "github": "https://github.com/sakshamsaxena22/Hand_Written_equation_solver"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Azure monitoring"""
    is_healthy = solver is not None
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "solver": solver_type,
        "ml_models": USE_FULL_SOLVER,
        "version": "2.0.0",
        "platform": "azure"
    }


@app.post("/solve")
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
            detail="Solver not initialized. Please check server logs."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Read image data
        logger.info(f"Processing uploaded file: {file.filename} (solver: {solver_type})")
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
        
        # Add metadata
        result['solver_used'] = solver_type
        result['ml_powered'] = USE_FULL_SOLVER
        
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


@app.get("/demo")
async def demo_page():
    """Interactive demo page for testing the API"""
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Equation Solver - Azure Demo</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 10px;
            }}
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                background: {'#4CAF50' if USE_FULL_SOLVER else '#FF9800'};
                color: white;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }}
            .info {{
                text-align: center;
                margin-bottom: 20px;
                color: #666;
            }}
            .upload-form {{
                margin: 20px 0;
            }}
            input[type="file"] {{
                margin: 10px 0;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                width: 100%;
            }}
            button {{
                background: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
            }}
            button:hover {{
                background: #45a049;
            }}
            #result {{
                margin-top: 20px;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 5px;
                display: none;
            }}
            .success {{
                color: #4CAF50;
            }}
            .error {{
                color: #f44336;
            }}
            pre {{
                background: #eee;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            .step {{
                padding: 10px;
                margin: 5px 0;
                background: white;
                border-left: 4px solid #4CAF50;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧮 Hand-Written Equation Solver</h1>
            <div class="info">
                <span class="badge">{'ML-POWERED (PyTorch)' if USE_FULL_SOLVER else 'LIGHTWEIGHT MODE'}</span>
                <p>Deployed on Azure App Service</p>
            </div>
            
            <div class="upload-form">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" name="file" accept="image/*" required>
                    <br>
                    <label>
                        <input type="checkbox" id="showSteps" checked> Show step-by-step solution
                    </label>
                    <br><br>
                    <button type="submit">🔍 Solve Equation</button>
                </form>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                const showSteps = document.getElementById('showSteps').checked;
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {{
                    alert('Please select an image file');
                    return;
                }}
                
                // Show loading
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>⏳ Processing with {'ML models' if USE_FULL_SOLVER else 'lightweight solver'}...</p>';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('show_steps', showSteps);
                
                try {{
                    const response = await fetch('/solve', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const result = await response.json();
                    
                    if (result.success) {{
                        let html = '<h3 class="success">✅ Success!</h3>';
                        html += '<p><strong>Solver:</strong> ' + (result.solver_used || 'unknown') + '</p>';
                        html += '<p><strong>Expression:</strong> ' + result.recognized_expression + '</p>';
                        html += '<p><strong>Type:</strong> ' + result.equation_types.join(', ') + '</p>';
                        html += '<p><strong>Solutions:</strong></p>';
                        html += '<pre>' + JSON.stringify(result.solutions, null, 2) + '</pre>';
                        
                        if (result.steps && result.steps.length > 0) {{
                            html += '<h4>Step-by-Step Solution:</h4>';
                            result.steps.forEach((step, i) => {{
                                html += '<div class="step">Step ' + (i+1) + ': ' + step + '</div>';
                            }});
                        }}
                        
                        if (result.latex) {{
                            html += '<p><strong>LaTeX:</strong></p><pre>' + result.latex + '</pre>';
                        }}
                        
                        if (result.note) {{
                            html += '<p><em>' + result.note + '</em></p>';
                        }}
                        
                        resultDiv.innerHTML = html;
                    }} else {{
                        resultDiv.innerHTML = '<h3 class="error">❌ Error</h3><p>' + 
                            (result.error || 'Unknown error') + '</p>';
                    }}
                }} catch (error) {{
                    resultDiv.innerHTML = '<h3 class="error">❌ Error</h3><p>' + error.message + '</p>';
                }}
            }});
        </script>
    </body>
    </html>
    """)


# For Azure App Service compatibility
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
