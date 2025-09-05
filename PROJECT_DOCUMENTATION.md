# Hand-Written Equation Solver - Complete Implementation

## 🎯 Project Overview

This is a comprehensive, state-of-the-art hand-written mathematical equation solver that uses deep learning, computer vision, and symbolic mathematics to recognize, parse, and solve handwritten mathematical expressions.

## 🚀 Features Completed

### ✅ Core Components
- **Advanced Image Preprocessing** - Noise reduction, skew correction, contrast enhancement
- **Mathematical Expression Segmentation** - Intelligent symbol detection and segmentation
- **Transformer-based OCR Model** - Deep learning model for symbol recognition
- **Context-Aware Recognition** - Enhanced accuracy using mathematical context
- **Expression Parser** - Converts recognized symbols to mathematical expressions
- **Symbolic Solver** - Solves various equation types using SymPy
- **LaTeX Generator** - Converts expressions to LaTeX format
- **Step-by-Step Solutions** - Detailed solution explanations
- **Web API Interface** - REST API with FastAPI
- **Comprehensive Training Pipeline** - Synthetic data generation and model training

### ✅ Mathematical Capabilities
- **Equation Types Supported:**
  - Linear and algebraic equations
  - Quadratic equations
  - Trigonometric equations
  - Logarithmic and exponential equations
  - Differential equations
  - Integral calculus
  - System of equations
  - Polynomial equations

### ✅ Technical Features
- **222 Mathematical Symbols** - Comprehensive vocabulary including:
  - Numbers (0-9)
  - Basic operations (+, -, ×, ÷, =, etc.)
  - Variables (a-z, A-Z)
  - Greek letters (α, β, γ, π, θ, etc.)
  - Mathematical functions (sin, cos, tan, log, ln, etc.)
  - Calculus symbols (∫, ∑, ∂, √, ∞, etc.)
  - Set theory and logic symbols
  - Fractions and special symbols

## 📁 Project Structure

```
Hand_Written_equation_solver/
├── main.py                          # Main entry point
├── train_model.py                   # Comprehensive training script
├── requirements.txt                 # Dependencies
├── setup.py                        # Package setup
├── README.md                       # Project README
├── PROJECT_DOCUMENTATION.md        # This file
├── 
├── src/                            # Source code
│   ├── __init__.py
│   ├── main.py                     # Core solver implementation
│   ├── 
│   ├── core/                       # Core modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py        # Image preprocessing
│   │   ├── segmentation.py         # Symbol segmentation
│   │   ├── expression_parser.py    # Expression parsing
│   │   ├── equation_classifier.py  # Equation classification
│   │   ├── context_detector.py     # Context detection
│   │   ├── syntax_analyzer.py      # Syntax analysis
│   │   └── vocabulary.py          # Symbol vocabulary
│   │   
│   ├── models/                     # ML models
│   │   ├── __init__.py
│   │   ├── transformer_ocr.py      # Transformer OCR model
│   │   ├── configs/
│   │   │   └── settings.py
│   │   └── pretrained/             # Pre-trained model weights
│   │       ├── ocr_transformer.pth
│   │       ├── attention_weights.pth
│   │       └── symbol_classifier.pth
│   │   
│   ├── solvers/                    # Mathematical solvers
│   │   ├── __init__.py
│   │   ├── symbolic_solver.py      # SymPy-based solver
│   │   ├── step_solver.py          # Step-by-step solutions
│   │   ├── latex_generator.py      # LaTeX generation
│   │   └── verification.py        # Solution verification
│   │   
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── image_utils.py          # Image processing utilities
│       ├── file_utils.py           # File operations
│       ├── math_utils.py           # Math utilities
│       ├── context_recogniser.py   # Context recognition
│       └── logging_config.py       # Logging configuration
│
├── api/                            # Web API
│   ├── __init__.py
│   ├── app.py                      # FastAPI application
│   ├── routes.py                   # API routes
│   ├── models.py                   # API data models
│   └── middleware.py               # API middleware
│
├── config/                         # Configuration
│   ├── __init__.py
│   ├── settings.py                 # Main settings
│   └── model_config.py             # Model configurations
│
├── data/                           # Data files
│   └── vocabularies/               # Symbol vocabularies
│       ├── symbols.json            # Complete symbol vocabulary
│       ├── operators.json          # Mathematical operators
│       └── functions.json          # Mathematical functions
│
├── templates/                      # HTML templates
│   ├── index.html                  # Main interface
│   ├── results.html                # Results display
│   └── solver.html                 # Solver interface
│
├── scripts/                        # Utility scripts
│   ├── setup.py                    # Setup scripts
│   └── train_model.py              # Training scripts
│
├── tests/                          # Test files
│   └── test_solver.py              # Unit tests
│
└── venv/                          # Virtual environment
```

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### 2. Installation Steps

```bash
# Clone the repository
git clone <your-repo-url>
cd Hand_Written_equation_solver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 3. Model Training

#### Train with Synthetic Data Only
```bash
python train_model.py --synthetic --epochs 100 --batch_size 32
```

#### Train with Real Data
```bash
python train_model.py --real_data_path /path/to/real/data --epochs 50
```

#### Train with Mixed Data
```bash
python train_model.py --mixed_training --epochs 100 --batch_size 64
```

#### Train with Configuration File
```bash
python train_model.py --config config/training_config.yaml
```

## 🎮 Usage Examples

### 1. Command Line Interface

```python
from main import UniversalEquationSolver
import cv2

# Initialize solver
solver = UniversalEquationSolver()

# Solve from image file
result = solver.solve_from_file('path/to/equation_image.png', show_steps=True)
print(result)

# Solve from numpy array
image = cv2.imread('equation.png', cv2.IMREAD_GRAYSCALE)
result = solver.solve_handwritten_equation(image, context='physics')
```

### 2. Web API Usage

#### Start the API Server
```bash
cd api
python app.py
```

#### API Endpoints

- **POST /solve** - Solve single equation
- **POST /batch-solve** - Solve multiple equations
- **GET /health** - Health check
- **GET /** - Web interface

#### Example API Call
```python
import requests

# Single equation
files = {'file': open('equation.png', 'rb')}
response = requests.post('http://localhost:5000/solve', files=files)
result = response.json()
```

### 3. Batch Processing

```python
# Process multiple images
image_paths = ['eq1.png', 'eq2.png', 'eq3.png']
results = solver.batch_solve(image_paths, show_steps=True)

for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Expression: {result['recognized_expression']}")
    print(f"Solutions: {result['solutions']}")
```

## 🧠 Model Architecture

### Transformer OCR Model
- **Input Size**: 64×64 grayscale images
- **Architecture**: Vision Transformer + Classification Head
- **Vocabulary Size**: 222 mathematical symbols
- **Features**: 
  - Convolutional feature extraction
  - Multi-head attention mechanism
  - Positional encoding
  - Dropout for regularization

### Training Features
- **Synthetic Data Generation**: Automated generation of mathematical symbols
- **Data Augmentation**: Rotation, scaling, translation, noise addition
- **Mixed Training**: Combines synthetic and real data
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **TensorBoard Integration**: Real-time training monitoring

## 📊 Performance & Metrics

### Training Metrics Tracked
- Training/Validation Loss
- Top-1 Accuracy
- Top-5 Accuracy
- Per-symbol accuracy
- Confusion matrices
- Learning rate schedules

### Expected Performance
- **Symbol Recognition**: >95% accuracy on test set
- **Expression Recognition**: >90% accuracy for clean handwriting
- **Solution Accuracy**: >98% for correctly recognized expressions
- **Processing Time**: <5 seconds per equation

## 🔧 Configuration Options

### Model Configuration
```python
MODEL_CONFIG = {
    'vocab_size': 222,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'dropout': 0.1,
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 100,
    'weight_decay': 0.01,
    'early_stopping_patience': 10,
}
```

## 🌐 Web Interface Features

- **Drag & Drop Upload**: Easy image upload
- **Real-time Processing**: Live equation solving
- **Step-by-Step Display**: Detailed solution steps
- **LaTeX Rendering**: Beautiful mathematical notation
- **Batch Processing**: Multiple equations at once
- **Mobile Responsive**: Works on all devices

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration

2. **Model Loading Errors**
   - Verify model files are in `src/models/pretrained/`
   - Check file permissions

3. **Low Recognition Accuracy**
   - Ensure images are clear and high contrast
   - Try preprocessing images manually
   - Consider retraining with domain-specific data

4. **Memory Issues**
   - Reduce batch size in training
   - Use CPU instead of GPU if memory limited

## 🚀 Advanced Features

### Context-Aware Recognition
The system can detect mathematical context (physics, engineering, pure math, etc.) and adjust recognition accordingly.

### Multi-Symbol Expressions
Handles complex expressions with multiple variables, functions, and operations.

### Verification System
Built-in solution verification by substitution back into original equations.

### Extensible Architecture
Easy to add new mathematical domains, symbol types, and solving methods.

## 📈 Future Enhancements

### Planned Features
- [ ] Handwritten text recognition (not just symbols)
- [ ] Graph and diagram interpretation
- [ ] Advanced calculus (multi-variable)
- [ ] Statistical analysis equations
- [ ] Matrix operations
- [ ] 3D equation visualization

### Integration Possibilities
- Jupyter notebook extension
- Mobile app development
- Integration with math learning platforms
- API for educational software

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Include unit tests for new features

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- SymPy team for symbolic mathematics
- PyTorch team for deep learning framework
- OpenCV team for computer vision tools
- FastAPI team for web framework
- Mathematical community for inspiration

---

## 🎯 Quick Start Summary

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Train model**: `python train_model.py --synthetic --epochs 50`
3. **Start API**: `python api/app.py`
4. **Open browser**: `http://localhost:5000`
5. **Upload equation image and solve!**

This comprehensive system provides everything needed for accurate handwritten mathematical equation recognition and solving. The modular architecture makes it easy to extend and customize for specific use cases.
