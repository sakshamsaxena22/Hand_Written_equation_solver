# 🧮 Hand-Written Equation Solver

An advanced AI-powered system for recognizing and solving handwritten mathematical equations using computer vision, deep learning, and symbolic mathematics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

## 🌟 Features

### 🎯 **Core Capabilities**
- **Advanced OCR**: Transformer-based deep learning model for mathematical symbol recognition
- **Smart Image Processing**: Multi-stage preprocessing with noise reduction, skew correction, and contrast enhancement  
- **Symbolic Mathematics**: Powered by SymPy for accurate equation solving
- **AI Fallback**: Google Gemini integration for difficult recognition cases
- **Step-by-Step Solutions**: Detailed mathematical explanations and derivations

### 🔧 **Supported Mathematics**
- ✅ **Linear Equations**: `2x + 5 = 13`
- ✅ **Quadratic Equations**: `x² - 4x + 4 = 0`
- ✅ **Polynomial Equations**: `x³ + 2x² - x - 2 = 0`
- ✅ **Trigonometric Equations**: `sin(x) + cos(x) = 1`
- ✅ **Calculus Operations**: Derivatives, integrals, limits
- ✅ **System of Equations**: Multiple variable solving

### 🌐 **Web Interface**
- **Modern UI**: Clean, responsive design with drag-and-drop upload
- **Real-time Processing**: Live equation solving with progress indicators
- **Mathematical Formatting**: LaTeX-rendered equations and solutions
- **Error Handling**: Comprehensive validation and user feedback

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sakshamsaxena22/Hand_Written_equation_solver.git
   cd Hand_Written_equation_solver
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration**
   ```bash
   # Copy and edit environment variables
   cp .env.example .env
   # Add your Gemini API key (optional)
   ```

4. **Run the demo**
   ```bash
   python demo_solver.py
   ```

### 🌐 Web Interface

Start the web application:
```bash
python web_app.py
```

Visit http://localhost:5000 to access the interactive interface.

## 📁 Project Structure

```
Hand_Written_equation_solver/
├── 🎯 Core Scripts
│   ├── main.py                    # Main equation solver orchestrator
│   ├── web_app.py                 # Flask web interface  
│   ├── demo_solver.py             # Mathematical solving demo
│   ├── train_model.py             # Model training pipeline
│   └── test_system.py             # System verification tests
├── 📁 src/                        # Source code modules
│   ├── core/                      # Core processing
│   │   ├── preprocessing.py       # Advanced image preprocessing
│   │   ├── segmentation.py        # Mathematical expression segmentation
│   │   └── expression_parser.py   # Expression parsing & classification
│   ├── models/                    # Machine learning models
│   │   └── transformer_ocr.py     # Transformer OCR model
│   ├── solvers/                   # Mathematical solvers
│   │   ├── symbolic_solver.py     # SymPy-based equation solver
│   │   ├── latex_generator.py     # LaTeX output generation
│   │   └── step_solver.py         # Step-by-step solutions
│   └── utils/                     # Utility modules
│       └── gemini_fallback.py     # Google Gemini AI integration
├── 📁 test_images/               # Sample test images
├── 📄 requirements.txt           # Python dependencies
├── 📄 .env                       # Configuration file
└── 📄 README.md                  # Project documentation
```

## 🛠️ Usage Examples

### Command Line Interface

```python
from main import UniversalEquationSolver
import cv2

# Initialize solver
solver = UniversalEquationSolver()

# Solve from image file
result = solver.solve_from_file('path/to/equation.jpg')
print(f"Solution: {result['solutions']}")
print(f"Steps: {result['steps']}")
```

### Web API

```bash
# Upload and solve equation via web interface
curl -X POST -F "file=@equation.jpg" http://localhost:5000/solve
```

### Mathematical Solving Demo

```bash
# Run comprehensive demo
python demo_solver.py

# Output:
# Solving: 2*x + 5 = 13
#   Variable: x  
#   Solution: ['4']
# 
# Solving: x**2 - 4 = 0
#   Variable: x
#   Solution: ['-2', '2']
```

## 🧠 Technical Architecture

### **Pipeline Overview**
```
Image Input → Preprocessing → Segmentation → OCR Recognition
     ↓                                            ↓
Web Interface ← Solution Display ← Symbolic Solving ← Expression Parsing
     ↓                                            ↑
Upload Handler → File Management → Pipeline Orchestrator
```

### **Core Components**
- **Image Processing**: OpenCV + scikit-image for advanced preprocessing
- **Deep Learning**: PyTorch transformer model for symbol recognition
- **Symbolic Math**: SymPy for mathematical computation and solving
- **Web Framework**: Flask for REST API and web interface
- **AI Integration**: Google Gemini for fallback recognition

### **Key Technologies**
- **Computer Vision**: OpenCV, Albumentations, scikit-image
- **Machine Learning**: PyTorch, NumPy, SciPy
- **Mathematical Computing**: SymPy, matplotlib
- **Web Development**: Flask, HTML5, JavaScript
- **External APIs**: Google Generative AI

## 📊 Performance & Accuracy

### **Recognition Accuracy**
- **Printed Mathematical Text**: ~95%+ accuracy
- **Clear Handwriting**: ~85-90% accuracy  
- **Complex Expressions**: ~80% accuracy with Gemini fallback

### **Supported Symbol Set**
- **222 Mathematical Symbols**: Numbers, operators, functions, Greek letters
- **Special Characters**: Integrals, summations, radicals, subscripts/superscripts
- **Variables & Constants**: Single/multi-character variables, mathematical constants

## 🔧 Advanced Configuration

### Environment Variables (.env)
```bash
# Gemini AI (Optional)
GEMINI_API_KEY=your_api_key_here

# Model Configuration  
LOCAL_MODEL_PATH=models/math_transformer.pth
CONFIDENCE_THRESHOLD=0.7

# Performance
USE_GPU=true
BATCH_SIZE=32

# Web Interface
FLASK_DEBUG=false
PORT=5000
```

### Training Custom Models
```bash
# Train with synthetic data
python train_model.py --synthetic --epochs 100 --batch_size 32

# Train with real data
python train_model.py --real_data_path /path/to/data --epochs 50

# Mixed training approach  
python train_model.py --mixed_training --synthetic --real_data_path /path/to/data
```

## 🧪 Testing & Validation

### Run System Tests
```bash
# Comprehensive system verification
python test_system.py

# Generate test images
python create_test_images.py

# Verify specific components
python -c "from src.core.preprocessing import AdvancedImagePreprocessor; print('✓ Image processing ready')"
```

### Test Results
```
✅ Mathematical Solver: Working
✅ Image Processing: Working  
✅ Web Interface: Working
✅ OCR Recognition: Working
✅ Gemini Integration: Available
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SymPy Community** for excellent symbolic mathematics library
- **PyTorch Team** for deep learning framework
- **OpenCV Contributors** for computer vision tools
- **Google** for Gemini AI API
- **Flask Community** for web framework

## 🔮 Future Enhancements

- [ ] **Mobile App**: React Native mobile application
- [ ] **Real-time Recognition**: Live camera-based equation solving
- [ ] **3D Math Support**: Geometry and vector mathematics
- [ ] **Collaborative Features**: Multi-user equation sharing
- [ ] **Enhanced Training**: More diverse handwriting datasets
- [ ] **API Marketplace**: Public API for third-party integration

## 📞 Contact & Support

**Author**: Saksham Saxena  
**GitHub**: [@sakshamsaxena22](https://github.com/sakshamsaxena22)  
**Project**: [Hand_Written_equation_solver](https://github.com/sakshamsaxena22/Hand_Written_equation_solver)

For issues and questions, please use the [GitHub Issues](https://github.com/sakshamsaxena22/Hand_Written_equation_solver/issues) page.

---

⭐ **Star this repository if you find it helpful!**
