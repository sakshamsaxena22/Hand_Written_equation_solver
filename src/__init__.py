"""
Hand-Written Equation Solver Package
====================================

A comprehensive package for recognizing and solving handwritten mathematical equations
using computer vision, deep learning, and symbolic mathematics.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components
from .core.preprocessing import AdvancedImagePreprocessor
from .core.segmentation import MathExpressionSegmenter
from .models.transformer_ocr import MathTransformerOCR
from .core.expression_parser import MathExpressionParser, EquationTypeClassifier
from .solvers.symbolic_solver import SymbolicMathSolver
from .solvers.latex_generator import LaTeXGenerator
from .solvers.step_solver import StepByStepSolver
from .utils.gemini_fallback import GeminiFallbackRecognizer

__all__ = [
    'AdvancedImagePreprocessor',
    'MathExpressionSegmenter', 
    'MathTransformerOCR',
    'MathExpressionParser',
    'EquationTypeClassifier',
    'SymbolicMathSolver',
    'LaTeXGenerator',
    'StepByStepSolver',
    'GeminiFallbackRecognizer'
]
