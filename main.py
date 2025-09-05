 
"""
Main Universal Equation Solver System
"""

import numpy as np
import cv2
from typing import Optional, Dict, Any, List
import logging

# Import all components
from src.core.preprocessing import AdvancedImagePreprocessor
from src.core.segmentation import MathExpressionSegmenter
from src.models.transformer_ocr import MathTransformerOCR
from src.core.expression_parser import MathExpressionParser, EquationTypeClassifier
from src.solvers.symbolic_solver import SymbolicMathSolver
from src.solvers.latex_generator import LaTeXGenerator
from src.solvers.step_solver import StepByStepSolver
from src.utils.context_recognizer import ContextAwareRecognizer

logger = logging.getLogger(__name__)

class UniversalEquationSolver:
    """Complete handwritten mathematical equation solver system"""
    
    def __init__(self):
        # Initialize all components
        self.preprocessor = AdvancedImagePreprocessor()
        self.segmenter = MathExpressionSegmenter()
        self.ocr_model = MathTransformerOCR()
        self.parser = MathExpressionParser()
        self.classifier = EquationTypeClassifier()
        self.symbolic_solver = SymbolicMathSolver()
        self.latex_generator = LaTeXGenerator()
        self.step_solver = StepByStepSolver(self.symbolic_solver)
        self.context_recognizer = ContextAwareRecognizer()
        
        logger.info("Universal Equation Solver initialized successfully")
    
    def solve_handwritten_equation(self, 
                                 image: np.ndarray, 
                                 context: Optional[str] = None,
                                 show_steps: bool = True) -> Dict[str, Any]:
        """Complete pipeline to solve handwritten mathematical equation"""
        
        try:
            # Phase 1: Image preprocessing and OCR
            logger.info("Phase 1: Processing image...")
            preprocessed = self.preprocessor.preprocess(image)
            
            # Segment mathematical expression
            symbols_data = self.segmenter.segment_expression(preprocessed)
            
            # Recognize symbols using transformer OCR
            recognized_symbols = []
            for symbol_data in symbols_data:
                symbol = self.ocr_model.predict_symbol(symbol_data['image'])
                recognized_symbols.append(symbol)
            
            # Context-aware enhancement
            if context:
                recognized_symbols = self.context_recognizer.enhance_recognition(
                    recognized_symbols, context
                )
            
            logger.info(f"Recognized symbols: {recognized_symbols}")
            
            # Phase 2: Expression parsing
            logger.info("Phase 2: Parsing expression...")
            tokens = self.parser.tokenize(recognized_symbols)
            expression_tree = self.parser.parse_to_tree(tokens)
            
            # Classify equation type
            expression_string = ' '.join(recognized_symbols)
            equation_types = self.classifier.classify(expression_string)
            
            logger.info(f"Equation types: {equation_types}")
            
            # Phase 3: Symbolic solving
            logger.info("Phase 3: Solving equation...")
            if show_steps:
                results = self.step_solver.solve_with_steps(expression_tree, equation_types)
            else:
                solutions = self.symbolic_solver.solve_equation(expression_tree, equation_types)
                results = {'solutions': solutions}
            
            # Phase 4: Generate LaTeX output
            logger.info("Phase 4: Generating LaTeX...")
            latex_output = self.latex_generator.generate_latex(
                expression_tree, results.get('solutions', {})
            )
            
            # Compile final result
            final_result = {
                'recognized_expression': expression_string,
                'equation_types': equation_types,
                'solutions': results.get('solutions', {}),
                'latex': latex_output,
                'success': True
            }
            
            if show_steps:
                final_result['steps'] = results.get('steps', [])
                final_result['verification'] = results.get('verification', {})
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in solving pipeline: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def solve_from_file(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Solve equation from image file"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            return self.solve_handwritten_equation(image, **kwargs)
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return {'error': str(e), 'success': False}
    
    def batch_solve(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Solve multiple equations in batch"""
        results = []
        
        for image_path in image_paths:
            logger.info(f"Processing: {image_path}")
            result = self.solve_from_file(image_path, **kwargs)
            result['image_path'] = image_path
            results.append(result)
        
        return results
