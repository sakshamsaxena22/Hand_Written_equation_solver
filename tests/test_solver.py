# File: tests/test_solver.py
"""
Unit tests for the equation solver
"""

import pytest
import numpy as np
from src.main import UniversalEquationSolver
from src.core.parser import ExpressionNode

class TestUniversalEquationSolver:
    """Test the equation solver with sample equations"""
    
    def setup_method(self):
        """Setup test solver"""
        self.solver = UniversalEquationSolver()
    
    def test_linear_equation(self):
        """Test linear equation parsing and solving"""
        symbols = ['2', 'x', '+', '3', '=', '7']
        tokens = self.solver.parser.tokenize(symbols)
        tree = self.solver.parser.parse_to_tree(tokens)
        
        expr_str = ' '.join(tokens)
        eq_types = self.solver.classifier.classify(expr_str)
        
        assert 'linear' in eq_types or 'algebraic' in eq_types
        
    def test_quadratic_equation(self):
        """Test quadratic equation recognition"""
        symbols = ['x', '^', '2', '+', '2', 'x', '+', '1', '=', '0']
        tokens = self.solver.parser.tokenize(symbols)
        expr_str = ' '.join(tokens)
        eq_types = self.solver.classifier.classify(expr_str)
        
        assert 'quadratic' in eq_types
    
    def test_trigonometric_function(self):
        """Test trigonometric function recognition"""
        symbols = ['sin', '(', 'x', ')', '=', '0']
        tokens = self.solver.parser.tokenize(symbols)
        expr_str = ' '.join(tokens)
        eq_types = self.solver.classifier.classify(expr_str)
        
        assert 'trigonometric' in eq_types