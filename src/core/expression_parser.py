# File: src/core/parser.py
"""
Mathematical expression parsing and understanding
"""

from dataclasses import dataclass
from typing import List, Optional
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExpressionNode:
    """Node in mathematical expression tree"""
    value: str
    left: Optional['ExpressionNode'] = None
    right: Optional['ExpressionNode'] = None
    node_type: str = 'operand'  # 'operand', 'operator', 'function'
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

class MathExpressionParser:
    """Advanced mathematical expression parser"""
    
    def __init__(self):
        self.precedence = {
            '+': 1, '-': 1, '*': 2, '/': 2, '^': 3,
            'sin': 4, 'cos': 4, 'tan': 4, 'log': 4, 'ln': 4, 'sqrt': 4,
            'integral': 5, 'sum': 5, 'lim': 5, 'partial': 5
        }
        
        self.functions = {
            'sin', 'cos', 'tan', 'log', 'ln', 'sqrt', 'exp',
            'integral', 'sum', 'lim', 'partial'
        }
        
        self.operators = {'+', '-', '*', '/', '^', '='}
        
    def tokenize(self, symbols: List[str]) -> List[str]:
        """Tokenize mathematical expression"""
        tokens = []
        i = 0
        
        while i < len(symbols):
            symbol = symbols[i]
            
            # Handle multi-character functions
            if i < len(symbols) - 2:
                three_char = ''.join(symbols[i:i+3])
                if three_char in ['sin', 'cos', 'tan', 'log', 'sum', 'lim']:
                    tokens.append(three_char)
                    i += 3
                    continue
            
            if i < len(symbols) - 1:
                two_char = ''.join(symbols[i:i+2])
                if two_char in ['ln', 'pi']:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Handle special symbols
            if symbol == 'sqrt':
                tokens.append('sqrt')
            elif symbol == 'integral':
                tokens.append('integral')
            else:
                tokens.append(symbol)
            
            i += 1
        
        return tokens
    
    def parse_to_tree(self, tokens: List[str]) -> ExpressionNode:
        """Convert tokenized expression to parse tree using Shunting Yard algorithm"""
        output_stack = []
        operator_stack = []
        
        for token in tokens:
            if self._is_number(token) or self._is_variable(token):
                output_stack.append(ExpressionNode(token, node_type='operand'))
            
            elif token in self.functions:
                operator_stack.append(token)
            
            elif token == '(':
                operator_stack.append(token)
            
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    self._create_operator_node(output_stack, operator_stack)
                if operator_stack:
                    operator_stack.pop()  # Remove '('
            
            elif token in self.operators:
                while (operator_stack and operator_stack[-1] != '(' and
                       operator_stack[-1] in self.precedence and
                       self.precedence.get(operator_stack[-1], 0) >= self.precedence.get(token, 0)):
                    self._create_operator_node(output_stack, operator_stack)
                operator_stack.append(token)
        
        # Process remaining operators
        while operator_stack:
            self._create_operator_node(output_stack, operator_stack)
        
        return output_stack[0] if output_stack else ExpressionNode('0')
    
    def _create_operator_node(self, output_stack: List[ExpressionNode], operator_stack: List[str]):
        """Create operator node and attach operands"""
        if not operator_stack:
            return
            
        operator = operator_stack.pop()
        
        if operator in self.functions:
            # Unary function
            if output_stack:
                operand = output_stack.pop()
                node = ExpressionNode(operator, right=operand, node_type='function')
                output_stack.append(node)
        else:
            # Binary operator
            if len(output_stack) >= 2:
                right = output_stack.pop()
                left = output_stack.pop()
                node = ExpressionNode(operator, left=left, right=right, node_type='operator')
                output_stack.append(node)
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number"""
        try:
            float(token)
            return True
        except ValueError:
            return token.replace('.', '').isdigit()
    
    def _is_variable(self, token: str) -> bool:
        """Check if token is a variable"""
        return token.isalpha() and len(token) == 1 and token not in self.functions

class EquationTypeClassifier:
    """Classify mathematical equation types"""
    
    def __init__(self):
        self.equation_patterns = {
            'linear': [r'[a-z]\s*=', r'^\d*[a-z]\s*[\+\-]\s*\d+\s*='],
            'quadratic': [r'[a-z]\^2', r'[a-z]²'],
            'differential': [r'dy/dx', r'd[a-z]/d[a-z]', r'∂', r'partial'],
            'integral': [r'∫', r'integral'],
            'trigonometric': [r'sin\(', r'cos\(', r'tan\('],
            'logarithmic': [r'log\(', r'ln\('],
            'exponential': [r'e\^', r'exp\('],
            'polynomial': [r'[a-z]\^[3-9]', r'[a-z]³', r'[a-z]⁴'],
            'system': [r'=.*\n.*=', r'\{.*\}'],
            'matrix': [r'\[.*\]', r'matrix'],
            'calculus': [r'lim', r'limit', r'∑', r'sum']
        }
    
    def classify(self, expression: str) -> List[str]:
        """Classify equation type(s)"""
        expression_lower = expression.lower()
        detected_types = []
        
        for eq_type, patterns in self.equation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, expression_lower):
                    detected_types.append(eq_type)
                    break
        
        # Default to algebraic if no specific type detected
        if not detected_types and '=' in expression:
            detected_types.append('algebraic')
        
        return detected_types if detected_types else ['arithmetic'] 
