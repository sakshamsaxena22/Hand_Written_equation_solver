# File: src/solvers/symbolic_solver.py
"""
Symbolic mathematics solver using SymPy
"""

import sympy as sp
from sympy import symbols, solve, diff, integrate, limit, Matrix, dsolve, Eq
from typing import Dict, Any, List
from src.core.expression_parser import ExpressionNode
import logging

logger = logging.getLogger(__name__)

class SymbolicMathSolver:
    """Advanced symbolic mathematics solver using SymPy"""
    
    def __init__(self):
        self.common_symbols = symbols('x y z a b c t u v w')
        self.x, self.y, self.z = symbols('x y z')
        
        self.functions_map = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'log': sp.log, 'ln': sp.ln, 'exp': sp.exp, 'sqrt': sp.sqrt,
            'integral': sp.integrate, 'sum': sp.Sum, 'lim': sp.limit
        }
    
    def solve_equation(self, expression_tree: ExpressionNode, equation_types: List[str]) -> Dict[str, Any]:
        """Solve mathematical equation based on type"""
        try:
            # Convert expression tree to SymPy
            sympy_expr = self._tree_to_sympy(expression_tree)
            
            results = {}
            
            for eq_type in equation_types:
                if eq_type == 'algebraic' or eq_type == 'linear':
                    results[eq_type] = self._solve_algebraic(sympy_expr)
                elif eq_type == 'quadratic':
                    results[eq_type] = self._solve_quadratic(sympy_expr)
                elif eq_type == 'differential':
                    results[eq_type] = self._solve_differential(sympy_expr)
                elif eq_type == 'integral':
                    results[eq_type] = self._solve_integral(sympy_expr)
                elif eq_type == 'calculus':
                    results[eq_type] = self._solve_calculus(sympy_expr)
                elif eq_type == 'trigonometric':
                    results[eq_type] = self._solve_trigonometric(sympy_expr)
                elif eq_type == 'system':
                    results[eq_type] = self._solve_system(sympy_expr)
            
            return results
            
        except Exception as e:
            logger.error(f"Error solving equation: {e}")
            return {'error': str(e)}
    
    def _tree_to_sympy(self, node: ExpressionNode) -> sp.Basic:
        """Convert expression tree to SymPy expression"""
        if node.is_leaf():
            # Handle numbers and variables
            if node.value.isdigit() or '.' in node.value:
                return sp.Float(node.value)
            elif node.value in ['x', 'y', 'z', 't', 'u', 'v', 'w', 'a', 'b', 'c']:
                return symbols(node.value)
            elif node.value == 'pi':
                return sp.pi
            elif node.value == 'e':
                return sp.E
            else:
                return symbols(node.value)
        
        # Handle operators and functions
        if node.node_type == 'operator':
            left_expr = self._tree_to_sympy(node.left)
            right_expr = self._tree_to_sympy(node.right)
            
            if node.value == '+':
                return left_expr + right_expr
            elif node.value == '-':
                return left_expr - right_expr
            elif node.value == '*':
                return left_expr * right_expr
            elif node.value == '/':
                return left_expr / right_expr
            elif node.value == '^':
                return left_expr ** right_expr
            elif node.value == '=':
                return sp.Eq(left_expr, right_expr)
        
        elif node.node_type == 'function':
            if node.right:
                arg = self._tree_to_sympy(node.right)
                if node.value in self.functions_map:
                    return self.functions_map[node.value](arg)
        
        return sp.Symbol('unknown')
    
    def _solve_algebraic(self, expr: sp.Basic) -> Dict[str, Any]:
        """Solve algebraic equations"""
        try:
            solutions = sp.solve(expr, self.x)
            return {
                'solutions': [str(sol) for sol in solutions],
                'numerical_solutions': [float(sol.evalf()) if sol.is_real else complex(sol.evalf()) 
                                      for sol in solutions if sol.is_number]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _solve_quadratic(self, expr: sp.Basic) -> Dict[str, Any]:
        """Solve quadratic equations"""
        try:
            solutions = sp.solve(expr, self.x)
            discriminant = None
            
            # Calculate discriminant for quadratic
            if len(solutions) == 2:
                poly = sp.Poly(expr.lhs - expr.rhs if hasattr(expr, 'lhs') else expr, self.x)
                if poly.degree() == 2:
                    coeffs = poly.all_coeffs()
                    if len(coeffs) == 3:
                        a, b, c = coeffs
                        discriminant = b**2 - 4*a*c
            
            return {
                'solutions': [str(sol) for sol in solutions],
                'discriminant': str(discriminant) if discriminant is not None else None,
                'nature': 'real_distinct' if discriminant and discriminant > 0 else 
                         'real_equal' if discriminant and discriminant == 0 else 'complex'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _solve_differential(self, expr: sp.Basic) -> Dict[str, Any]:
        """Solve differential equations"""
        try:
            # Assume first-order ODE in y(x)
            y = sp.Function('y')
            x = sp.Symbol('x')
            
            solution = sp.dsolve(expr, y(x))
            return {
                'general_solution': str(solution),
                'type': 'ODE'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _solve_integral(self, expr: sp.Basic) -> Dict[str, Any]:
        """Solve integral expressions"""
        try:
            # Determine integration variable
            variables = list(expr.free_symbols)
            var = variables[0] if variables else self.x
            
            indefinite = sp.integrate(expr, var)
            
            return {
                'indefinite_integral': str(indefinite),
                'variable': str(var)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _solve_calculus(self, expr: sp.Basic) -> Dict[str, Any]:
        """Solve calculus operations"""
        try:
            variables = list(expr.free_symbols)
            var = variables[0] if variables else self.x
            
            derivative = sp.diff(expr, var)
            integral = sp.integrate(expr, var)
            
            return {
                'derivative': str(derivative),
                'integral': str(integral),
                'variable': str(var)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _solve_trigonometric(self, expr: sp.Basic) -> Dict[str, Any]:
        """Solve trigonometric equations"""
        try:
            solutions = sp.solve(expr, self.x)
            simplified = sp.trigsimp(expr)
            
            return {
                'solutions': [str(sol) for sol in solutions],
                'simplified': str(simplified)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _solve_system(self, equations: List[sp.Basic]) -> Dict[str, Any]:
        """Solve system of equations"""
        try:
            variables = list(set().union(*[eq.free_symbols for eq in equations]))
            solutions = sp.solve(equations, variables)
            
            return {
                'solutions': {str(var): str(val) for var, val in solutions.items()},
                'variables': [str(var) for var in variables]
            }
        except Exception as e:
            return {'error': str(e)}
 
