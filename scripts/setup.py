# File: scripts/setup.py
"""
Setup script for the project
"""

import os
import logging
from pathlib import Path

def setup_project():
    """Setup project directories and initial configuration"""
    
    # Create necessary directories
    directories = [
        'models',
        'temp',
        'static',
        'logs',
        'data',
        'src/core',
        'src/models',
        'src/solvers', 
        'src/output',
        'src/utils',
        'api',
        'tests',
        'config',
        'scripts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/core/__init__.py',
        'src/models/__init__.py',
        'src/solvers/__init__.py',
        'src/output/__init__.py',
        'src/utils/__init__.py',
        'api/__init__.py',
        'tests/__init__.py',
        'config/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print(f"✅ Created: {init_file}")
    
    print("\n🚀 Project setup complete!")
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run setup: python scripts/setup.py")
    print("3. Start server: python cli.py --server")
    print("4. Or solve single equation: python cli.py --image path/to/equation.jpg")

if __name__ == "__main__":
    setup_project()


print("✅ Universal Mathematical Equation Solver - Complete File Structure")
print("\n📁 Project Structure:")
print("├── src/")
print("│   ├── core/")
print("│   │   ├── preprocessor.py         # Advanced image preprocessing")
print("│   │   ├── segmentation.py         # Mathematical expression segmentation") 
print("│   │   └── parser.py              # Expression parsing and classification")
print("│   ├── models/")
print("│   │   └── transformer_ocr.py     # Transformer-based OCR model")
print("│   ├── solvers/")
print("│   │   ├── symbolic_solver.py     # SymPy-based symbolic solver")
print("│   │   └── step_solver.py         # Step-by-step solution generator")
print("│   ├── output/")
print("│   │   └── latex_generator.py     # LaTeX output generation")
print("│   ├── utils/")
print("│   │   └── context_recognizer.py  # Context-aware recognition")
print("│   └── main.py                    # Main solver system")
print("├── api/")
print("│   └── app.py                     # FastAPI web application")
print("├── tests/")
print("│   └── test_solver.py             # Unit tests")
print("├── config/")
print("│   └── settings.py                # Configuration settings")
print("├── scripts/")
print("│   ├── train_model.py             # Model training script")
print("│   └── setup.py                   # Project setup script")
print("├── cli.py                         # Command line interface")
print("├── requirements.txt               # Dependencies")
print("├── Dockerfile                     # Container configuration")
print("└── README.md                      # Documentation")
print("\n🚀 Ready to deploy!")
'error': str(e)}
    
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

