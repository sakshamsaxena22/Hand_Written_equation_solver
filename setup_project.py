"""
Project Setup Script
====================

Simple setup script to install dependencies and test the system.
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_basic_requirements():
    """Install only the basic requirements needed for testing"""
    basic_requirements = [
        'numpy',
        'opencv-python',
        'torch',
        'torchvision', 
        'flask',
        'sympy',
        'matplotlib',
        'scikit-image',
        'Pillow',
        'tqdm'
    ]
    
    logger.info("Installing basic requirements...")
    for package in basic_requirements:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to install {package}, continuing...")

def test_basic_functionality():
    """Test basic functionality without advanced features"""
    logger.info("Testing basic functionality...")
    
    try:
        import numpy as np
        import cv2
        logger.info("✓ NumPy and OpenCV imported successfully")
        
        # Create test image
        image = np.ones((200, 400), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, '2x + 5 = 13', (50, 100), font, 1, (0, 0, 0), 2)
        logger.info("✓ Test image created successfully")
        
        # Test image processing
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
        logger.info("✓ Basic image processing works")
        
        import sympy as sp
        x = sp.Symbol('x')
        equation = sp.Eq(2*x + 5, 13)
        solution = sp.solve(equation, x)
        logger.info(f"✓ SymPy works: 2x + 5 = 13 -> x = {solution}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

def create_demo_solver():
    """Create a simplified demo solver"""
    demo_code = '''
"""
Demo Mathematical Expression Solver
"""

import numpy as np
import cv2
import sympy as sp
from typing import Dict, Any

def solve_simple_equation(equation_str: str) -> Dict[str, Any]:
    """Solve a simple equation using SymPy"""
    try:
        # Parse the equation
        if '=' in equation_str:
            left, right = equation_str.split('=')
            left_expr = sp.sympify(left.strip())
            right_expr = sp.sympify(right.strip())
            equation = sp.Eq(left_expr, right_expr)
        else:
            equation = sp.sympify(equation_str)
        
        # Find variables
        variables = list(equation.free_symbols)
        
        if variables:
            # Solve for the first variable
            solution = sp.solve(equation, variables[0])
            
            return {
                'success': True,
                'equation': str(equation),
                'variable': str(variables[0]),
                'solution': [str(sol) for sol in solution],
                'steps': [
                    {'step': 1, 'description': f'Original equation: {equation}'},
                    {'step': 2, 'description': f'Solve for {variables[0]}'},
                    {'step': 3, 'description': f'Solution: {solution}'}
                ]
            }
        else:
            # No variables, just evaluate
            result = sp.simplify(equation)
            return {
                'success': True,
                'equation': str(equation),
                'result': str(result),
                'steps': [
                    {'step': 1, 'description': f'Expression: {equation}'},
                    {'step': 2, 'description': f'Simplified: {result}'}
                ]
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'equation': equation_str
        }

def demo_solve():
    """Demo the solver with sample equations"""
    test_equations = [
        '2*x + 5 = 13',
        'x**2 - 4 = 0', 
        '3*y - 7 = 14',
        'sin(x) = 0',
        'x + y = 10'
    ]
    
    print("=== Demo Mathematical Expression Solver ===\\n")
    
    for eq in test_equations:
        print(f"Solving: {eq}")
        result = solve_simple_equation(eq)
        
        if result['success']:
            print(f"✓ Success!")
            if 'solution' in result:
                print(f"  Variable: {result['variable']}")
                print(f"  Solution: {result['solution']}")
            else:
                print(f"  Result: {result['result']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 50)

if __name__ == '__main__':
    demo_solve()
'''
    
    with open('demo_solver.py', 'w') as f:
        f.write(demo_code)
    
    logger.info("✓ Demo solver created as demo_solver.py")

def main():
    """Main setup function"""
    logger.info("Setting up Hand-Written Equation Solver project...")
    
    # Install basic requirements
    install_basic_requirements()
    
    # Test basic functionality
    if test_basic_functionality():
        logger.info("✅ Basic setup successful!")
        
        # Create demo solver
        create_demo_solver()
        
        logger.info("\\n🎉 Setup complete! You can now:")
        logger.info("  1. Run the demo solver: python demo_solver.py")
        logger.info("  2. Run the web interface: python web_app.py")
        logger.info("  3. Access the web interface at: http://localhost:5000")
        
        return True
    else:
        logger.error("❌ Basic setup failed. Please install dependencies manually.")
        return False

if __name__ == '__main__':
    main()
