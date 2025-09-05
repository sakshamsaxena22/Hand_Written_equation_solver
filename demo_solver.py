"""
Demo Mathematical Expression Solver
===================================

A simple demonstration of the equation solver capabilities using SymPy.
"""

import sympy as sp
from typing import Dict, Any

def solve_simple_equation(equation_str: str) -> Dict[str, Any]:
    """Solve a simple equation using SymPy"""
    try:
        print(f"Solving: {equation_str}")
        
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
            
            result = {
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
            
            print(f"  Variable: {result['variable']}")
            print(f"  Solution: {result['solution']}")
            
            return result
        else:
            # No variables, just evaluate
            result_val = sp.simplify(equation)
            result = {
                'success': True,
                'equation': str(equation),
                'result': str(result_val),
                'steps': [
                    {'step': 1, 'description': f'Expression: {equation}'},
                    {'step': 2, 'description': f'Simplified: {result_val}'}
                ]
            }
            
            print(f"  Result: {result['result']}")
            return result
            
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'equation': equation_str
        }
        print(f"  Error: {result['error']}")
        return result

def demo_solve():
    """Demo the solver with sample equations"""
    test_equations = [
        '2*x + 5 = 13',
        'x**2 - 4 = 0', 
        '3*y - 7 = 14',
        'x + y = 10',
        '2*x**2 - 8*x + 8 = 0'
    ]
    
    print("=== Basic Mathematical Expression Solver ===\n")
    
    for eq in test_equations:
        solve_simple_equation(eq)
        print("-" * 50)

def demo_advanced_math():
    """Demo advanced mathematical capabilities"""
    print("\n=== Advanced Mathematics Demo ===\n")
    
    # Calculus examples
    print("Calculus Operations:")
    print("-" * 20)
    
    x = sp.Symbol('x')
    
    # Derivative
    expr = x**2 + 3*x + 1
    derivative = sp.diff(expr, x)
    print(f"d/dx({expr}) = {derivative}")
    
    # Integration
    integral = sp.integrate(expr, x)
    print(f"Integral of ({expr})dx = {integral}")
    
    # Trigonometric identity
    trig_expr = sp.sin(x)**2 + sp.cos(x)**2
    simplified = sp.simplify(trig_expr)
    print(f"sin^2(x) + cos^2(x) = {simplified}")
    
    print("-" * 50)

def main():
    """Main demo function"""
    print("Hand-Written Equation Solver - Demo")
    print("=" * 50)
    
    # Basic equation solving
    demo_solve()
    
    # Advanced math demo
    demo_advanced_math()
    
    print("\n=== Demo Complete ===")
    print("This demonstrates the core mathematical solving capabilities.")
    print("The full system integrates this with:")
    print("  1. Image preprocessing and OCR")
    print("  2. Character recognition with Gemini fallback")
    print("  3. Web interface for easy interaction")

if __name__ == '__main__':
    main()
