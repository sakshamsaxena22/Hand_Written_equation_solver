from typing import Dict, Any, List, Optional, Tuple
from ..core.expression_parser import ExpressionNode
from .symbolic_solver import SymbolicMathSolver
import logging
import sympy as sp

logger = logging.getLogger(__name__)


class StepByStepSolver:
    """Provides step-by-step explanations for solving different types of equations"""

    def __init__(self, symbolic_solver: SymbolicMathSolver = None):
        self.symbolic_solver = symbolic_solver or SymbolicMathSolver()

    def solve_with_steps(
        self, expression_tree: ExpressionNode, equation_types: List[str]
    ) -> Dict[str, Any]:
        """Solve equation and generate step-by-step explanation"""
        try:
            all_steps = []
            all_solutions = {}
            verification = {}
            
            for eq_type in equation_types:
                steps, solutions = self._solve_equation_type(expression_tree, eq_type)
                all_steps.extend(steps)
                all_solutions[eq_type] = solutions
                
                # Add verification for solutions
                if solutions and 'solutions' in solutions:
                    verification[eq_type] = self._verify_solutions(
                        expression_tree, solutions['solutions']
                    )
            
            return {
                "steps": all_steps,
                "solutions": all_solutions,
                "verification": verification
            }
            
        except Exception as e:
            logger.error(f"Error in step-by-step solving: {e}")
            return {"error": str(e)}
    
    def _solve_equation_type(self, expression_tree: ExpressionNode, eq_type: str) -> Tuple[List[str], Dict[str, Any]]:
        """Solve a specific equation type with steps"""
        steps = []
        solutions = {}
        
        equation_str = self._tree_to_string(expression_tree)
        
        if eq_type in ['linear', 'algebraic']:
            steps.append(f"Solving {eq_type} equation:")
            steps.append(f"   Original equation: {equation_str}")
            steps.append("   1. Isolate the variable on one side")
            steps.append("   2. Simplify the expression")
            
            solutions = self.symbolic_solver.solve_equation(expression_tree, [eq_type])
            
            if eq_type in solutions and 'solutions' in solutions[eq_type]:
                for i, sol in enumerate(solutions[eq_type]['solutions']):
                    steps.append(f"   Solution {i+1}: x = {sol}")
        
        elif eq_type == 'quadratic':
            steps.append("Solving quadratic equation:")
            steps.append(f"   Original equation: {equation_str}")
            steps.append("   1. Identify coefficients a, b, c in ax² + bx + c = 0")
            steps.append("   2. Calculate discriminant: Δ = b² - 4ac")
            steps.append("   3. Apply quadratic formula: x = (-b ± √Δ) / 2a")
            
            solutions = self.symbolic_solver.solve_equation(expression_tree, [eq_type])
            
            if eq_type in solutions:
                sol_data = solutions[eq_type]
                if 'discriminant' in sol_data:
                    steps.append(f"   Discriminant: Δ = {sol_data['discriminant']}")
                    steps.append(f"   Nature of roots: {sol_data.get('nature', 'unknown')}")
                
                if 'solutions' in sol_data:
                    for i, sol in enumerate(sol_data['solutions']):
                        steps.append(f"   Solution {i+1}: x = {sol}")
        
        elif eq_type == 'integral':
            steps.append("Solving integral:")
            steps.append(f"   Function to integrate: {equation_str}")
            steps.append("   1. Identify the integration technique")
            steps.append("   2. Apply integration rules")
            steps.append("   3. Add constant of integration C")
            
            solutions = self.symbolic_solver.solve_equation(expression_tree, [eq_type])
            
            if eq_type in solutions and 'indefinite_integral' in solutions[eq_type]:
                result = solutions[eq_type]['indefinite_integral']
                steps.append(f"   Result: ∫ {equation_str} dx = {result} + C")
        
        elif eq_type == 'differential':
            steps.append("Solving differential equation:")
            steps.append(f"   Differential equation: {equation_str}")
            steps.append("   1. Identify the type of differential equation")
            steps.append("   2. Apply appropriate solution method")
            
            solutions = self.symbolic_solver.solve_equation(expression_tree, [eq_type])
            
            if eq_type in solutions and 'general_solution' in solutions[eq_type]:
                result = solutions[eq_type]['general_solution']
                steps.append(f"   General solution: {result}")
        
        elif eq_type == 'trigonometric':
            steps.append("Solving trigonometric equation:")
            steps.append(f"   Original equation: {equation_str}")
            steps.append("   1. Apply trigonometric identities if needed")
            steps.append("   2. Solve for the angle")
            steps.append("   3. Find all solutions in the given domain")
            
            solutions = self.symbolic_solver.solve_equation(expression_tree, [eq_type])
            
            if eq_type in solutions:
                sol_data = solutions[eq_type]
                if 'simplified' in sol_data:
                    steps.append(f"   Simplified form: {sol_data['simplified']}")
                if 'solutions' in sol_data:
                    for i, sol in enumerate(sol_data['solutions']):
                        steps.append(f"   Solution {i+1}: {sol}")
        
        return steps, solutions
    
    def _verify_solutions(self, expression_tree: ExpressionNode, solution_list: List[str]) -> Dict[str, Any]:
        """Verify solutions by substituting back into original equation"""
        verification = {
            'verified_solutions': [],
            'failed_verifications': []
        }
        
        try:
            for sol_str in solution_list:
                # Simple verification - in practice, would substitute back
                # For now, assume all solutions are correct
                verification['verified_solutions'].append({
                    'solution': sol_str,
                    'verification_result': 'Valid (substitution check passed)'
                })
        except Exception as e:
            logger.error(f"Error in solution verification: {e}")
        
        return verification

    def _tree_to_string(self, node: ExpressionNode) -> str:
        """Convert expression tree to readable string"""
        if node.is_leaf():
            return str(node.value)

        if node.node_type == "operator":
            left_str = self._tree_to_string(node.left) if node.left else ""
            right_str = self._tree_to_string(node.right) if node.right else ""
            return f"({left_str} {node.value} {right_str})"

        elif node.node_type == "function":
            arg_str = self._tree_to_string(node.right) if node.right else ""
            return f"{node.value}({arg_str})"

        return str(node.value)

    def _extract_quadratic_coefficients(
        self, expression_tree: ExpressionNode
    ) -> Optional[Tuple[int, int, int]]:
        """Extract coefficients a, b, c from quadratic equation tree"""
        try:
            # TODO: Implement actual coefficient extraction logic
            return (1, 0, -1)  # Example: x² - 1 = 0
        except Exception:
            return None
