"""
LaTeX output generation for mathematical expressions (Solver side)
"""

from typing import Dict, Any
from ..core.expression_parser import ExpressionNode   # fixed relative import
import logging

logger = logging.getLogger(__name__)

class LaTeXGenerator:
    """Generate LaTeX output for mathematical expressions and solutions"""

    def __init__(self):
        self.latex_mappings = {
            'integral': r'\int',
            'sum': r'\sum',
            'sqrt': r'\sqrt',
            'lim': r'\lim',
            'partial': r'\partial',
            'sin': r'\sin',
            'cos': r'\cos',
            'tan': r'\tan',
            'log': r'\log',
            'ln': r'\ln',
            'pi': r'\pi',
            'alpha': r'\alpha',
            'beta': r'\beta',
            'gamma': r'\gamma'
        }

    def generate_latex(
        self, expression_tree: ExpressionNode, solutions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate LaTeX representation of equation and solutions"""
        try:
            equation_latex = self._tree_to_latex(expression_tree)

            latex_output: Dict[str, Any] = {
                'equation': f"${equation_latex}$",
                'solutions': {}
            }

            for eq_type, solution in solutions.items():
                if isinstance(solution, dict) and 'solutions' in solution:
                    solutions_latex = [f"${sol}$" for sol in solution['solutions']]
                    latex_output['solutions'][eq_type] = solutions_latex
                else:
                    latex_output['solutions'][eq_type] = f"${str(solution)}$"

            return latex_output

        except Exception as e:
            logger.error(f"Error generating LaTeX: {e}")
            return {'error': str(e)}

    def _tree_to_latex(self, node: ExpressionNode) -> str:
        """Convert expression tree to LaTeX format"""
        if node is None:
            return ""

        if node.is_leaf():
            return self.latex_mappings.get(node.value, str(node.value))

        if node.node_type == 'operator':
            left_latex = self._tree_to_latex(node.left) if node.left else ''
            right_latex = self._tree_to_latex(node.right) if node.right else ''

            if node.value == '/':
                return f"\\frac{{{left_latex}}}{{{right_latex}}}"
            elif node.value == '^':
                return f"{left_latex}^{{{right_latex}}}"
            elif node.value == '*':
                return f"{left_latex} \\cdot {right_latex}"
            elif node.value == '=':
                return f"{left_latex} = {right_latex}"
            else:
                return f"{left_latex} {node.value} {right_latex}"

        elif node.node_type == 'function':
            arg_latex = self._tree_to_latex(node.right) if node.right else ''
            func_latex = self.latex_mappings.get(node.value, node.value)

            if node.value == 'sqrt':
                return f"\\sqrt{{{arg_latex}}}"
            elif node.value == 'integral':
                return f"\\int {arg_latex} \\, dx"
            elif node.value in ['sin', 'cos', 'tan', 'log', 'ln']:
                return f"\\{node.value}({arg_latex})"
            else:
                return f"{func_latex}({arg_latex})"

        return str(node.value) if node.value else ""
