# File: src/utils/context_recognizer.py
"""
Context-aware mathematical expression recognition
"""

from typing import List, Dict, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)

class ContextAwareRecognizer:
    """Context-aware mathematical expression recognition and enhancement"""
    
    def __init__(self):
        self.context_keywords = {
            'physics': {
                'keywords': ['force', 'velocity', 'acceleration', 'energy', 'momentum', 
                           'F=ma', 'newton', 'kinetic', 'potential', 'mass', 'gravity'],
                'symbols': {'F': 'force', 'm': 'mass', 'a': 'acceleration', 'v': 'velocity',
                          'E': 'energy', 'p': 'momentum', 'g': 'gravity'}
            },
            'engineering': {
                'keywords': ['circuit', 'voltage', 'current', 'resistance', 'power',
                           'ohm', 'watt', 'ampere', 'volt', 'capacitor', 'inductor'],
                'symbols': {'V': 'voltage', 'I': 'current', 'R': 'resistance', 'P': 'power',
                          'C': 'capacitance', 'L': 'inductance'}
            },
            'pure_math': {
                'keywords': ['theorem', 'proof', 'lemma', 'derivative', 'integral',
                           'limit', 'function', 'domain', 'range', 'continuous'],
                'symbols': {'f': 'function', 'x': 'variable', 'y': 'variable', 'n': 'index'}
            },
            'statistics': {
                'keywords': ['mean', 'variance', 'probability', 'distribution',
                           'normal', 'gaussian', 'correlation', 'regression'],
                'symbols': {'μ': 'mean', 'σ': 'standard_deviation', 'σ²': 'variance',
                          'ρ': 'correlation', 'p': 'probability'}
            },
            'geometry': {
                'keywords': ['area', 'volume', 'perimeter', 'angle', 'triangle',
                           'circle', 'radius', 'diameter', 'circumference'],
                'symbols': {'A': 'area', 'V': 'volume', 'P': 'perimeter', 'r': 'radius',
                          'd': 'diameter', 'θ': 'angle', 'π': 'pi'}
            },
            'calculus': {
                'keywords': ['derivative', 'integral', 'limit', 'differential',
                           'taylor', 'series', 'convergence', 'divergence'],
                'symbols': {'dx': 'differential', 'dy': 'differential', '∫': 'integral',
                          '∂': 'partial_derivative', '∑': 'sum', 'lim': 'limit'}
            }
        }
        
        # Common symbol confusions that context can help resolve
        self.symbol_confusions = {
            ('I', '1'): ['current', 'identity', 'one'],
            ('O', '0'): ['origin', 'zero'],
            ('l', '1', 'I'): ['length', 'one', 'current'],
            ('x', '×'): ['variable', 'multiply']
        }
    
    def detect_context(self, text: str) -> Tuple[str, float]:
        """Detect mathematical context from surrounding text with confidence"""
        if not text:
            return 'general', 0.0
            
        text_lower = text.lower()
        
        context_scores = {}
        
        for context, context_data in self.context_keywords.items():
            keywords = context_data['keywords']
            score = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight longer keywords more heavily
                    score += len(keyword.split())
            
            # Normalize score
            if total_keywords > 0:
                context_scores[context] = score / total_keywords
        
        if not context_scores:
            return 'general', 0.0
        
        best_context = max(context_scores.items(), key=lambda x: x[1])
        return best_context[0], best_context[1]
    
    def detect_context_from_symbols(self, symbols: List[str]) -> Tuple[str, float]:
        """Detect context from recognized mathematical symbols"""
        if not symbols:
            return 'general', 0.0
        
        context_scores = {}
        
        for context, context_data in self.context_keywords.items():
            context_symbols = context_data['symbols']
            score = 0
            
            for symbol in symbols:
                if symbol in context_symbols:
                    score += 1
                # Check for partial matches (e.g., Greek letters)
                elif any(sym in symbol for sym in context_symbols.keys()):
                    score += 0.5
            
            if len(context_symbols) > 0:
                context_scores[context] = score / len(context_symbols)
        
        if not context_scores:
            return 'general', 0.0
        
        best_context = max(context_scores.items(), key=lambda x: x[1])
        return best_context[0], best_context[1]
    def enhance_recognition(self, base_recognition: List[str], context: str) -> List[str]:
        """Enhance recognition based on context with symbol disambiguation"""
        if not base_recognition or context == 'general':
            return base_recognition
        
        enhanced = []
        context_data = self.context_keywords.get(context, {})
        context_symbols = context_data.get('symbols', {})
        
        for i, symbol in enumerate(base_recognition):
            # Apply context-specific enhancements
            enhanced_symbol = self._enhance_symbol(symbol, context, context_symbols)
            
            # Resolve ambiguous symbols using context
            resolved_symbol = self._resolve_ambiguous_symbol(
                enhanced_symbol, base_recognition, i, context
            )
            
            enhanced.append(resolved_symbol)
        
        return enhanced
    
    def _enhance_symbol(self, symbol: str, context: str, context_symbols: Dict[str, str]) -> str:
        """Apply context-specific symbol enhancements"""
        
        # Direct symbol mapping
        if symbol in context_symbols:
            return symbol  # Keep the symbol but mark as context-appropriate
        
        # Context-specific transformations
        if context == 'physics':
            # Common physics symbol corrections
            if symbol.lower() == 'f' and self._is_likely_force_context():
                return 'F'  # Force is typically uppercase
            elif symbol.lower() == 'm' and self._is_likely_mass_context():
                return 'm'  # Mass is typically lowercase
                
        elif context == 'engineering':
            # Electrical engineering conventions
            if symbol.lower() == 'v':
                return 'V'  # Voltage is typically uppercase
            elif symbol.lower() == 'i' and self._is_likely_current_context():
                return 'I'  # Current is typically uppercase
                
        elif context == 'statistics':
            # Statistics symbol conventions
            if symbol == 'u' and self._is_likely_mean_context():
                return 'μ'  # Greek mu for mean
            elif symbol == 's' and self._is_likely_std_context():
                return 'σ'  # Greek sigma for standard deviation
        
        return symbol
    
    def _resolve_ambiguous_symbol(self, symbol: str, all_symbols: List[str], 
                                position: int, context: str) -> str:
        """Resolve ambiguous symbols using context and surrounding symbols"""
        
        # Check for common confusions
        for confusion_set, meanings in self.symbol_confusions.items():
            if symbol in confusion_set:
                # Use context to disambiguate
                if context == 'engineering' and 'current' in meanings:
                    return 'I' if symbol in ['I', '1', 'l'] else symbol
                elif context == 'pure_math' and 'one' in meanings:
                    return '1' if symbol in ['I', 'l', '1'] else symbol
        
        # Context-based disambiguation
        if symbol in ['0', 'O', 'o']:
            if context in ['pure_math', 'statistics'] and self._is_likely_zero_context(all_symbols, position):
                return '0'
            elif context == 'geometry' and self._is_likely_origin_context(all_symbols, position):
                return 'O'
        
        return symbol
    
    def _is_likely_force_context(self) -> bool:
        """Check if we're likely in a force equation context"""
        # Simple heuristic - in practice would be more sophisticated
        return True
    
    def _is_likely_mass_context(self) -> bool:
        """Check if we're likely in a mass equation context"""
        return True
    
    def _is_likely_current_context(self) -> bool:
        """Check if we're likely in an electrical current context"""
        return True
    
    def _is_likely_mean_context(self) -> bool:
        """Check if we're likely referring to statistical mean"""
        return True
    
    def _is_likely_std_context(self) -> bool:
        """Check if we're likely referring to standard deviation"""
        return True
    
    def _is_likely_zero_context(self, all_symbols: List[str], position: int) -> bool:
        """Check if symbol is likely the number zero"""
        # Look for numerical context
        adjacent_symbols = []
        if position > 0:
            adjacent_symbols.append(all_symbols[position - 1])
        if position < len(all_symbols) - 1:
            adjacent_symbols.append(all_symbols[position + 1])
        
        # If surrounded by numbers or operators, likely a zero
        return any(s.isdigit() or s in ['+', '-', '*', '/', '='] for s in adjacent_symbols)
    
    def _is_likely_origin_context(self, all_symbols: List[str], position: int) -> bool:
        """Check if symbol is likely referring to origin point"""
        # Look for coordinate context like (O, 0) or point O
        return any(s in ['(', ')', ',', 'point'] for s in all_symbols)
    
    def get_context_suggestions(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Get context-based suggestions for symbol meanings"""
        suggestions = {}
        
        for symbol in symbols:
            symbol_suggestions = []
            
            for context, context_data in self.context_keywords.items():
                context_symbols = context_data['symbols']
                if symbol in context_symbols:
                    symbol_suggestions.append(f"{context}: {context_symbols[symbol]}")
            
            if symbol_suggestions:
                suggestions[symbol] = symbol_suggestions
        
        return suggestions
