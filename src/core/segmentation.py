"""
Mathematical Expression Segmentation Module
Segments handwritten mathematical expressions into individual symbols and sub-expressions
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

logger = logging.getLogger(__name__)

class MathExpressionSegmenter:
    """Advanced segmentation for mathematical expressions"""
    
    def __init__(self, min_symbol_area: int = 50, max_symbol_area: int = 5000):
        self.min_symbol_area = min_symbol_area
        self.max_symbol_area = max_symbol_area
        self.symbol_height_range = (10, 100)
        self.symbol_width_range = (5, 80)
    
    def segment_expression(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Segment mathematical expression into individual symbols"""
        try:
            # Preprocess for segmentation
            processed = self._preprocess_for_segmentation(image)
            
            # Detect connected components
            components = self._detect_components(processed)
            
            # Filter and classify components
            symbols = self._filter_components(components, image)
            
            # Sort symbols left-to-right, top-to-bottom
            symbols = self._sort_symbols(symbols)
            
            # Extract symbol regions
            symbol_data = self._extract_symbol_regions(symbols, image)
            
            logger.info(f"Segmented {len(symbol_data)} symbols")
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return []
    
    def _preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for segmentation"""
        # Ensure binary image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold if not already binary
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = image
        
        # Remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _detect_components(self, binary_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect connected components in the image"""
        # Label connected components
        labeled = label(binary_image)
        regions = regionprops(labeled)
        
        components = []
        for region in regions:
            bbox = region.bbox
            area = region.area
            centroid = region.centroid
            
            components.append({
                'bbox': bbox,  # (min_row, min_col, max_row, max_col)
                'area': area,
                'centroid': centroid,
                'region': region
            })
        
        return components
    
    def _filter_components(self, components: List[Dict], original_image: np.ndarray) -> List[Dict]:
        """Filter components based on size and shape criteria"""
        filtered = []
        
        for comp in components:
            bbox = comp['bbox']
            area = comp['area']
            
            # Calculate dimensions
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            
            # Apply filters
            if (self.min_symbol_area <= area <= self.max_symbol_area and
                self.symbol_height_range[0] <= height <= self.symbol_height_range[1] and
                self.symbol_width_range[0] <= width <= self.symbol_width_range[1]):
                
                # Additional shape analysis
                aspect_ratio = width / height if height > 0 else 0
                
                # Filter out very thin lines (likely noise) unless they're dashes
                if aspect_ratio > 5 and area < 100:
                    # Could be a minus sign or fraction line
                    if self._is_horizontal_line(comp):
                        comp['symbol_type'] = 'horizontal_line'
                        filtered.append(comp)
                elif aspect_ratio < 0.1 and area < 100:
                    # Could be a vertical line or part of a symbol
                    if self._is_vertical_line(comp):
                        comp['symbol_type'] = 'vertical_line'
                        filtered.append(comp)
                else:
                    comp['symbol_type'] = 'symbol'
                    filtered.append(comp)
        
        return filtered
    
    def _is_horizontal_line(self, component: Dict) -> bool:
        """Check if component is a horizontal line (minus, fraction line)"""
        bbox = component['bbox']
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        
        return width > 3 * height and component['area'] > 10
    
    def _is_vertical_line(self, component: Dict) -> bool:
        """Check if component is a vertical line"""
        bbox = component['bbox']
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        
        return height > 3 * width and component['area'] > 10
    
    def _sort_symbols(self, symbols: List[Dict]) -> List[Dict]:
        """Sort symbols in reading order (left-to-right, top-to-bottom)"""
        # Group by approximate vertical position (row)
        rows = self._group_by_rows(symbols)
        
        sorted_symbols = []
        for row in rows:
            # Sort each row left-to-right
            row_sorted = sorted(row, key=lambda x: x['bbox'][1])  # Sort by min_col
            sorted_symbols.extend(row_sorted)
        
        return sorted_symbols
    
    def _group_by_rows(self, symbols: List[Dict]) -> List[List[Dict]]:
        """Group symbols into rows based on vertical position"""
        if not symbols:
            return []
        
        # Sort by vertical position
        symbols_by_y = sorted(symbols, key=lambda x: x['centroid'][0])
        
        rows = []
        current_row = [symbols_by_y[0]]
        current_y = symbols_by_y[0]['centroid'][0]
        
        # Group symbols with similar y-coordinates
        for symbol in symbols_by_y[1:]:
            symbol_y = symbol['centroid'][0]
            
            # If within reasonable distance, add to current row
            if abs(symbol_y - current_y) < 20:  # Threshold for row grouping
                current_row.append(symbol)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [symbol]
                current_y = symbol_y
        
        rows.append(current_row)
        return rows
    
    def _extract_symbol_regions(self, symbols: List[Dict], original_image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract image regions for each symbol"""
        symbol_data = []
        
        for i, symbol in enumerate(symbols):
            bbox = symbol['bbox']
            
            # Extract region with padding
            padding = 5
            min_row = max(0, bbox[0] - padding)
            min_col = max(0, bbox[1] - padding)
            max_row = min(original_image.shape[0], bbox[2] + padding)
            max_col = min(original_image.shape[1], bbox[3] + padding)
            
            symbol_image = original_image[min_row:max_row, min_col:max_col]
            
            # Resize to standard size for recognition
            symbol_resized = cv2.resize(symbol_image, (32, 32))
            
            symbol_data.append({
                'id': i,
                'image': symbol_resized,
                'original_image': symbol_image,
                'bbox': (min_row, min_col, max_row, max_col),
                'original_bbox': bbox,
                'centroid': symbol['centroid'],
                'area': symbol['area'],
                'symbol_type': symbol.get('symbol_type', 'symbol'),
                'position': {
                    'x': (min_col + max_col) // 2,
                    'y': (min_row + max_row) // 2
                }
            })
        
        return symbol_data
    
    def detect_expression_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect horizontal lines that might be fraction bars or equation separators"""
        # Use Hough line detection for precise line detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=20, maxLineGap=10)
        
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is mostly horizontal
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15 or angle > 165:  # Nearly horizontal
                    horizontal_lines.append({
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': np.sqrt((x2-x1)**2 + (y2-y1)**2),
                        'y_position': (y1 + y2) // 2
                    })
        
        return horizontal_lines
    
    def detect_composite_symbols(self, symbols: List[Dict[str, Any]], 
                               spatial_threshold: float = 30.0) -> List[Dict[str, Any]]:
        """
        Detect and merge composite symbols like fractions, integrals with limits, etc.
        
        Args:
            symbols: List of segmented symbols
            spatial_threshold: Distance threshold for grouping related symbols
            
        Returns:
            Updated list with composite symbols merged
        """
        if not symbols:
            return symbols
        
        # Detect fractions
        symbols = self._merge_fractions(symbols)
        
        # Detect superscripts and subscripts
        symbols = self._group_scripts(symbols)
        
        # Detect square roots and radicals
        symbols = self._detect_radicals(symbols)
        
        return symbols
    
    def _merge_fractions(self, symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect and merge fraction components (numerator, line, denominator).
        """
        # Find horizontal lines that could be fraction bars
        fraction_lines = [s for s in symbols if s.get('symbol_type') == 'horizontal_line']
        
        merged_symbols = []
        used_indices = set()
        
        for line_symbol in fraction_lines:
            if line_symbol.get('id') in used_indices:
                continue
            
            line_y = line_symbol['centroid'][0]
            line_x = line_symbol['centroid'][1]
            
            # Find potential numerator (above the line)
            numerators = []
            denominators = []
            
            for symbol in symbols:
                if symbol.get('id') in used_indices or symbol == line_symbol:
                    continue
                
                symbol_y = symbol['centroid'][0]
                symbol_x = symbol['centroid'][1]
                
                # Check if symbol is vertically aligned with fraction line
                if abs(symbol_x - line_x) < 50:  # Horizontal alignment threshold
                    if symbol_y < line_y - 10:  # Above the line
                        numerators.append(symbol)
                    elif symbol_y > line_y + 10:  # Below the line
                        denominators.append(symbol)
            
            # If we have both numerator and denominator, create fraction
            if numerators and denominators:
                fraction_data = {
                    'id': f"fraction_{line_symbol.get('id', 0)}",
                    'type': 'fraction',
                    'numerator': numerators,
                    'denominator': denominators,
                    'fraction_line': line_symbol,
                    'bbox': self._calculate_composite_bbox([line_symbol] + numerators + denominators),
                    'centroid': line_symbol['centroid']
                }
                
                merged_symbols.append(fraction_data)
                
                # Mark components as used
                used_indices.add(line_symbol.get('id'))
                for num in numerators:
                    used_indices.add(num.get('id'))
                for den in denominators:
                    used_indices.add(den.get('id'))
        
        # Add remaining symbols that weren't part of fractions
        for symbol in symbols:
            if symbol.get('id') not in used_indices:
                merged_symbols.append(symbol)
        
        return merged_symbols
    
    def _group_scripts(self, symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group superscripts and subscripts with their base symbols.
        """
        # Calculate baseline (median y-coordinate)
        y_coords = [s['centroid'][0] for s in symbols if s.get('symbol_type') == 'symbol']
        if not y_coords:
            return symbols
        
        baseline = np.median(y_coords)
        
        grouped_symbols = []
        used_indices = set()
        
        for symbol in symbols:
            if symbol.get('id') in used_indices:
                continue
            
            if symbol.get('symbol_type') != 'symbol':
                grouped_symbols.append(symbol)
                continue
            
            symbol_y = symbol['centroid'][0]
            symbol_x = symbol['centroid'][1]
            
            # Check if this is a main line symbol
            if abs(symbol_y - baseline) < 15:  # Main line threshold
                # Look for superscripts and subscripts
                superscripts = []
                subscripts = []
                
                for other in symbols:
                    if (other.get('id') in used_indices or 
                        other == symbol or 
                        other.get('symbol_type') != 'symbol'):
                        continue
                    
                    other_y = other['centroid'][0]
                    other_x = other['centroid'][1]
                    
                    # Check if horizontally close
                    if abs(other_x - symbol_x) < 40:
                        if other_y < baseline - 15:  # Above baseline (superscript)
                            superscripts.append(other)
                        elif other_y > baseline + 15:  # Below baseline (subscript)
                            subscripts.append(other)
                
                # Create grouped symbol if scripts exist
                if superscripts or subscripts:
                    grouped_data = {
                        'id': f"scripted_{symbol.get('id', 0)}",
                        'type': 'scripted_symbol',
                        'base': symbol,
                        'superscripts': superscripts,
                        'subscripts': subscripts,
                        'bbox': self._calculate_composite_bbox([symbol] + superscripts + subscripts),
                        'centroid': symbol['centroid']
                    }
                    grouped_symbols.append(grouped_data)
                    
                    # Mark as used
                    used_indices.add(symbol.get('id'))
                    for sup in superscripts:
                        used_indices.add(sup.get('id'))
                    for sub in subscripts:
                        used_indices.add(sub.get('id'))
                else:
                    grouped_symbols.append(symbol)
                    used_indices.add(symbol.get('id'))
            else:
                # Not a main line symbol, will be handled by other symbols
                pass
        
        # Add any remaining unused symbols
        for symbol in symbols:
            if symbol.get('id') not in used_indices:
                grouped_symbols.append(symbol)
        
        return grouped_symbols
    
    def _detect_radicals(self, symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect square root and other radical symbols.
        This is a simplified implementation - in practice would use shape analysis.
        """
        # For now, just return symbols as-is
        # In a full implementation, you would:
        # 1. Identify radical symbols by shape analysis
        # 2. Group them with their radicands
        # 3. Handle index notation for nth roots
        return symbols
    
    def _calculate_composite_bbox(self, components: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box that encompasses all components.
        """
        if not components:
            return (0, 0, 0, 0)
        
        min_rows = []
        min_cols = []
        max_rows = []
        max_cols = []
        
        for comp in components:
            bbox = comp.get('bbox', comp.get('original_bbox', (0, 0, 0, 0)))
            min_rows.append(bbox[0])
            min_cols.append(bbox[1])
            max_rows.append(bbox[2])
            max_cols.append(bbox[3])
        
        return (min(min_rows), min(min_cols), max(max_rows), max(max_cols))
    
    def visualize_segmentation(self, image: np.ndarray, symbols: List[Dict[str, Any]], 
                             save_path: str = None) -> None:
        """
        Visualize the segmentation results with bounding boxes and labels.
        
        Args:
            image: Original image
            symbols: List of segmented symbols
            save_path: Optional path to save the visualization
        """
        # Create colored version of image
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw bounding boxes and labels
        for i, symbol in enumerate(symbols):
            bbox = symbol.get('bbox', symbol.get('original_bbox'))
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Choose color based on symbol type
                color = (0, 255, 0)  # Green for regular symbols
                if symbol.get('type') == 'fraction':
                    color = (255, 0, 0)  # Red for fractions
                elif symbol.get('type') == 'scripted_symbol':
                    color = (0, 0, 255)  # Blue for scripted symbols
                elif symbol.get('symbol_type') == 'horizontal_line':
                    color = (255, 255, 0)  # Yellow for lines
                
                # Draw rectangle
                cv2.rectangle(vis_image, (y1, x1), (y2, x2), color, 2)
                
                # Add label
                label = f"{i}: {symbol.get('type', symbol.get('symbol_type', 'symbol'))}"
                cv2.putText(vis_image, label, (y1, x1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display or save
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        # Show with matplotlib for better display
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Mathematical Expression Segmentation - {len(symbols)} components')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.jpg', '_matplotlib.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_segmentation_statistics(self, symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the segmentation results.
        
        Args:
            symbols: List of segmented symbols
            
        Returns:
            Dictionary containing segmentation statistics
        """
        if not symbols:
            return {
                'total_symbols': 0,
                'symbol_types': {},
                'average_area': 0,
                'size_distribution': {}
            }
        
        # Count symbol types
        type_counts = {}
        areas = []
        
        for symbol in symbols:
            symbol_type = symbol.get('type', symbol.get('symbol_type', 'unknown'))
            type_counts[symbol_type] = type_counts.get(symbol_type, 0) + 1
            
            area = symbol.get('area', 0)
            if area > 0:
                areas.append(area)
        
        # Calculate size distribution
        if areas:
            areas = np.array(areas)
            size_distribution = {
                'small': np.sum(areas < np.percentile(areas, 33)),
                'medium': np.sum((areas >= np.percentile(areas, 33)) & 
                               (areas < np.percentile(areas, 67))),
                'large': np.sum(areas >= np.percentile(areas, 67))
            }
        else:
            size_distribution = {'small': 0, 'medium': 0, 'large': 0}
        
        return {
            'total_symbols': len(symbols),
            'symbol_types': type_counts,
            'average_area': np.mean(areas) if areas else 0,
            'size_distribution': size_distribution,
            'area_stats': {
                'min': np.min(areas) if areas else 0,
                'max': np.max(areas) if areas else 0,
                'median': np.median(areas) if areas else 0,
                'std': np.std(areas) if areas else 0
            }
        }
