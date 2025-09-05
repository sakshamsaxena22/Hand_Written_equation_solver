"""
Image utility functions for mathematical expression processing
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import logging
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from skimage import morphology, filters
from scipy import ndimage

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Advanced image processing utilities for mathematical expressions"""
    
    def __init__(self):
        self.default_size = (224, 224)
        self.symbol_size = (32, 32)
    
    def load_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Load and preprocess image from file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image from {image_path}")
                return None
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if target_size:
                image = cv2.resize(image, target_size)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to standard range"""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Normalize to [0, 1] range
        normalized = image.astype(np.float32) / 255.0
        return normalized
    
    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def remove_noise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Remove noise from image using various methods"""
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        else:
            logger.warning(f"Unknown noise removal method: {method}")
            return image
    
    def binarize_image(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """Binarize image using various thresholding methods"""
        if method == 'adaptive':
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == 'global':
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary
        else:
            logger.warning(f"Unknown binarization method: {method}")
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary
    
    def correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct skew in handwritten text"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image, 0.0
        
        # Get the largest contour (assume it's the main text)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        
        # Correct angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Only correct significant skew
        if abs(angle) < 1:
            return image, 0.0
        
        # Apply rotation
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return corrected, angle
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in image using MSER"""
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(image)
            
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            
            bounding_boxes = []
            for hull in hulls:
                x, y, w, h = cv2.boundingRect(hull)
                # Filter out very small regions
                if w > 10 and h > 10:
                    bounding_boxes.append((x, y, w, h))
            
            return bounding_boxes
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    def crop_to_content(self, image: np.ndarray, padding: int = 10) -> np.ndarray:
        """Crop image to content with optional padding"""
        # Find non-zero pixels
        coords = cv2.findNonZero(image)
        
        if coords is None:
            return image
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
    
    def resize_with_aspect_ratio(self, image: np.ndarray, target_size: Tuple[int, int], 
                               fill_color: int = 255) -> np.ndarray:
        """Resize image maintaining aspect ratio with padding"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas and center the image
        canvas = np.full((target_h, target_w), fill_color, dtype=image.dtype)
        
        # Calculate position to center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def augment_image(self, image: np.ndarray, augmentation_params: Dict[str, Any]) -> np.ndarray:
        """Apply data augmentation to image"""
        augmented = image.copy()
        
        # Rotation
        if 'rotation_range' in augmentation_params:
            angle = np.random.uniform(-augmentation_params['rotation_range'], 
                                    augmentation_params['rotation_range'])
            h, w = augmented.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # Scaling
        if 'scale_range' in augmentation_params:
            scale = np.random.uniform(1 - augmentation_params['scale_range'], 
                                    1 + augmentation_params['scale_range'])
            h, w = augmented.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            augmented = cv2.resize(augmented, (new_w, new_h))
            augmented = self.resize_with_aspect_ratio(augmented, (w, h))
        
        # Translation
        if 'translation_range' in augmentation_params:
            tx = np.random.randint(-augmentation_params['translation_range'], 
                                 augmentation_params['translation_range'])
            ty = np.random.randint(-augmentation_params['translation_range'], 
                                 augmentation_params['translation_range'])
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            h, w = augmented.shape[:2]
            augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # Add noise
        if 'noise_level' in augmentation_params:
            noise = np.random.normal(0, augmentation_params['noise_level'], augmented.shape)
            augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return augmented
    
    def create_image_grid(self, images: List[np.ndarray], grid_size: Tuple[int, int]) -> np.ndarray:
        """Create a grid of images for visualization"""
        rows, cols = grid_size
        if len(images) > rows * cols:
            images = images[:rows * cols]
        
        # Pad with empty images if needed
        while len(images) < rows * cols:
            if images:
                empty_img = np.zeros_like(images[0])
            else:
                empty_img = np.zeros((64, 64), dtype=np.uint8)
            images.append(empty_img)
        
        # Ensure all images are the same size
        img_h, img_w = images[0].shape[:2]
        resized_images = []
        for img in images:
            if img.shape[:2] != (img_h, img_w):
                img = cv2.resize(img, (img_w, img_h))
            resized_images.append(img)
        
        # Create grid
        grid_rows = []
        for i in range(0, len(resized_images), cols):
            row_images = resized_images[i:i + cols]
            if len(row_images) == cols:
                row = np.hstack(row_images)
                grid_rows.append(row)
        
        if grid_rows:
            grid = np.vstack(grid_rows)
            return grid
        else:
            return np.zeros((img_h * rows, img_w * cols), dtype=np.uint8)
    
    def calculate_image_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate various image quality metrics"""
        metrics = {}
        
        # Basic statistics
        metrics['mean'] = np.mean(image)
        metrics['std'] = np.std(image)
        metrics['min'] = np.min(image)
        metrics['max'] = np.max(image)
        
        # Contrast metrics
        metrics['contrast'] = np.std(image) / np.mean(image) if np.mean(image) > 0 else 0
        
        # Sharpness (using Laplacian variance)
        if len(image.shape) == 2:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            metrics['sharpness'] = np.var(laplacian)
        
        # Entropy (measure of information content)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        metrics['entropy'] = entropy
        
        return metrics
