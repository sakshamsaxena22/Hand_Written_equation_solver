"""
Advanced Image Preprocessing for Handwritten Mathematical Expressions
====================================================================

This module provides sophisticated image preprocessing capabilities specifically
designed for handwritten mathematical equations, including noise removal,
skew correction, contrast enhancement, and normalization."""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import logging
from scipy import ndimage
try:
    from skimage import filters, morphology, measure
    from skimage.transform import rotate
except ImportError:
    logger.warning("scikit-image not available, some features may be limited")
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class AdvancedImagePreprocessor:
    """
    Advanced image preprocessor for handwritten mathematical expressions
    with specialized techniques for character recognition and equation solving.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target size for processed images
        """
        self.target_size = target_size
        self.processing_stats = {}
        
        # Set up augmentation transforms for training
        self.transform = A.Compose([
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])
        
    def preprocess(self, image: np.ndarray, preserve_aspect: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for handwritten equations.
        
        Args:
            image: Input image (BGR or grayscale)
            preserve_aspect: Whether to preserve aspect ratio during resizing
            
        Returns:
            Preprocessed image ready for character recognition
        """
        logger.info("Starting image preprocessing pipeline...")
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                processed = image.copy()
            
            # Store original dimensions
            self.processing_stats['original_shape'] = processed.shape
            
            # 1. Noise reduction
            processed = self._denoise(processed)
            
            # 2. Skew correction
            processed = self._correct_skew(processed)
            
            # 3. Contrast enhancement
            processed = self._enhance_contrast(processed)
            
            # 4. Binarization with adaptive thresholding
            processed = self._adaptive_binarization(processed)
            
            # 5. Morphological operations
            processed = self._morphological_cleanup(processed)
            
            # 6. Resize while preserving aspect ratio
            processed = self._smart_resize(processed, preserve_aspect)
            
            # 7. Final normalization
            processed = self._normalize_image(processed)
            
            logger.info("Image preprocessing completed successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            raise
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced noise reduction using multiple techniques.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        # Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Non-local means denoising for better preservation of edges
        denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        
        # Bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        return denoised
    
    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in handwritten equations.
        
        Args:
            image: Input image
            
        Returns:
            Skew-corrected image
        """
        # Create binary image for skew detection
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find contours to detect text regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (assumed to be main equation)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Correct the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Only correct if skew is significant (> 0.5 degrees)
        if abs(angle) > 0.5:
            # Get image center
            h, w = image.shape
            center = (w // 2, h // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                     borderMode=cv2.BORDER_REPLICATE)
            
            self.processing_stats['skew_angle'] = angle
            logger.info(f"Corrected skew by {angle:.2f} degrees")
            
            return corrected
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using multiple techniques.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Gamma correction for better visibility
        gamma = self._estimate_gamma(image)
        enhanced = self._apply_gamma_correction(enhanced, gamma)
        
        # Enhance using histogram stretching
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        
        self.processing_stats['gamma'] = gamma
        
        return enhanced
    
    def _estimate_gamma(self, image: np.ndarray) -> float:
        """
        Estimate optimal gamma for gamma correction.
        
        Args:
            image: Input image
            
        Returns:
            Estimated gamma value
        """
        # Calculate mean intensity
        mean_intensity = np.mean(image) / 255.0
        
        # Estimate gamma based on mean intensity
        if mean_intensity < 0.3:
            return 0.7  # Brighten dark images
        elif mean_intensity > 0.7:
            return 1.3  # Darken bright images
        else:
            return 1.0  # No correction needed
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to image.
        
        Args:
            image: Input image
            gamma: Gamma value
            
        Returns:
            Gamma-corrected image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def _adaptive_binarization(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced binarization using adaptive techniques.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image
        """
        # Method 1: Otsu's thresholding
        _, binary1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        binary2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 2)
        
        # Method 3: Sauvola's method for handwritten text
        binary3 = self._sauvola_threshold(image)
        
        # Combine methods by taking the intersection (most restrictive)
        combined = cv2.bitwise_and(cv2.bitwise_and(binary1, binary2), binary3)
        
        # If combination is too restrictive, use adaptive threshold
        white_pixels = np.sum(combined == 255)
        total_pixels = combined.shape[0] * combined.shape[1]
        
        if white_pixels / total_pixels < 0.1:  # Too little content
            return binary2
        
        return combined
    
    def _sauvola_threshold(self, image: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
        """
        Sauvola's adaptive thresholding method, good for handwritten text.
        
        Args:
            image: Input grayscale image
            window_size: Size of local window
            k: Parameter controlling threshold adaptation
            
        Returns:
            Binary image
        """
        # Convert to float
        img_float = image.astype(np.float64)
        
        # Calculate local mean and standard deviation
        mean = cv2.boxFilter(img_float, -1, (window_size, window_size), normalize=True)
        sqmean = cv2.boxFilter(img_float**2, -1, (window_size, window_size), normalize=True)
        std = np.sqrt(sqmean - mean**2)
        
        # Sauvola threshold
        R = 128  # Dynamic range of standard deviation
        threshold = mean * (1 + k * (std / R - 1))
        
        # Apply threshold
        binary = np.where(img_float > threshold, 255, 0).astype(np.uint8)
        
        return binary
    
    def _morphological_cleanup(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Clean up binary image using morphological operations.
        
        Args:
            binary_image: Input binary image
            
        Returns:
            Cleaned binary image
        """
        # Remove small noise components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPENING, kernel, iterations=1)
        
        # Close small gaps in characters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small connected components
        cleaned = self._remove_small_components(cleaned, min_size=10)
        
        return cleaned
    
    def _remove_small_components(self, binary_image: np.ndarray, min_size: int = 10) -> np.ndarray:
        """
        Remove small connected components from binary image.
        
        Args:
            binary_image: Input binary image
            min_size: Minimum component size to keep
            
        Returns:
            Cleaned binary image
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        # Create output image
        output = np.zeros_like(binary_image)
        
        # Keep only components larger than min_size
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output[labels == i] = 255
        
        return output
    
    def _smart_resize(self, image: np.ndarray, preserve_aspect: bool = True) -> np.ndarray:
        """
        Intelligently resize image while preserving important features.
        
        Args:
            image: Input image
            preserve_aspect: Whether to preserve aspect ratio
            
        Returns:
            Resized image
        """
        h, w = image.shape
        target_h, target_w = self.target_size
        
        if preserve_aspect:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            
            # New dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas and center the image
            canvas = np.zeros((target_h, target_w), dtype=image.dtype)
            
            # Calculate padding
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image in center
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return canvas
        else:
            # Direct resize without preserving aspect ratio
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Final normalization of the processed image.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Ensure values are in [0, 255] range
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert to float32 and normalize to [0, 1]
        normalized = normalized.astype(np.float32) / 255.0
        
        return normalized
    
    def preprocess_for_character_recognition(self, image: np.ndarray, char_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Specialized preprocessing for individual character recognition.
        
        Args:
            image: Input character image
            char_size: Target size for character
            
        Returns:
            Preprocessed character image
        """
        # Basic preprocessing
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image.copy()
        
        # Denoise
        processed = cv2.medianBlur(processed, 3)
        
        # Normalize intensity
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        
        # Binarize
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Resize to target size
        processed = cv2.resize(processed, char_size, interpolation=cv2.INTER_AREA)
        
        # Final normalization to [0, 1]
        processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def _enhance_math_symbols(self, image: np.ndarray) -> np.ndarray:
        """Enhance mathematical symbols for better recognition"""
        # Morphological operations to clean up symbols
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Edge enhancement
        edges = cv2.Canny(cleaned, 50, 150, apertureSize=3)
        enhanced = cv2.bitwise_or(cleaned, edges)
        
        return enhanced
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the last preprocessing operation.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.processing_stats.copy()
    
    def visualize_preprocessing_steps(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Visualize the preprocessing steps for debugging and analysis.
        
        Args:
            image: Input image
            save_path: Optional path to save the visualization
        """
        # Store original image
        original = image.copy()
        
        # Apply each step and store results
        steps = {}
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        steps['1. Original'] = original
        steps['2. Grayscale'] = gray
        
        # Denoise
        denoised = self._denoise(gray)
        steps['3. Denoised'] = denoised
        
        # Skew correction
        skew_corrected = self._correct_skew(denoised)
        steps['4. Skew Corrected'] = skew_corrected
        
        # Contrast enhancement
        enhanced = self._enhance_contrast(skew_corrected)
        steps['5. Enhanced Contrast'] = enhanced
        
        # Binarization
        binary = self._adaptive_binarization(enhanced)
        steps['6. Binarized'] = binary
        
        # Morphological cleanup
        cleaned = self._morphological_cleanup(binary)
        steps['7. Cleaned'] = cleaned
        
        # Resize
        resized = self._smart_resize(cleaned)
        steps['8. Resized'] = resized
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (title, img) in enumerate(steps.items()):
            if i < len(axes):
                axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[i].set_title(title)
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


