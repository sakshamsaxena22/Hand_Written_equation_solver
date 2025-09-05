"""
Create Test Images for Equation Solver
======================================

This script creates sample test images with different equations
to test the web interface functionality.
"""

import cv2
import numpy as np
import os

def create_test_equation_image(equation_text: str, filename: str):
    """Create a test image with equation text"""
    # Create white background
    image = np.ones((200, 400), dtype=np.uint8) * 255
    
    # Add equation text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Calculate text size and position
    text_size = cv2.getTextSize(equation_text, font, font_scale, thickness)[0]
    x = (image.shape[1] - text_size[0]) // 2
    y = (image.shape[0] + text_size[1]) // 2
    
    # Add the text
    cv2.putText(image, equation_text, (x, y), font, font_scale, (0, 0, 0), thickness)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.addWeighted(image, 0.95, noise, 0.05, 0)
    
    # Save the image
    cv2.imwrite(filename, image)
    print(f"Created test image: {filename}")

def main():
    """Create multiple test images"""
    
    # Create test_images directory
    os.makedirs("test_images", exist_ok=True)
    
    # Test equations
    test_equations = [
        ("3x + 2y = 12", "test_images/equation1.png"),
        ("x^2 - 4x + 4 = 0", "test_images/equation2.png"),
        ("2x + 5 = 13", "test_images/equation3.png"),
        ("5x - 7 = 23", "test_images/equation4.png"),
        ("x + y = 10", "test_images/equation5.png")
    ]
    
    print("Creating test images...")
    
    for equation, filename in test_equations:
        create_test_equation_image(equation, filename)
    
    print(f"\nCreated {len(test_equations)} test images in 'test_images' directory")
    print("You can use these images to test the web interface")

if __name__ == '__main__':
    main()
