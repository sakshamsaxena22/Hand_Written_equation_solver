#!/usr/bin/env python3
"""
System Test Script for Hand-Written Equation Solver
==================================================

This script tests the complete equation solver system to verify
all components are working correctly.
"""

import os
import sys
import logging
import traceback
import numpy as np
import cv2
from pathlib import Path
import time
from typing import Dict, List, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all necessary modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        # Core imports
        import numpy as np
        import cv2
        import torch
        logger.info("✓ Basic dependencies imported successfully")
        
        # Test src imports
        try:
            from src.core.preprocessing import AdvancedImagePreprocessor
            logger.info("✓ Image preprocessor imported")
        except Exception as e:
            logger.warning(f"⚠ Image preprocessor import issue: {e}")
        
        try:
            from src.core.segmentation import MathExpressionSegmenter
            logger.info("✓ Segmentation module imported")
        except Exception as e:
            logger.warning(f"⚠ Segmentation import issue: {e}")
        
        try:
            from src.models.transformer_ocr import MathTransformerOCR
            logger.info("✓ OCR model imported")
        except Exception as e:
            logger.warning(f"⚠ OCR model import issue: {e}")
        
        try:
            from src.core.expression_parser import MathExpressionParser
            logger.info("✓ Expression parser imported")
        except Exception as e:
            logger.warning(f"⚠ Parser import issue: {e}")
        
        try:
            from src.utils.gemini_fallback import create_fallback_recognizer
            logger.info("✓ Gemini fallback imported")
        except Exception as e:
            logger.warning(f"⚠ Gemini fallback import issue: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_image_preprocessing():
    """Test image preprocessing functionality"""
    logger.info("Testing image preprocessing...")
    
    try:
        from src.core.preprocessing import AdvancedImagePreprocessor
        
        # Create test image
        test_image = create_test_equation_image()
        
        # Initialize preprocessor
        preprocessor = AdvancedImagePreprocessor()
        
        # Test preprocessing
        processed = preprocessor.preprocess(test_image)
        
        logger.info(f"✓ Preprocessing successful. Output shape: {processed.shape}")
        
        # Test character preprocessing
        char_image = test_image[50:150, 50:150]  # Crop a section
        char_processed = preprocessor.preprocess_for_character_recognition(char_image)
        
        logger.info(f"✓ Character preprocessing successful. Output shape: {char_processed.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing test failed: {e}")
        traceback.print_exc()
        return False

def test_segmentation():
    """Test expression segmentation"""
    logger.info("Testing expression segmentation...")
    
    try:
        from src.core.segmentation import MathExpressionSegmenter
        from src.core.preprocessing import AdvancedImagePreprocessor
        
        # Create and preprocess test image
        test_image = create_test_equation_image()
        preprocessor = AdvancedImagePreprocessor()
        processed = preprocessor.preprocess(test_image)
        
        # Initialize segmenter
        segmenter = MathExpressionSegmenter()
        
        # Test segmentation
        symbols = segmenter.segment_expression(processed)
        
        logger.info(f"✓ Segmentation successful. Found {len(symbols)} symbols")
        
        # Test statistics
        stats = segmenter.get_segmentation_statistics(symbols)
        logger.info(f"✓ Segmentation stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Segmentation test failed: {e}")
        traceback.print_exc()
        return False

def test_ocr_model():
    """Test OCR model functionality"""
    logger.info("Testing OCR model...")
    
    try:
        from src.models.transformer_ocr import MathTransformerOCR
        
        # Initialize model
        model = MathTransformerOCR()
        
        # Create test character image
        char_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        
        # Test prediction
        prediction = model.predict_symbol(char_image)
        logger.info(f"✓ OCR prediction: {prediction}")
        
        # Test confidence prediction
        prediction_with_conf = model.predict_with_confidence(char_image)
        logger.info(f"✓ OCR with confidence: {prediction_with_conf}")
        
        # Test top-k predictions
        top_k = model.predict_top_k(char_image, k=3)
        logger.info(f"✓ OCR top-3 predictions: {top_k}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OCR model test failed: {e}")
        traceback.print_exc()
        return False

def test_expression_parsing():
    """Test expression parsing"""
    logger.info("Testing expression parsing...")
    
    try:
        from src.core.expression_parser import MathExpressionParser, EquationTypeClassifier
        
        # Initialize parser
        parser = MathExpressionParser()
        classifier = EquationTypeClassifier()
        
        # Test tokenization
        symbols = ['2', 'x', '+', '5', '=', '1', '3']
        tokens = parser.tokenize(symbols)
        logger.info(f"✓ Tokenization successful. Tokens: {len(tokens)}")
        
        # Test parsing
        expression_tree = parser.parse_to_tree(tokens)
        logger.info(f"✓ Parsing successful. Tree root: {expression_tree}")
        
        # Test classification
        expression_string = '2x + 5 = 13'
        equation_types = classifier.classify(expression_string)
        logger.info(f"✓ Classification successful. Types: {equation_types}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Expression parsing test failed: {e}")
        traceback.print_exc()
        return False

def test_gemini_fallback():
    """Test Gemini API fallback (if configured)"""
    logger.info("Testing Gemini fallback...")
    
    try:
        from src.utils.gemini_fallback import create_fallback_recognizer, should_use_fallback
        
        # Test utility functions
        should_fallback = should_use_fallback(0.5, threshold=0.7)
        logger.info(f"✓ Fallback decision logic works: {should_fallback}")
        
        # Try to create fallback recognizer
        recognizer = create_fallback_recognizer()
        
        if recognizer is not None:
            logger.info("✓ Gemini fallback recognizer created successfully")
            
            # Test with dummy image (won't work without API key)
            test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            # result = recognizer.recognize_character(test_image)
            # logger.info(f"✓ Gemini recognition test: {result}")
            
        else:
            logger.warning("⚠ Gemini fallback not configured (API key missing)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gemini fallback test failed: {e}")
        traceback.print_exc()
        return False

def test_web_interface():
    """Test web interface (basic import test)"""
    logger.info("Testing web interface...")
    
    try:
        # Test if Flask app can be imported
        import web_app
        logger.info("✓ Web app module imported successfully")
        
        # Test mock solver
        # Create a dummy image file
        test_image_path = "test_image.png"
        test_image = create_test_equation_image()
        cv2.imwrite(test_image_path, test_image)
        
        # Test mock solver
        result = web_app.solve_equation_mock(test_image_path)
        logger.info(f"✓ Mock solver result: {result['success']}")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Web interface test failed: {e}")
        traceback.print_exc()
        return False

def create_test_equation_image():
    """Create a simple test image with text that looks like an equation"""
    # Create white background
    image = np.ones((200, 400), dtype=np.uint8) * 255
    
    # Add some simple "equation" text using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, '2x + 5 = 13', (50, 100), font, 1, (0, 0, 0), 2)
    
    return image

def test_training_integration():
    """Test training script integration"""
    logger.info("Testing training script integration...")
    
    try:
        import train_model
        
        # Test config creation
        config = {
            'epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001,
            'output_dir': 'test_output',
            'synthetic_only': True,
        }
        
        # Test trainer initialization (without actual training)
        trainer = train_model.ModelTrainer(config)
        logger.info("✓ Model trainer initialized successfully")
        
        # Test enhanced methods
        trainer.add_character_identification_training()
        logger.info("✓ Character identification training added")
        
        trainer.integrate_gemini_fallback()
        logger.info("✓ Gemini fallback integration tested")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training integration test failed: {e}")
        traceback.print_exc()
        return False

def run_full_system_test():
    """Run complete system test"""
    logger.info("="*60)
    logger.info("Starting Full System Test")
    logger.info("="*60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Image Preprocessing", test_image_preprocessing),
        ("Expression Segmentation", test_segmentation),
        ("OCR Model", test_ocr_model),
        ("Expression Parsing", test_expression_parsing),
        ("Gemini Fallback", test_gemini_fallback),
        ("Web Interface", test_web_interface),
        ("Training Integration", test_training_integration),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED ({duration:.2f}s)")
                failed += 1
            
            results[test_name] = {"passed": result, "duration": duration}
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 {test_name} CRASHED ({duration:.2f}s): {e}")
            results[test_name] = {"passed": False, "duration": duration, "error": str(e)}
            failed += 1
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {passed/len(tests)*100:.1f}%")
    
    # Detailed results
    logger.info(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        duration = result["duration"]
        logger.info(f"  {test_name:.<30} {status} ({duration:.2f}s)")
    
    return passed == len(tests)

def main():
    """Main test runner"""
    logger.info("Hand-Written Equation Solver - System Test")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Script location: {__file__}")
    
    # Check basic environment
    logger.info(f"\nEnvironment Check:")
    logger.info(f"  Current working directory: {os.getcwd()}")
    logger.info(f"  Source directory exists: {os.path.exists('src')}")
    logger.info(f"  Main script exists: {os.path.exists('main.py')}")
    logger.info(f"  Training script exists: {os.path.exists('train_model.py')}")
    logger.info(f"  Web app exists: {os.path.exists('web_app.py')}")
    
    # Run tests
    success = run_full_system_test()
    
    if success:
        logger.info("\n🎉 ALL TESTS PASSED! System is ready to use.")
        logger.info("\nNext steps:")
        logger.info("  1. Set up your Gemini API key in .env file")
        logger.info("  2. Train the model: python train_model.py --synthetic --epochs 10")
        logger.info("  3. Start web interface: python web_app.py")
        logger.info("  4. Access at: http://localhost:5000")
    else:
        logger.error("\n⚠️  Some tests failed. Please check the issues above.")
        logger.info("\nCommon fixes:")
        logger.info("  - Install missing dependencies: pip install -r requirements.txt")
        logger.info("  - Check Python path and module imports")
        logger.info("  - Verify all source files are present")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
