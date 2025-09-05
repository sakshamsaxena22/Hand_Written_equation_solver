"""
Configuration settings
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'src' / 'models' / 'pretrained'
LOGS_DIR = BASE_DIR / 'logs'
TEMP_DIR = BASE_DIR / 'temp'

# Model configuration
MODEL_CONFIG = {
    'transformer_ocr': {
        'vocab_size': 222,  # Updated to match actual vocabulary
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'input_size': (64, 64),  # Standard symbol size
        'pretrained_path': MODELS_DIR / 'ocr_transformer.pth',
    },
    'vision_encoder': {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'image_size': (224, 224),
        'num_classes': 222,
    },
    'preprocessing': {
        'image_size': (224, 224),
        'symbol_size': (64, 64),
        'noise_reduction': True,
        'skew_correction': True,
        'contrast_enhancement': True,
    },
    'segmentation': {
        'min_symbol_area': 50,
        'max_symbol_area': 5000,
        'symbol_height_range': (10, 100),
        'symbol_width_range': (5, 80),
        'grouping_threshold': 20,
    },
}

# Training configuration
TRAINING_CONFIG = {
    'datasets': {
        'synthetic': {
            'enabled': True,
            'samples_per_symbol': 1000,
            'augmentation': {
                'rotation_range': 15,
                'scale_range': 0.1,
                'translation_range': 5,
                'noise_level': 0.05,
            },
        },
        'real_data': {
            'paths': [
                DATA_DIR / 'CROHME-2019',
                DATA_DIR / 'Im2LaTeX-100K',
                DATA_DIR / 'custom',
            ],
            'formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
        },
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 100,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'val_split': 0.2,
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001,
        },
        'scheduler': {
            'type': 'StepLR',
            'step_size': 10,
            'gamma': 0.1,
        },
        'optimizer': {
            'type': 'Adam',
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        },
    },
    'validation': {
        'frequency': 1,  # Every epoch
        'metrics': ['accuracy', 'top_k_accuracy', 'confusion_matrix'],
        'save_best_only': True,
        'save_frequency': 10,  # Save checkpoint every 10 epochs
    },
}

# Processing configuration
PROCESSING_CONFIG = {
    'context_recognition': {
        'enabled': True,
        'confidence_threshold': 0.8,
        'context_types': ['physics', 'engineering', 'pure_math', 'statistics', 'geometry', 'calculus'],
    },
    'expression_parsing': {
        'max_recursion_depth': 100,
        'operator_precedence': {
            '+': 1, '-': 1, '*': 2, '/': 2, '^': 3,
            'sin': 4, 'cos': 4, 'tan': 4, 'log': 4, 'ln': 4, 'sqrt': 4,
            'integral': 5, 'sum': 5, 'lim': 5, 'partial': 5
        },
    },
    'equation_solving': {
        'timeout_seconds': 30,
        'max_solutions': 10,
        'numerical_precision': 10,
        'symbolic_simplification': True,
    },
}

# API and Web Interface configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'cors': {
        'enabled': True,
        'origins': ['http://localhost:3000', 'http://127.0.0.1:3000'],
    },
    'authentication': {
        'enabled': False,
        'jwt_secret': os.environ.get('JWT_SECRET', 'your-secret-key'),
        'token_expiry': 3600,  # 1 hour
    },
}

# Deployment configuration
DEPLOYMENT_CONFIG = {
    'model_path': MODELS_DIR,
    'temp_dir': TEMP_DIR,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf'],
    'batch_size_limit': 10,
    'rate_limiting': {
        'requests_per_minute': 60,
        'requests_per_hour': 1000,
        'burst_limit': 10,
    },
    'caching': {
        'enabled': True,
        'cache_size': 1000,
        'ttl_seconds': 3600,
    },
    'monitoring': {
        'enabled': True,
        'metrics': ['response_time', 'accuracy', 'error_rate'],
        'alerts': {
            'error_rate_threshold': 0.05,
            'response_time_threshold': 5.0,
        },
    },
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'equation_solver.log',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
