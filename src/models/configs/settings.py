# File: config/settings.py
"""
Configuration settings
"""

# Model configuration
MODEL_CONFIG = {
    'transformer_ocr': {
        'vocab_size': 1000,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6
    },
    'vision_encoder': {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12
    }
}

# Training configuration
TRAINING_CONFIG = {
    'datasets': [
        'CROHME-2019',
        'Im2LaTeX-100K',
        'Custom-Math-Expressions'
    ],
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 100,
        'warmup_steps': 1000,
        'weight_decay': 0.01
    }
}

# Deployment configuration
DEPLOYMENT_CONFIG = {
    'model_path': './models/',
    'temp_dir': './temp/',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp'],
    'batch_size_limit': 10,
    'rate_limiting': {
        'requests_per_minute': 60,
        'requests_per_hour': 1000
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
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
            'propagate': False
        }
    }
}
