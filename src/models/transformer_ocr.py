# File: src/models/transformer_ocr.py
"""
Transformer-based OCR model for mathematical expressions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

class MathTransformerOCR(nn.Module):
    """Transformer-based OCR for mathematical expressions"""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        
        # Vision encoder for mathematical symbols
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.feature_projection = nn.Linear(256 * 64, d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, vocab_size)
        
        logger = logging.getLogger(__name__)
        
        # Load vocabulary from file or use default
        self.vocab = self._load_vocabulary()
        self.idx_to_symbol = {v: k for k, v in self.vocab.items()}
        
        # Update vocab_size to match actual vocabulary
        if vocab_size != len(self.vocab):
            logger.warning(f"vocab_size ({vocab_size}) != actual vocab size ({len(self.vocab)}). Using actual size.")
            vocab_size = len(self.vocab)
            
        # Update classifier layer
        self.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Extract visual features
        features = self.conv_features(x)  # [batch, 256, 8, 8]
        features = features.view(batch_size, 256, -1).permute(0, 2, 1)  # [batch, 64, 256]
        features = features.view(batch_size, -1, 256)  # [batch, 64, 256]
        features = self.feature_projection(features)  # [batch, 64, d_model]
        
        # Apply transformer
        output = self.transformer(features)
        
        # Classification
        logits = self.classifier(output.mean(dim=1))  # Global average pooling
        
        return logits
    
    def predict_symbol(self, image: np.ndarray) -> str:
        """Predict mathematical symbol from image"""
        self.eval()
        with torch.no_grad():
            # Preprocess image
            if len(image.shape) == 2:
                image = cv2.resize(image, (64, 64))
                image = image.astype(np.float32) / 255.0
                image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            
            # Predict
            logits = self.forward(image)
            prediction = torch.argmax(logits, dim=1).item()
            
            return self.idx_to_symbol.get(prediction, '?')
    
    def _load_vocabulary(self) -> Dict[str, int]:
        """Load mathematical symbol vocabulary"""
        try:
            vocab_path = Path(__file__).parent.parent.parent / 'data' / 'vocabularies' / 'symbols.json'
            if vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not load vocabulary file: {e}")
        
        # Default comprehensive vocabulary
        return {
            # Numbers
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            
            # Basic operations
            '+': 10, '-': 11, '*': 12, '×': 13, '·': 14, '/': 15, '÷': 16, '=': 17,
            '≠': 18, '<': 19, '>': 20, '≤': 21, '≥': 22, '≈': 23, '∝': 24,
            
            # Brackets and parentheses
            '(': 25, ')': 26, '[': 27, ']': 28, '{': 29, '}': 30, '|': 31,
            
            # Variables and constants
            'a': 32, 'b': 33, 'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39,
            'i': 40, 'j': 41, 'k': 42, 'l': 43, 'm': 44, 'n': 45, 'o': 46, 'p': 47,
            'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54, 'x': 55,
            'y': 56, 'z': 57,
            
            # Greek letters (common in math)
            'α': 58, 'β': 59, 'γ': 60, 'δ': 61, 'ε': 62, 'ζ': 63, 'η': 64, 'θ': 65,
            'ι': 66, 'κ': 67, 'λ': 68, 'μ': 69, 'ν': 70, 'ξ': 71, 'ο': 72, 'π': 73,
            'ρ': 74, 'σ': 75, 'τ': 76, 'υ': 77, 'φ': 78, 'χ': 79, 'ψ': 80, 'ω': 81,
            
            # Powers and indices
            '^': 82, '²': 83, '³': 84, '₀': 85, '₁': 86, '₂': 87, '₃': 88, '₄': 89,
            '₅': 90, '₆': 91, '₇': 92, '₈': 93, '₉': 94,
            
            # Functions
            'sin': 95, 'cos': 96, 'tan': 97, 'cot': 98, 'sec': 99, 'csc': 100,
            'arcsin': 101, 'arccos': 102, 'arctan': 103, 
            'sinh': 104, 'cosh': 105, 'tanh': 106,
            'log': 107, 'ln': 108, 'lg': 109, 'exp': 110,
            
            # Calculus
            '∫': 111, '∑': 112, '∏': 113, '∂': 114, '∇': 115, '∆': 116, 'lim': 117,
            '∞': 118, 'd': 119, 'dx': 120, 'dy': 121, 'dz': 122, 'dt': 123,
            
            # Roots and radicals
            '√': 124, '∛': 125, '∜': 126,
            
            # Set theory and logic
            '∈': 127, '∉': 128, '⊂': 129, '⊆': 130, '⊃': 131, '⊇': 132,
            '∪': 133, '∩': 134, '∅': 135, '∀': 136, '∃': 137, '¬': 138,
            '∧': 139, '∨': 140, '→': 141, '↔': 142,
            
            # Special symbols
            '±': 143, '∓': 144, '‰': 145, '%': 146, '°': 147, "'": 148, '"': 149,
            ',': 150, '.': 151, ':': 152, ';': 153,
            
            # Fractions
            '½': 154, '⅓': 155, '¼': 156, '¾': 157, '⅕': 158, '⅖': 159, '⅗': 160,
            '⅘': 161, '⅙': 162, '⅚': 163, '⅛': 164, '⅜': 165, '⅝': 166, '⅞': 167,
            
            # Matrices and vectors
            '→': 168, '←': 169, '↑': 170, '↓': 171, '⃗': 172,
            
            # Unknown symbol
            '<UNK>': 173
        }
    
    def predict_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict mathematical symbol with confidence score"""
        self.eval()
        with torch.no_grad():
            # Preprocess image
            if len(image.shape) == 2:
                image = cv2.resize(image, (64, 64))
                image = image.astype(np.float32) / 255.0
                image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            
            # Predict
            logits = self.forward(image)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
            
            symbol = self.idx_to_symbol.get(prediction, '<UNK>')
            return symbol, confidence
    
    def predict_top_k(self, image: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k predictions with confidence scores"""
        self.eval()
        with torch.no_grad():
            # Preprocess image
            if len(image.shape) == 2:
                image = cv2.resize(image, (64, 64))
                image = image.astype(np.float32) / 255.0
                image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            
            # Predict
            logits = self.forward(image)
            probabilities = F.softmax(logits, dim=1)
            
            # Get top-k
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)
            
            results = []
            for i in range(k):
                idx = top_k_indices[0, i].item()
                prob = top_k_probs[0, i].item()
                symbol = self.idx_to_symbol.get(idx, '<UNK>')
                results.append((symbol, prob))
            
            return results
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab': self.vocab,
            'model_config': {
                'vocab_size': len(self.vocab),
                'd_model': self.d_model
            }
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'vocab' in checkpoint:
            self.vocab = checkpoint['vocab']
            self.idx_to_symbol = {v: k for k, v in self.vocab.items()}
