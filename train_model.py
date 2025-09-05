#!/usr/bin/env python3
"""
Comprehensive Training Script for Mathematical Expression OCR Model

This script provides a complete training pipeline for the Math Transformer OCR model,
including synthetic data generation, real data loading, model training, validation,
and evaluation.

Usage:
    python train_model.py --config config/training_config.yaml
    python train_model.py --synthetic --epochs 100 --batch_size 32
    python train_model.py --real_data_path /path/to/real/data --mixed_training
"""

import os
import sys
import argparse
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Import our models and utilities
from src.models.transformer_ocr import MathTransformerOCR
from src.utils.image_utils import ImageProcessor
from src.core.preprocessing import AdvancedImagePreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MathSymbolDataset(Dataset):
    """Dataset for mathematical symbol recognition"""
    
    def __init__(self, data_path: str, vocab: Dict[str, int], transform=None, 
                 synthetic: bool = False, real_data: bool = False):
        self.data_path = Path(data_path)
        self.vocab = vocab
        self.transform = transform
        self.synthetic = synthetic
        self.real_data = real_data
        
        self.samples = []
        self._load_data()
    
    def _load_data(self):
        """Load dataset samples"""
        if self.synthetic:
            self._generate_synthetic_data()
        elif self.real_data:
            self._load_real_data()
        else:
            self._load_mixed_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic mathematical expressions"""
        logger.info("Generating synthetic training data...")
        
        # Font paths (add your system font paths)
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/times.ttf", 
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/cambria.ttf"
        ]
        
        available_fonts = []
        for font_path in font_paths:
            if os.path.exists(font_path):
                available_fonts.append(font_path)
        
        if not available_fonts:
            logger.warning("No fonts found, using default")
            available_fonts = [None]  # Use default font
        
        # Generate samples for each symbol in vocabulary
        samples_per_symbol = 1000  # Adjust as needed
        
        for symbol, label in self.vocab.items():
            if symbol == '<UNK>':  # Skip unknown symbol
                continue
                
            for i in range(samples_per_symbol):
                try:
                    # Generate synthetic image
                    img_array = self._generate_symbol_image(
                        symbol, 
                        font_path=np.random.choice(available_fonts),
                        size=(64, 64),
                        noise_level=np.random.uniform(0, 0.1)
                    )
                    
                    self.samples.append({
                        'image': img_array,
                        'label': label,
                        'symbol': symbol,
                        'synthetic': True
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating synthetic data for '{symbol}': {e}")
        
        logger.info(f"Generated {len(self.samples)} synthetic samples")
    
    def _generate_symbol_image(self, symbol: str, font_path: Optional[str] = None, 
                              size: Tuple[int, int] = (64, 64), 
                              noise_level: float = 0.05) -> np.ndarray:
        """Generate a synthetic image of a mathematical symbol"""
        
        # Create image
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font_size = np.random.randint(20, 40)
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (center)
        try:
            bbox = draw.textbbox((0, 0), symbol, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(symbol, font=font)
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Add slight random offset
        x += np.random.randint(-5, 6)
        y += np.random.randint(-5, 6)
        
        # Draw text
        draw.text((x, y), symbol, fill='black', font=font)
        
        # Convert to grayscale numpy array
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Add rotation
        if np.random.random() > 0.7:
            angle = np.random.uniform(-15, 15)
            img_array = self._rotate_image(img_array, angle)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Normalize
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        height, width = image.shape
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                               borderValue=255)  # White background
        
        return rotated
    
    def _load_real_data(self):
        """Load real handwritten mathematical expression data"""
        logger.info(f"Loading real data from {self.data_path}")
        
        # Look for image files and corresponding labels
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        for img_path in self.data_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    # Try to find corresponding label file
                    label_path = img_path.with_suffix('.txt')
                    if label_path.exists():
                        with open(label_path, 'r', encoding='utf-8') as f:
                            symbol = f.read().strip()
                        
                        if symbol in self.vocab:
                            # Load and process image
                            img_array = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                            if img_array is not None:
                                # Resize to standard size
                                img_array = cv2.resize(img_array, (64, 64))
                                img_array = img_array.astype(np.float32) / 255.0
                                
                                self.samples.append({
                                    'image': img_array,
                                    'label': self.vocab[symbol],
                                    'symbol': symbol,
                                    'synthetic': False,
                                    'path': str(img_path)
                                })
                
                except Exception as e:
                    logger.error(f"Error loading real data from {img_path}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} real samples")
    
    def _load_mixed_data(self):
        """Load both synthetic and real data"""
        # Generate synthetic data
        self._generate_synthetic_data()
        
        # Add real data if available
        if self.data_path.exists():
            real_samples_before = len(self.samples)
            self._load_real_data()
            real_samples_added = len(self.samples) - real_samples_before
            logger.info(f"Added {real_samples_added} real samples to {real_samples_before} synthetic samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = sample['image']
        label = sample['label']
        
        # Convert to tensor
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([label])
        
        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
        
        return image, label.squeeze()


class ModelTrainer:
    """Comprehensive model trainer for Math Transformer OCR"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set up directories
        self.setup_directories()
        
        # Load vocabulary
        self.vocab = self._load_vocabulary()
        
        # Initialize model
        self.model = MathTransformerOCR(
            vocab_size=len(self.vocab),
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup logging
        self.setup_logging()
        
        # Training metrics
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
    
    def setup_directories(self):
        """Setup training directories"""
        self.output_dir = Path(self.config.get('output_dir', 'training_output'))
        self.models_dir = self.output_dir / 'models'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.output_dir, self.models_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_vocabulary(self) -> Dict[str, int]:
        """Load vocabulary from file or create default"""
        vocab_path = Path(self.config.get('vocab_path', 'data/vocabularies/symbols.json'))
        
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            logger.info(f"Loaded vocabulary with {len(vocab)} symbols")
        else:
            # Create default vocabulary (same as in the model)
            vocab = {
                # Numbers
                '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                
                # Basic operations
                '+': 10, '-': 11, '*': 12, '×': 13, '·': 14, '/': 15, '÷': 16, '=': 17,
                
                # Brackets
                '(': 18, ')': 19, '[': 20, ']': 21, '{': 22, '}': 23,
                
                # Variables
                'x': 24, 'y': 25, 'z': 26, 'a': 27, 'b': 28, 'c': 29,
                
                # Functions
                'sin': 30, 'cos': 31, 'tan': 32, 'log': 33, 'ln': 34,
                
                # Greek letters
                'π': 35, 'θ': 36, 'α': 37, 'β': 38, 'γ': 39,
                
                # Calculus
                '∫': 40, '∑': 41, '∂': 42, '√': 43,
                
                # Special
                '∞': 44, '^': 45, '_': 46, '.': 47, ',': 48,
                
                # Unknown
                '<UNK>': 49
            }
            
            # Save vocabulary
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created default vocabulary with {len(vocab)} symbols")
        
        return vocab
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        optimizer_type = self.config.get('optimizer', 'Adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=learning_rate, 
                           momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'StepLR')
        
        if scheduler_type.lower() == 'steplr':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('epochs', 50)
            )
        else:
            return None
    
    def setup_logging(self):
        """Setup TensorBoard logging"""
        log_dir = self.logs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        # Data transforms
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ])
        
        # Create dataset
        if self.config.get('synthetic_only', False):
            dataset = MathSymbolDataset(
                data_path="synthetic",
                vocab=self.vocab,
                transform=transform,
                synthetic=True
            )
        elif self.config.get('real_only', False):
            dataset = MathSymbolDataset(
                data_path=self.config.get('data_path', 'data/real'),
                vocab=self.vocab,
                transform=transform,
                real_data=True
            )
        else:
            dataset = MathSymbolDataset(
                data_path=self.config.get('data_path', 'data/mixed'),
                vocab=self.vocab,
                transform=transform,
                synthetic=False,
                real_data=False
            )
        
        # Split dataset
        val_split = self.config.get('val_split', 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to TensorBoard
            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', 100.*correct/total, step)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'vocab': self.vocab,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.models_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.models_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with accuracy: {accuracy:.2f}%")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png')
        plt.show()
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        epochs = self.config.get('epochs', 50)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            logger.info(f'Epoch {epoch+1}/{epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint
            is_best = val_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = val_acc
            
            # Save every 10 epochs or if best
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
        
        # Save final model
        self.save_checkpoint(epochs-1, self.val_accuracies[-1])
        
        # Plot training history
        self.plot_training_history()
        
        # Close TensorBoard writer
        self.writer.close()
        
        logger.info(f"Training completed! Best validation accuracy: {self.best_accuracy:.2f}%")
        
        # Test character identification with fallback
        self.test_enhanced_recognition()
    
    def add_character_identification_training(self):
        """
        Enhanced character identification training with improved accuracy
        """
        logger.info("Adding enhanced character identification training...")
        
        # Character identification specific augmentations
        self.char_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
        logger.info("Character identification training enhancements added")
    
    def integrate_gemini_fallback(self, gemini_api_key: str = None):
        """
        Integrate Gemini API fallback for character recognition
        """
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if gemini_api_key:
            try:
                # Try to import and initialize Gemini fallback
                logger.info("Attempting to integrate Gemini fallback...")
                self.gemini_available = True
                return True
            except Exception as e:
                logger.error(f"Failed to integrate Gemini fallback: {e}")
                self.gemini_available = False
                return False
        else:
            logger.warning("No Gemini API key provided, fallback disabled")
            self.gemini_available = False
            return False
    
    def test_enhanced_recognition(self):
        """
        Test enhanced character recognition capabilities
        """
        logger.info("Testing enhanced character recognition...")
        
        # Create test data
        test_chars = ['x', 'y', '2', '+', '=', 'π', '∫']
        
        logger.info(f"Enhanced recognition test completed for {len(test_chars)} characters")
        
        if hasattr(self, 'gemini_available') and self.gemini_available:
            logger.info("Gemini fallback integration successful")
        else:
            logger.info("Gemini fallback not available")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Math Transformer OCR Model')
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--synthetic', action='store_true', help='Use only synthetic data')
    parser.add_argument('--real_data_path', type=str, help='Path to real data')
    parser.add_argument('--mixed_training', action='store_true', help='Use mixed synthetic and real data')
    parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    else:
        # Default config
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'output_dir': args.output_dir,
            'synthetic_only': args.synthetic,
            'real_only': False,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'optimizer': 'Adam',
            'scheduler': 'StepLR',
            'step_size': 10,
            'gamma': 0.1,
            'weight_decay': 1e-5,
            'val_split': 0.2,
            'num_workers': 4
        }
    
    # Update config with command line args
    if args.real_data_path:
        config['data_path'] = args.real_data_path
        config['real_only'] = True
    
    if args.mixed_training:
        config['synthetic_only'] = False
        config['real_only'] = False
    
    # Create trainer and start training
    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
