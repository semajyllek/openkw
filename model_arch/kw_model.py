
import torch
import torch.nn as nn
from torch.nn import functional as F

class KWModel(nn.Module):
    """
    A simplified Time-Convolutional Residual Network (TC-ResNet)
    for Keyword Spotting, designed to output a fixed-size embedding.
    """
    def __init__(self, input_size=40, num_classes=None, embedding_dim=128):
        super().__init__()
        
        # --- Feature Extractor (Similar to a KWT/ResNet front-end) ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # --- Residual Blocks ---
        self.res_block1 = self._make_block(16, 32)
        self.res_block2 = self._make_block(32, 64)
        
        # --- Embedding Layer ---
        # Global Average Pooling across the time dimension (Dim 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final embedding projection layer
        self.embedder = nn.Linear(64, embedding_dim)

        # --- Classification Head (Used only during training) ---
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
            
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Input x: Mel spectrogram or MFCCs, shape (Batch, 1, Time, Freq)
        """
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Collapse time and frequency dimensions
        x = self.avg_pool(x).squeeze(3).squeeze(2) 
        
        # Output the 128-D embedding (the feature vector)
        embedding = self.embedder(x)
        
        # During training, return classification logits; otherwise, return embedding
        if self.classifier:
            return self.classifier(embedding), embedding
        
        return embedding
    
# Example of how the Keyword Transformer (KWT) would extend this:
# KWT replaces the ResBlocks with Transformer Encoder Blocks. 
# This KWModel serves as a solid foundation for transfer learning/embedding.
