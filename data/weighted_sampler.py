
import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Imports from your repository structure
from model_arch.kwt_model import KWTModel
from data.kw_dataset import get_mel_spectrogram_transform, collate_fn

class HardNegativeWeightedSampler(Sampler):
    """
    A custom PyTorch Sampler that assigns higher probability to hard negative
    samples from a large pool based on their embedding distance to positive anchors.
    """
    def __init__(self, negative_dataset: Dataset, positive_anchors: np.ndarray, 
                 model: KWTModel, device: torch.device, batch_size: int, 
                 reweight_frequency=1):
        
        self.negative_dataset = negative_dataset
        self.positive_anchors = positive_anchors # Embeddings of positive classes/keywords
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.reweight_frequency = reweight_frequency 
        
        self.num_samples = len(self.negative_dataset)
        self.indices = list(range(self.num_samples))
        self.mel_transform = get_mel_spectrogram_transform(device=self.device)
        
        self.weights = torch.ones(self.num_samples, dtype=torch.double)
        print("Weighted Sampler Initialized. Calculating initial negative sample weights...")
        self.recalculate_weights()

    def __iter__(self):
        # Samples indices based on weights, with replacement, for the entire epoch
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

    # ----------------------------------------------------------------------
    # --- HELPER FUNCTIONS ---
    # ----------------------------------------------------------------------

    def _generate_negative_embeddings(self):
        """Generates embeddings for all audio files in the negative pool."""
        self.model.eval()
        loader = DataLoader(
            self.negative_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=4
        )
        
        all_neg_embeddings = []
        
        with torch.no_grad():
            for audio_data, _ in tqdm(loader, desc="Generating Negative Embeddings"):
                audio_data = audio_data.to(self.device)
                mel_spec = self.mel_transform(audio_data)
                
                # Model returns only the embedding (inference mode)
                embeddings = self.model(mel_spec) 
                all_neg_embeddings.append(embeddings.cpu().numpy())

        self.model.train() # Return model to training mode
        return np.concatenate(all_neg_embeddings, axis=0)

    def _calculate_hardness_scores(self, neg_embeddings):
        """Calculates the hardness score based on max similarity to positive anchors."""
        
        # 1. Calculate Cosine Similarity to the positive anchors
        # Output shape: (Num_Negatives, Num_Anchors)
        similarities = cosine_similarity(neg_embeddings, self.positive_anchors)
        
        # 2. Hardness Score: Max similarity to any positive anchor
        # Higher score = Closer to a positive anchor = Harder Negative
        max_similarities = np.max(similarities, axis=1)

        return max_similarities

    def _assign_weights_from_scores(self, hardness_scores):
        """Converts hardness scores into normalized probabilities (weights)."""
        
        # Use a power function (or exponential) to heavily bias sampling towards the hardest samples
        # Add a small epsilon to stabilize the calculation
        weights_np = np.power(hardness_scores + 1e-6, 3) 
        
        weights = torch.from_numpy(weights_np).double()
        weights = weights / weights.sum() # Normalize to sum to 1
        
        return weights

    # ----------------------------------------------------------------------
    # --- MAIN PUBLIC METHOD ---
    # ----------------------------------------------------------------------

    def recalculate_weights(self):
        """
        Public method to recalculate weights based on the model's current state.
        Called at the start of training and periodically throughout.
        """
        # 1. Generate Embeddings
        neg_embeddings = self._generate_negative_embeddings()

        # 2. Calculate Hardness
        hardness_scores = self._calculate_hardness_scores(neg_embeddings)

        # 3. Assign and Store Weights
        self.weights = self._assign_weights_from_scores(hardness_scores)

        print(f"Weights recalculated. Max weight: {self.weights.max().item():.5f}. Min weight: {self.weights.min().item():.5f}")
