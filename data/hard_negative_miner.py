import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

class HardNegativeMiner:
    """
    A class to mine hard negative samples.
    """
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval() # Put model in evaluation mode for mining

    # --- Method 1: Robust Full-Dataset Ranking ---
    def _rank_all_negatives(self, negative_dataset: Dataset) -> List[Tuple[str, float]]:
        """
        Processes the entire negative dataset, scores each sample based on 
        the model's false positive probability, and returns a sorted list.
        
        This is the most robust method for finding true hard negatives.
        """
        all_samples_with_scores = []
        data_loader = DataLoader(negative_dataset, batch_size=64, shuffle=False)
        
        print("Scoring all negatives to find the hardest ones...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Ranking Negatives")):
                audio_tensors = [a.numpy() for a in batch['audio']]
                
                inputs = self.processor(
                    audio=audio_tensors,
                    sampling_rate=16000,
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get the probability of the wake word class (class 1)
                wake_word_probs = torch.softmax(logits, dim=1)[:, 1]
                
                for i, prob in enumerate(wake_word_probs):
                    # Find the original file path
                    # This assumes your NegativeWordUnitDataset returns file paths
                    file_path = negative_dataset.file_paths[batch_idx * data_loader.batch_size + i]
                    all_samples_with_scores.append((file_path, prob.item()))
                    
        # Sort all samples from highest to lowest wake word probability
        all_samples_with_scores.sort(key=lambda x: x[1], reverse=True)
        return all_samples_with_scores

    # --- Method 2: Simplified Top-K Per-Batch Approach ---
    def _get_top_k_per_batch(self, negative_dataset: Dataset, k: int = 5) -> List[str]:
        """
        A simpler, less robust method that selects the top K hardest negatives 
        from each batch. It is faster but may miss the absolute hardest samples.
        """
        hard_negatives = []
        data_loader = DataLoader(negative_dataset, batch_size=64, shuffle=False)

        print(f"Mining top {k} hard negatives per batch...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Mining Negatives")):
                audio_tensors = [a.numpy() for a in batch['audio']]
                
                inputs = self.processor(
                    audio=audio_tensors,
                    sampling_rate=16000,
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities for the wake word class (index 1)
                wake_word_probs = torch.softmax(logits, dim=1)[:, 1]
                
                # Get the top K indices with the highest probability
                top_prob_per_batch, indices_in_batch = torch.topk(wake_word_probs, k=min(len(wake_word_probs), k))
                
                for i in indices_in_batch:
                    # Get the global index from the batch index
                    global_idx = batch_idx * data_loader.batch_size + i
                    file_path = negative_dataset.file_paths[global_idx]
                    hard_negatives.append(file_path)

        return hard_negatives

    def mine_hard_negatives(self, negative_dataset: Dataset, num_to_mine: int, strategy: str = "ranking") -> List[str]:
        """
        Public method to choose the mining strategy.
        """
        if strategy == "ranking":
            scored_negatives = self._rank_all_negatives(negative_dataset)
            return [path for path, score in scored_negatives[:num_to_mine]]
        elif strategy == "topk_per_batch":
            return self._get_top_k_per_batch(negative_dataset, k=num_to_mine)
        else:
            raise ValueError("Invalid mining strategy. Choose 'ranking' or 'topk_per_batch'.")


# A small dummy class for the mined negative samples, since it's a new class in the overall plan.
class HardNegativeDataset(Dataset):
    def __init__(self, file_paths, label, sample_rate=16000):
        self.file_paths = file_paths
        self.label = label
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        return {"audio": waveform.squeeze(0), "labels": torch.tensor(self.label)}
