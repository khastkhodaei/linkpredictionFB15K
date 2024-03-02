import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples.iloc[idx]  # Use iloc instead of directly accessing by index
        head = triple.iloc[0]  # Access value by position using iloc
        relation = triple.iloc[1]  # Access value by position using iloc
        tail = triple.iloc[2]  # Access value by position using iloc
        label = 1  # Placeholder label, modify this accordingly
        return {
            'head': torch.tensor(head),
            'relation': torch.tensor(relation),
            'tail': torch.tensor(tail),
            'label': torch.tensor(label)  # Add a placeholder label
        }
