import torch
from torch.utils.data import Dataset, DataLoader
import random

class SortDataset(Dataset):
    """
    Toy task: 무작위 숫자 시퀀스 → 정렬된 시퀀스
    예) [3, 1, 4, 1, 5] → [1, 1, 3, 4, 5]
    Attention이 각 위치에서 어느 입력을 참조하는지 시각적으로 확인 가능
    """
    def __init__(self, n_samples, seq_len, vocab_size):
        self.data = []
        for _ in range(n_samples):
            src = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            tgt = sorted(src)
            self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


def get_dataloader(n_samples, seq_len, vocab_size, batch_size, shuffle=True):
    dataset = SortDataset(n_samples, seq_len, vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)