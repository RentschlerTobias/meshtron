"""
dataset_domain.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron). Kept for reference only, not maintained or extended.

Dataset für Domain-Partition Training (kompatibel mit tokenizer-sorting Trainer).
Ladt vorverarbeitete Daten (domain_data.pt) und tokenisiert sie.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer_domain import DomainTokenizer
from typing import Optional, List, Dict
from tqdm import tqdm


class DomainMeshData(Dataset):
    """
    Dataset für Domain-Partition Meshtron.
    API-kompatibel mit MeshData (dataset.py).
    """

    def __init__(
        self,
        meshes: List[Dict],
        tokenizer: DomainTokenizer,
        max_seq_length: Optional[int] = None,
        n_sample_points: int = 1500,
        with_labels: bool = False,
        verbose: bool = True,
    ):
        self.meshes = meshes
        self.tokenizer = tokenizer
        self.n_sample_points = n_sample_points
        self.with_labels = with_labels
        self.data: List[List[int]] = []
        self.face_count: List[int] = []

        if verbose:
            print("Starte Domain-Partition Tokenisierung...")

        for i in tqdm(range(len(meshes)), desc="domain meshes", disable=not verbose):
            mesh = meshes[i]
            tokens = tokenizer.tokenize(mesh)
            self.data.append(tokens)
            self.face_count.append(mesh['faces'].shape[1])

        if max_seq_length is None:
            self.max_seq_length = tokenizer.max_length_token_sequence
        else:
            self.max_seq_length = max_seq_length

        self.min_seq_length = tokenizer.min_length_token_sequence

        if verbose:
            print(f"\nMax Sequenzlänge: {self.max_seq_length}\nMin Sequenzlänge: {self.min_seq_length}")

    def _sample_point_cloud(self, tri_coordinates: torch.Tensor) -> torch.Tensor:
        """Sample point cloud. Prioritaet: Ecken(Label 0) > Rand(1) > Feld(2).
        Ecken UND Rand werden IMMER vollstaendig behalten (nie durch Sampling
        verworfen), der Rest mit Feld-Punkten (ggf. wiederholt+Noise) aufgefuellt.
        with_labels=True -> Ausgabe [n, 3] mit Label-Spalte (0/1/2, Pad=3), sonst
        [n, 2]. Nur die x/y werden auf [-1,1] normalisiert, das Label bleibt."""
        n = self.n_sample_points
        labels = tri_coordinates[:, 2].long()
        keep = tri_coordinates[labels != 2]          # Ecken(0) + Rand(1), IMMER behalten
        field = tri_coordinates[labels == 2]         # Feld(2), fuellt auf
        nk = keep.size(0)

        if nk >= n:
            # sehr selten (Ecken+Rand > n): Ecken zuerst, Rest zufaellig aus Rand
            corners = tri_coordinates[labels == 0]
            bound = tri_coordinates[labels == 1]
            nc = corners.size(0)
            idx = torch.randperm(bound.size(0))[:max(0, n - nc)]
            cloud = torch.cat([corners[:n], bound[idx]], dim=0)[:n]
        else:
            rem = n - nk
            nf = field.size(0)
            if nf == 0:
                fill = keep[torch.randint(0, nk, (rem,))]
            elif nf >= rem:
                fill = field[torch.randint(0, nf, (rem,))]
            else:
                rep = (rem + nf - 1) // nf
                field_rep = field.repeat(rep, 1)
                field_rep[:, :2] = field_rep[:, :2] + torch.rand_like(field_rep[:, :2]) * 0.001
                fill = field_rep[torch.randint(0, field_rep.size(0), (rem,))]
            cloud = torch.cat([keep, fill], dim=0)   # [n, 3]

        # Normalisierung der x/y wie in dataset.py: [-1, 1]; Label unberuehrt
        all_coords = tri_coordinates[:, :2]
        center = (all_coords.max(dim=0).values + all_coords.min(dim=0).values) / 2
        scale = (all_coords.max(dim=0).values - all_coords.min(dim=0).values).max().clamp(min=1e-6)
        xy = (cloud[:, :2] - center) / scale * 2

        if self.with_labels:
            return torch.cat([xy, cloud[:, 2:3]], dim=1)         # [n, 3]
        return xy                                                # [n, 2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        mesh = self.meshes[idx]
        point_cloud = self._sample_point_cloud(mesh['tri_coordinates'])
        num_faces = self.face_count[idx]
        pad_token = self.tokenizer.pad_token

        # Padding
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        tokens_padded = tokens + [pad_token] * (self.max_seq_length - len(tokens))
        tokens_tensor = torch.LongTensor(tokens_padded)

        input_tokens = tokens_tensor[:-1]
        target_tokens = tokens_tensor[1:]
        padding_mask = (input_tokens != pad_token).float()

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'padding_mask': padding_mask,
            'pad_token': pad_token,
            'point_cloud': torch.FloatTensor(point_cloud),
            'face_count': num_faces,
            'seq_length': len(tokens) - 1,
        }


def get_domain_loaders(
    data_path: str = '/root/repos/meshtron/domain_data.pt',
    tokenizer: Optional[DomainTokenizer] = None,
    train_ratio: float = 0.8,
    batch_size: int = 8,
    n_sample_points: int = 1500,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Erstellt Train/Val Loader für Domain-Partition."""
    meshes = torch.load(data_path, weights_only=False)
    if tokenizer is None:
        raise ValueError("Tokenizer muss übergeben werden")

    dataset = DomainMeshData(
        meshes=meshes,
        tokenizer=tokenizer,
        n_sample_points=n_sample_points,
        verbose=True,
    )

    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, dataset.max_seq_length, dataset.min_seq_length


if __name__ == '__main__':
    tok = DomainTokenizer(quantization_r=64, quantization_a=32, sorting_strategy=0, embedding_mode=0, verbose=False)
    train_loader, val_loader, max_len, min_len = get_domain_loaders(
        tokenizer=tok, batch_size=4
    )
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("input_tokens:", batch['input_tokens'].shape)
    print("point_cloud:", batch['point_cloud'].shape)
    print("face_count:", batch['face_count'])
