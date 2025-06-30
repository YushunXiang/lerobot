#!/usr/bin/env python

import time
import sys
from pathlib import Path
import torch
import numpy as np

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lerobot.common.datasets.factory import make_dataset
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.utils.random_utils import set_seed

@parser.wrap()
def profile_dataloader(cfg: TrainPipelineConfig):
    """Profile dataloader performance"""
    print("="*80)
    print("DATALOADER PROFILING")
    print("="*80)
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Create dataset
    print("Creating dataset...")
    start = time.perf_counter()
    dataset = make_dataset(cfg)
    print(f"Dataset creation took: {time.perf_counter() - start:.3f}s")
    print(f"Dataset has {dataset.num_frames} frames and {dataset.num_episodes} episodes")
    
    # Create dataloader with different configurations
    configs = [
        {"num_workers": 0, "batch_size": cfg.batch_size},
        {"num_workers": 1, "batch_size": cfg.batch_size},
        {"num_workers": 2, "batch_size": cfg.batch_size},
        {"num_workers": 4, "batch_size": cfg.batch_size},
        {"num_workers": 8, "batch_size": cfg.batch_size},
        {"num_workers": cfg.num_workers, "batch_size": 64},
        {"num_workers": cfg.num_workers, "batch_size": 128},
        {"num_workers": cfg.num_workers, "batch_size": 256},
        {"num_workers": cfg.num_workers, "batch_size": 512},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing with num_workers={config['num_workers']}, batch_size={config['batch_size']}")
        print('='*60)
        
        # Create dataloader
        if hasattr(cfg.policy, "drop_n_last_frames"):
            shuffle = False
            sampler = EpisodeAwareSampler(
                dataset.episode_data_index,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=True,
            )
        else:
            shuffle = True
            sampler = None
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=config['num_workers'],
            batch_size=config['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2 if config['num_workers'] > 0 else None,
        )
        
        # Warm up
        print("Warming up...")
        iter_loader = iter(dataloader)
        for _ in range(5):
            try:
                _ = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dataloader)
                _ = next(iter_loader)
        
        # Time batches
        times = []
        n_batches = 20
        
        print(f"Timing {n_batches} batches...")
        for i in range(n_batches):
            start = time.perf_counter()
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dataloader)
                batch = next(iter_loader)
            batch_time = time.perf_counter() - start
            times.append(batch_time)
            
            if i == 0:
                # Print batch info
                print(f"\nBatch info:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Calculate statistics
        times = np.array(times)
        print(f"\nDataloader timing statistics:")
        print(f"  Mean: {times.mean():.3f}s")
        print(f"  Std: {times.std():.3f}s")
        print(f"  Min: {times.min():.3f}s")
        print(f"  Max: {times.max():.3f}s")
        print(f"  Total for {n_batches} batches: {times.sum():.3f}s")
        print(f"  Throughput: {config['batch_size']/times.mean():.1f} samples/sec")

if __name__ == "__main__":
    profile_dataloader()