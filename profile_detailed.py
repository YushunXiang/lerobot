#!/usr/bin/env python

import time
import sys
from pathlib import Path
from collections import defaultdict
import torch

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lerobot.scripts.train import train, update_policy
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.utils.train_utils import load_training_state
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device
from torch.amp import GradScaler
from lerobot.common.utils.logging_utils import MetricsTracker, AverageMeter

@parser.wrap()
def profile_training_detailed(cfg: TrainPipelineConfig):
    """Profile training with detailed timing for each component"""
    cfg.validate()
    
    # Limit steps for profiling
    cfg.steps = min(cfg.steps, 100)
    
    # Disable wandb for profiling
    cfg.wandb.enable = False
    
    # Timers
    timers = defaultdict(list)
    
    print("="*80)
    print("DETAILED PROFILING OF TRAINING COMPONENTS")
    print("="*80)
    
    # Setup
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Create dataset
    start = time.perf_counter()
    dataset = make_dataset(cfg)
    timers['dataset_creation'].append(time.perf_counter() - start)
    print(f"Dataset creation: {timers['dataset_creation'][-1]:.3f}s")
    
    # Create policy
    start = time.perf_counter()
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    timers['policy_creation'].append(time.perf_counter() - start)
    print(f"Policy creation: {timers['policy_creation'][-1]:.3f}s")
    
    # Create optimizer
    start = time.perf_counter()
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    timers['optimizer_creation'].append(time.perf_counter() - start)
    print(f"Optimizer creation: {timers['optimizer_creation'][-1]:.3f}s")
    
    # Create dataloader
    start = time.perf_counter()
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
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    timers['dataloader_creation'].append(time.perf_counter() - start)
    print(f"Dataloader creation: {timers['dataloader_creation'][-1]:.3f}s")
    
    # Initialize metrics
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=0
    )
    
    policy.train()
    
    print("\n" + "="*80)
    print("PROFILING TRAINING LOOP")
    print("="*80)
    
    # Profile first few batches in detail
    for step in range(min(10, cfg.steps)):
        # Data loading
        start = time.perf_counter()
        batch = next(dl_iter)
        timers['data_loading'].append(time.perf_counter() - start)
        
        # Data transfer to device
        start = time.perf_counter()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        timers['data_transfer'].append(time.perf_counter() - start)
        
        # Policy forward + backward
        start = time.perf_counter()
        with torch.cuda.synchronize(device) if device.type == 'cuda' else torch.no_grad():
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
            )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timers['policy_update'].append(time.perf_counter() - start)
        
        if step < 5:
            print(f"\nStep {step}:")
            print(f"  Data loading: {timers['data_loading'][-1]:.3f}s")
            print(f"  Data transfer: {timers['data_transfer'][-1]:.3f}s")
            print(f"  Policy update: {timers['policy_update'][-1]:.3f}s")
    
    # Continue profiling rest of steps with less detail
    for step in range(10, cfg.steps):
        start = time.perf_counter()
        batch = next(dl_iter)
        timers['data_loading'].append(time.perf_counter() - start)
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        start = time.perf_counter()
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timers['policy_update'].append(time.perf_counter() - start)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TIMING SUMMARY (seconds)")
    print("="*80)
    
    for key in ['data_loading', 'data_transfer', 'policy_update']:
        if key in timers and timers[key]:
            values = timers[key]
            avg = sum(values) / len(values)
            print(f"{key:20s}: avg={avg:.3f}, min={min(values):.3f}, max={max(values):.3f}, total={sum(values):.3f}")
    
    # Calculate percentages
    total_data_loading = sum(timers['data_loading'])
    total_policy_update = sum(timers['policy_update'])
    total_time = total_data_loading + total_policy_update
    
    print("\n" + "="*80)
    print("TIME DISTRIBUTION")
    print("="*80)
    print(f"Data loading: {total_data_loading:.3f}s ({100*total_data_loading/total_time:.1f}%)")
    print(f"Policy update: {total_policy_update:.3f}s ({100*total_policy_update/total_time:.1f}%)")
    
    # Profile memory usage if on CUDA
    if device.type == 'cuda':
        print("\n" + "="*80)
        print("GPU MEMORY USAGE")
        print("="*80)
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

if __name__ == "__main__":
    init_logging()
    profile_training_detailed()