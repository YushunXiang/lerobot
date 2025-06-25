# Performance Analysis for Diffusion Policy Training

## Profiling Scripts Created

1. **profile_train.py** - Uses cProfile to profile the entire training process
2. **profile_detailed.py** - Provides detailed timing for each component
3. **profile_dataloader.py** - Specifically profiles dataloader performance with different configurations
4. **profile.sh** - Bash script to run profiling with your specific configuration

## How to Run Profiling

1. **Basic cProfile profiling:**
   ```bash
   ./profile.sh
   ```
   This will create a `training_profile.prof` file and print top functions by time.

2. **Detailed component timing:**
   ```bash
   python profile_detailed.py --dataset.repo_id=lerobot/pp-red-apple-pot-2 --dataset.root=/dev/shm/lerobot/pp-red-apple-pot-2 --policy.type=diffusion --num_workers=4 --batch_size=256
   ```

3. **Dataloader profiling:**
   ```bash
   python profile_dataloader.py --dataset.repo_id=lerobot/pp-red-apple-pot-2 --dataset.root=/dev/shm/lerobot/pp-red-apple-pot-2 --policy.type=diffusion --num_workers=4 --batch_size=256
   ```

## Potential Performance Bottlenecks and Optimizations

### 1. **Data Loading**
- **Issue**: Data loading can be a major bottleneck in training
- **Optimizations**:
  - Adjust `num_workers` - test with 0, 2, 4, 8 workers
  - Enable `persistent_workers=True` in DataLoader
  - Increase `prefetch_factor` (default is 2)
  - Use `pin_memory=True` (already enabled)
  - Consider caching frequently accessed data

### 2. **Data Transformations**
- **Issue**: Image transforms can be expensive
- **Optimizations**:
  - Move transforms to GPU if possible
  - Use torchvision.transforms.v2 for better performance
  - Consider pre-computing transforms if they're deterministic

### 3. **Batch Size**
- **Current**: 256
- **Optimization**: Try larger batch sizes (512, 1024) if GPU memory allows

### 4. **Mixed Precision Training**
- **Check**: Verify `use_amp` is enabled in policy config
- **Optimization**: Ensure using automatic mixed precision for faster training

### 5. **Gradient Accumulation**
- **Optimization**: If limited by GPU memory, use gradient accumulation to simulate larger batches

### 6. **PyTorch Settings**
- Already enabled:
  - `torch.backends.cudnn.benchmark = True`
  - `torch.backends.cuda.matmul.allow_tf32 = True`
- Additional optimizations:
  - Try `torch.compile()` for PyTorch 2.0+
  - Enable CUDNN deterministic if not needed: `torch.backends.cudnn.deterministic = False`

### 7. **Data Storage**
- **Current**: Using `/dev/shm` (shared memory)
- **Good**: This is already optimal for I/O performance

### 8. **Dataset-specific Optimizations**
- Cache episode indices if repeatedly accessing same episodes
- Pre-load frequently accessed data into memory
- Use memory mapping for large datasets

## Recommended Profiling Steps

1. Run `profile_dataloader.py` first to find optimal `num_workers` and `batch_size`
2. Run `profile_detailed.py` to identify time distribution between data loading and model computation
3. Run `profile_train.py` for full profiling to identify unexpected bottlenecks
4. Analyze results using:
   ```bash
   python -m pstats training_profile.prof
   ```

## Quick Optimizations to Try

1. **Immediate**: Add to your DataLoader:
   ```python
   persistent_workers=True,  # Keep workers alive between epochs
   prefetch_factor=4,       # Increase from default 2
   ```

2. **Test different worker counts**:
   ```bash
   # Modify num_workers in your script
   --num_workers=0  # Sometimes faster for small datasets
   --num_workers=8  # May be better for large datasets
   ```

3. **Enable torch.compile (if using PyTorch 2.0+)**:
   Add after policy creation:
   ```python
   policy = torch.compile(policy)
   ```