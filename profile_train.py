#!/usr/bin/env python

import cProfile
import pstats
import io
import sys
import os
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lerobot.scripts.train import train
from lerobot.common.utils.utils import init_logging

def profile_training():
    """Profile the training script using cProfile"""
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    try:
        # Run the training
        train()
    finally:
        # Stop profiling
        profiler.disable()
        
        # Save raw profiling data
        profiler.dump_stats('training_profile.prof')
        
        # Create a stats object and print a summary
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        
        # Sort by cumulative time and print top 50 functions
        print("\n" + "="*80)
        print("PROFILING RESULTS - Top 50 functions by cumulative time:")
        print("="*80)
        ps.sort_stats('cumulative')
        ps.print_stats(50)
        print(s.getvalue())
        
        # Also sort by total time spent in the function itself
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        print("\n" + "="*80)
        print("PROFILING RESULTS - Top 50 functions by time spent in function:")
        print("="*80)
        ps.sort_stats('time')
        ps.print_stats(50)
        print(s.getvalue())
        
        # Print callers of expensive functions
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        print("\n" + "="*80)
        print("PROFILING RESULTS - Callers of expensive functions:")
        print("="*80)
        ps.sort_stats('cumulative')
        ps.print_callers(20)
        print(s.getvalue())
        
        print("\n" + "="*80)
        print(f"Profile data saved to: training_profile.prof")
        print("You can analyze it further with: python -m pstats training_profile.prof")
        print("="*80)

if __name__ == "__main__":
    init_logging()
    profile_training()