"""
BrainECHO Training Entry Point - Phase 1

Main entry point for BrainECHO experiments using Hydra configuration.

Usage:
    # Basic training
    python main_run.py dataset=hbn dataset_name=hbn \\
        dataset_path=/path/to/hbn/ save_path=./checkpoints/
    
    # Quick test
    python main_run.py dataset=hbn dataset_name=hbn \\
        dataset_path=/path/to/hbn/ save_path=./test/ \\
        max_samples=10
    
    # Override parameters
    python main_run.py dataset=brennan dataset_name=brennan \\
        dataset_path=/path/to/brennan/ save_path=./ckpt/ \\
        batch_size=[32,32] lr_finetune=[2e-4,2e-4]
"""

import os
import hydra
from omegaconf import DictConfig
from BrainECHORunner import BrainECHORunner


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig):
    """
    Main entry point for BrainECHO training
    
    Args:
        cfg: Hydra configuration object loaded from configs/
    
    Flow:
        1. Load configuration from configs/config.yaml + dataset overrides
        2. Add root directory to path if specified
        3. Initialize Runner
        4. Execute training
    """
    # Add root directory to path if specified
    if hasattr(cfg, 'root_directory') and cfg.root_directory:
        import sys
        sys.path.insert(1, cfg.root_directory)
    
    # Initialize and run
    runner = BrainECHORunner()
    runner.run(cfg)


if __name__ == "__main__":
    run()
