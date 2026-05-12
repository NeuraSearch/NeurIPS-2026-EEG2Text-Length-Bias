"""
Dataset Configuration Module for BrainECHO Training

This module defines dataset-specific configurations for training the BrainECHO model.
Each dataset configuration specifies the unique parameters needed for that dataset.

To add a new dataset:
1. Create a new DatasetConfig instance with all required fields
2. Register it in DATASET_CONFIGS
3. Use it by passing --dataset_name <name> to the training script
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any
import pickle


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset"""
    
    # dataset identification
    name: str
    description: str
    
    # EEG-specific parameters
    in_channels: int  # Number of EEG channels (e.g., 62 or 129)
    eeg_length_mode: str = "fixed"  # "fixed" or "dynamic"
    fixed_eeg_length: Optional[int] = 2400  # Used when eeg_length_mode="fixed"
    
    # Mel spectrogram parameters
    mel_length_mode: str = "fixed"  # "fixed" or "from_data"
    fixed_mel_length: Optional[int] = 1200  # Used when mel_length_mode="fixed"
    mel_features_path_pattern: str = "../log_mel_features.pickle"  # Relative to dataset_path
    
    # dataset-specific loading options
    dataset_class_name: str = "EEG_dataset_whisper"  # Name of dataset class to use
    dataset_kwargs_stage2: dict = None  # Extra kwargs for stage 2 dataset loading
    dataset_kwargs_stage3: dict = None  # Extra kwargs for stage 3 dataset loading
    reload_dataset_stage3: bool = False  # Whether to reload datasets between stages
    
    # AdaLoRA configuration
    adalora_mode: str = "dynamic"  # "fixed" or "dynamic"
    adalora_tinit: Optional[int] = 200  # Used when adalora_mode="fixed"
    adalora_tfinal: Optional[int] = 1000  # Used when adalora_mode="fixed"
    adalora_tinit_ratio: float = 0.015  # Used when adalora_mode="dynamic"
    adalora_tfinal_ratio: float = 0.07  # Used when adalora_mode="dynamic"
    whisper_model: str = "openai/whisper-base.en"
    language: Optional[str] = "en"
    
    # Optional preprocessing/validation
    preprocess_fn: Optional[Callable] = None  # Custom preprocessing function
    
    def __post_init__(self):
        """Initialize mutable default values"""
        if self.dataset_kwargs_stage2 is None:
            self.dataset_kwargs_stage2 = {}
        if self.dataset_kwargs_stage3 is None:
            self.dataset_kwargs_stage3 = {}
    
    def get_eeg_length(self, rawEEG=None):
        """Get EEG sequence length based on configuration"""
        if self.eeg_length_mode == "fixed":
            return self.fixed_eeg_length
        elif self.eeg_length_mode == "dynamic":
            if rawEEG is None:
                raise ValueError("rawEEG must be provided for dynamic EEG length")
            return rawEEG.shape[-1]
        else:
            raise ValueError(f"Unknown eeg_length_mode: {self.eeg_length_mode}")
    
    def get_mel_length(self, dataset_path: str, split_mel: bool = False, split_len: int = 400):
        """Get mel spectrogram length based on configuration"""
        if split_mel:
            return split_len
        
        if self.mel_length_mode == "fixed":
            return self.fixed_mel_length
        elif self.mel_length_mode == "from_data":
            # Load mel features to determine length
            mel_path = dataset_path + self.mel_features_path_pattern
            with open(mel_path, "rb") as f:
                log_mel_features = pickle.load(f)
                return log_mel_features.shape[2]
        else:
            raise ValueError(f"Unknown mel_length_mode: {self.mel_length_mode}")
    
    def get_adalora_params(self, total_steps: int):
        """Get AdaLoRA tinit and tfinal parameters"""
        if self.adalora_mode == "fixed":
            return self.adalora_tinit, self.adalora_tfinal
        elif self.adalora_mode == "dynamic":
            tinit = int(self.adalora_tinit_ratio * total_steps)
            tfinal = int(self.adalora_tfinal_ratio * total_steps)
            return tinit, tfinal
        else:
            raise ValueError(f"Unknown adalora_mode: {self.adalora_mode}")


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

BRENNAN_CONFIG = DatasetConfig(
    name="brennan",
    description="Brennan et al. Alice in Wonderland dataset (49 subjects, 140 sentences)",
    in_channels=62,
    eeg_length_mode="fixed",
    fixed_eeg_length=2400,
    mel_length_mode="fixed",
    fixed_mel_length=1200,
    dataset_class_name="EEG_dataset_whisper",
    adalora_mode="fixed",
    adalora_tinit=200,
    adalora_tfinal=1000
)

HBN_CONFIG = DatasetConfig(
    name="hbn",
    description="Healthy Brain Network movie dataset (136 subjects, 4 movies)",
    in_channels=129,
    eeg_length_mode="fixed",
    fixed_eeg_length=2400,
    mel_length_mode="fixed",
    fixed_mel_length=1200,
    dataset_class_name="EEG_dataset_whisper",
    dataset_kwargs_stage2={},  # Will be populated with exclude_the_present=False, max_samples
    dataset_kwargs_stage3={},  # Will be populated based on --exclude_the_present_stage3
    reload_dataset_stage3=True,  # Need to reload if filtering ThePresent
    adalora_mode="dynamic",
    adalora_tinit_ratio=0.015,
    adalora_tfinal_ratio=0.07,
)

# Registry of datasets
DATASET_CONFIGS = {
    "brennan": BRENNAN_CONFIG,
    "hbn": HBN_CONFIG,
}

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Get dataset configuration by name.

    Args:
        dataset_name: Name of the dataset ('brennan' or 'hbn')

    Returns:
        DatasetConfig object for the specified dataset

    Raises:
        ValueError: If dataset_name is not found in registry
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    return DATASET_CONFIGS[dataset_name]


def list_available_datasets():
    """Print information about all available datasets"""
    print("\n" + "="*80)
    print("AVAILABLE DATASETS")
    print("="*80)
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"Description: {config.description}")
        print(f"Channels: {config.in_channels}")
        print(f"EEG Length: {config.eeg_length_mode} "
              f"({config.fixed_eeg_length if config.eeg_length_mode == 'fixed' else 'from data'})")
        print(f"Mel Length: {config.mel_length_mode} "
              f"({config.fixed_mel_length if config.mel_length_mode == 'fixed' else 'from data'})")
        print(f"AdaLoRA: {config.adalora_mode}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    list_available_datasets()

