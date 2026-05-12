"""
VQVAE Trainer - Stage 1 (Mel-Spectrogram Autoencoding)

Trains the Mel-VQVAE (Vector Quantized Variational AutoEncoder) that learns
to encode and reconstruct mel spectrograms. This is Stage 1 of BrainECHO pipeline.

The trained VQVAE checkpoint is then used in Stages 2 & 3 for brain-audio alignment.
"""

import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm
from transformers import WhisperProcessor
import pickle
import math
from einops.einops import rearrange
from omegaconf import OmegaConf

from Components.TrainingLayer.TrainingInterface import TrainingInterface
from model import MelVQVAE
from utils.utils import *
from optim_new import *


class VQVAETrainer(TrainingInterface):
    """
    Trainer for Stage 1: Mel-VQVAE Pretraining
    
    Trains an autoencoder on mel spectrograms to learn compressed audio representations.
    This is the first stage of BrainECHO pipeline, before brain-audio alignment.
    
    Unlike BrainECHOTrainer, this trains only on audio data (mel spectrograms),
    not EEG signals.
    """
    
    def __init__(self, model=None, loss_fn=None, optimizer=None, device="cpu", 
                 scheduler=None, metrics=None, cfg=None):
        """
        Initialize VQVAE Trainer
        
        Args:
            model: MelVQVAE model (can be None, will be created from cfg)
            loss_fn: Loss function (not used, VQVAE has internal loss)
            optimizer: Optimizer (not used, created internally)
            device: Device for training
            scheduler: Learning rate scheduler (not used, created internally)
            metrics: Metrics dict (not used)
            cfg: Hydra configuration with all VQVAE parameters
        """
        # Don't call super().__init__() - we manage everything differently
        self.device = device
        self.cfg = cfg
        
        # Extract parameters from config
        self.dataset_name = cfg.get('vqvae_dataset', 'all_datasets')  # Default: train on all
        self.split_mel = cfg.get('split_mel', False)
        self.split_len = cfg.get('split_len', 4) * 100
        self.codebook_size = cfg.get('cb_size', 2048)
        self.downsample_ratio = cfg.get('ratio', 4)
        self.mel_len = 1200 if not self.split_mel else self.split_len
        
        # Save path
        self.save_path = cfg.get('vqvae_save_path') or cfg.get('save_path', './checkpoints/Brennan/')

        # Initialize logger
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        init_logger(config_dict)
        self.logger = getLogger()
        
        print(f"\n{'='*80}")
        print(f"VQVAE Trainer Initialized (Stage 1)")
        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"Codebook size: {self.codebook_size}")
        print(f"Downsample ratio: {self.downsample_ratio}")
        print(f"Device: {self.device}")
        print(f"Save path: {self.save_path}")
        print(f"{'='*80}\n")
    
    def train_step(self, batch):
        """
        Single training step (required by TrainingInterface but not used)
        VQVAE handles training internally
        """
        raise NotImplementedError("VQVAE uses custom training loop in fit()")
    
    def validation_step(self, batch):
        """
        Single validation step (required by TrainingInterface but not used)
        VQVAE handles validation internally
        """
        raise NotImplementedError("VQVAE uses custom training loop in fit()")
    
    def fit(self, train_loader=None, val_loader=None, epochs=None):
        """
        Train VQVAE model (Stage 1)
        
        Trains mel spectrogram autoencoder to learn compressed audio representations.
        
        Args:
            train_loader: Ignored (data loaded internally)
            val_loader: Ignored (single training phase, no validation)
            epochs: Ignored (epochs from config)
            
        Returns:
            dict: Training metrics including checkpoint path
        """
        print("\n" + "="*80)
        print("STARTING STAGE 1: MEL-VQVAE PRETRAINING")
        print("="*80 + "\n")
        
        # Create save directory
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Set random seeds
        seed_val = self.cfg.get('seed', 312)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.benchmark = True
        
        # Load mel features from specified datasets
        print(f"[INFO] Loading mel features for: {self.dataset_name}")
        log_mel_features = self._load_mel_features()
        print(f"[INFO] Loaded mel features: {log_mel_features.shape}")
        
        # Add noise samples to enhance robustness
        mel_length = log_mel_features.shape[2]
        audio_duration = mel_length / 100
        noise_audio = [0] * (int(audio_duration) * 16000)
        processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
        noise_mel = processor(noise_audio, sampling_rate=16000, return_tensors="pt").input_features
        noise_mel = noise_mel[..., :mel_length]
        log_mel_features = torch.cat([log_mel_features, noise_mel])
        
        print(f"[INFO] Total mel features (with noise): {log_mel_features.shape}")
        
        # Create dataset and dataloader
        train_set = TensorDataset(log_mel_features)
        dataset_size = len(train_set)
        print(f"[INFO] Train set size: {dataset_size}")
        
        if dataset_size == 0:
            raise ValueError("Training set is empty!")
        
        # Training parameters
        num_epochs = self.cfg.get('vqvae_num_epochs', 400)
        lr = self.cfg.get('vqvae_lr', 2e-4)
        batch_size = self.cfg.get('vqvae_batch_size', 16)
        T_max = self.cfg.get('vqvae_T_max', num_epochs)
        
        print(f"[INFO] Epochs: {num_epochs}, LR: {lr}, Batch: {batch_size}, T_max: {T_max}")
        
        # Dataloader
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Initialize model
        print(f"[INFO] Initializing MelVQVAE model...")
        model = MelVQVAE(codebook_size=self.codebook_size, downsample_ratio=self.downsample_ratio)
        model.to(self.device)
        
        # Optimizer and scheduler
        early_stopping = EarlyStopper(patience=4, min_delta=0.01)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        config_dict['lr_finetune'] = lr
        optimizer = build_optimizer(config_dict, model, mode="finetune")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
        # Checkpoint path
        save_name = f"MelVQVAE_b{batch_size}_{num_epochs}_{lr}_autoencode"
        checkpoint_path = f"{self.save_path}/{save_name}.pt"
        print(f"[INFO] Checkpoint will be saved to: {checkpoint_path}")
        
        # Train
        print("\n[INFO] Starting Stage 1 training...")
        self._show_require_grad_layers(model)
        metrics = self._train_vqvae(
            train_dataloader,
            model,
            optimizer,
            scheduler,
            early_stopping,
            num_epochs,
            checkpoint_path,
            dataset_size
        )
        print("[INFO] Stage 1 training complete!")
        
        print("\n" + "="*80)
        print("STAGE 1 (VQVAE) COMPLETE!")
        print("="*80)
        print(f"Best Epoch: {metrics['best_epoch']}")
        print(f"Best Loss: {metrics['best_loss']:.4f}")
        print(f"Training Time: {metrics['training_time_seconds']}s")
        print(f"Checkpoint: {checkpoint_path}")
        print("="*80 + "\n")
        
        # Return metrics including checkpoint path
        return {
            "stage1_metrics": metrics,
            "stage1_checkpoint": checkpoint_path,
            "checkpoint_path": checkpoint_path,  # For easy access
        }

    def _load_mel_features(self):
        """
        Load mel features from specified datasets

        Returns:
            torch.Tensor: Concatenated mel features
        """
        dataset_name = self.dataset_name.lower()

        # Get base path from config
        if hasattr(self.cfg, 'dataset_path'):
            base_path = self.cfg.dataset_path
            base_path = os.path.dirname(base_path.rstrip('/'))
            print(f"[INFO] Base path for mel features: {base_path}")
        else:
            base_path = "datasets/"

        def _load_as_tensor(path, label):
            """Load a mel pickle and convert to float32 tensor if needed."""
            with open(path, "rb") as f:
                m = pickle.load(f)
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m).float()
            elif isinstance(m, torch.Tensor):
                m = m.float()
            else:
                raise TypeError(
                    f"Unexpected mel pickle type for {label}: {type(m).__name__}"
                )
            print(f"       {label}: {tuple(m.shape)}  dtype={m.dtype}")
            return m

        if dataset_name in ["brennan", "selis", "hbn", "sparrkulee", "lpphk", "largespanisheeg"]:
            # All single datasets: just load from base_path/log_mel_features.pickle
            mel_path = os.path.join(base_path, "log_mel_features.pickle")
            print(f"[INFO] Loading {dataset_name.upper()} mel features from: {mel_path}")
            log_mel_features = _load_as_tensor(mel_path, dataset_name.upper())

        elif dataset_name == "all_datasets":
            # Get the root datasets directory (two levels up)
            datasets_root = os.path.dirname(os.path.dirname(base_path.rstrip('/')))

            dataset_paths = {
                "Brennan": os.path.join(datasets_root, "Brennan", "log_mel_features.pickle"),
                "selis": os.path.join(datasets_root, "selis", "log_mel_features.pickle"),
                "HBN": os.path.join(datasets_root, "HBN", "log_mel_features.pickle"),
                "SparrKULee": os.path.join(datasets_root, "SparrKuLee", "log_mel_features.pickle"),
                "LPPHK": os.path.join(datasets_root, "LPPHK", "log_mel_features.pickle"),
                "LargeSpanishEEG": os.path.join(datasets_root, "LargeSpanishEEG", "log_mel_features.pickle"),
            }

            print("[INFO] Loading mel features from all datasets:")
            mels_list = [_load_as_tensor(path, label) for label, path in dataset_paths.items()]
            log_mel_features = torch.cat(mels_list)
            print(f"[INFO] Concatenated shape: {tuple(log_mel_features.shape)}")

        else:
            raise ValueError(
                "Unknown dataset. Use: brennan, selis, hbn, sparrkulee, lpphk, largespanisheeg, or all_datasets"
            )

        return log_mel_features
    
    def _train_vqvae(self, train_dataloader, model, optimizer, scheduler, early_stopping,
                     num_epochs, checkpoint_path, dataset_size):
        """
        Internal training loop for VQVAE
        
        Args:
            train_dataloader: DataLoader with mel features
            model: MelVQVAE model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            early_stopping: Early stopping object
            num_epochs: Number of epochs
            checkpoint_path: Path to save checkpoint
            dataset_size: Size of training set
            
        Returns:
            dict: Training metrics
        """
        start_time = time.time()
        best_valid_loss = None
        stop = False
        
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            self.logger.info("Epoch {}/{}\n".format(epoch, num_epochs - 1))
            print("-" * 10)
            
            model.train()
            
            running_loss = 0.0
            mel_running_loss = 0.0
            
            # Iterate over data
            for data in tqdm(train_dataloader, ncols=80):
                b = data[0].shape[0]
                
                target_mel_feature_batch = data[0].to(self.device)
                
                # Split mel spectrogram into segments if needed
                if self.split_mel:
                    target_mel_feature_batch[
                        target_mel_feature_batch
                        == target_mel_feature_batch.reshape(b, -1).min(1).values.reshape(b, 1, 1)
                    ] = -1
                    target_mel_feature_batch = rearrange(
                        target_mel_feature_batch, "b d (p t)->(b p) d t", t=self.mel_len
                    )
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                out = model(target_mel_feature_batch, loss="mse")
                loss = out.loss
                mel_loss = out.mel_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * b
                mel_running_loss += mel_loss.item() * b
            
            scheduler.step()
            
            epoch_loss = running_loss / dataset_size
            epoch_mel_loss = mel_running_loss / dataset_size
            
            print("train Loss: {:.4f}".format(epoch_loss))
            print("train mel Loss: {:.4f}".format(epoch_mel_loss))
            self.logger.info("train Loss: {:.4f}".format(epoch_loss))
            self.logger.info("train mel Loss: {:.4f}".format(epoch_mel_loss))
            
            # Save best model
            save = False
            if best_valid_loss is None or best_valid_loss[1] > epoch_loss:
                best_valid_loss = (epoch, epoch_loss)
                save = True
            
            if early_stopping.early_stop(epoch_loss):
                print("Early stopping")
                stop = True
            
            if save:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"update best checkpoint: {checkpoint_path}")
                self.logger.info("update best checkpoint:{}".format(checkpoint_path))
            
            print()
            if stop:
                break
        
        print("Finish training, please take a look.")
        print("The best loss {} in epoch {},".format(best_valid_loss[1], best_valid_loss[0]))
        self.logger.info("\n" "The best loss {} in epoch {}".format(best_valid_loss[1], best_valid_loss[0]))
        
        end_time = time.time()
        run_time = math.ceil(end_time - start_time)
        print(f"time consuming: {datetime.timedelta(seconds=run_time)}")
        self.logger.info(f"time consuming: {datetime.timedelta(seconds=run_time)}")
        
        return {
            "best_epoch": best_valid_loss[0],
            "best_loss": best_valid_loss[1],
            "training_time_seconds": run_time,
        }
    
    def _show_require_grad_layers(self, model):
        """Print layers that require gradient computation"""
        print()
        print("require_grad layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
