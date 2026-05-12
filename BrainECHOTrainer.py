"""
BrainECHO Trainer - Stages 2 & 3

Handles brain-audio alignment (Stage 2) and Whisper finetuning (Stage 3)
Inherits from engine's TrainingInterface

"""

import os
import sys
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
import copy
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import pickle
import math
from einops.einops import rearrange
from peft import AdaLoraConfig

from Components.TrainingLayer.TrainingInterface import TrainingInterface
from dataset_configs import get_dataset_config
from dataset import EEG_dataset_whisper
from model import BrainECHO, BrainECHOBIOT, BrainECHOSTFT
from utils.utils import *
from optim_new import *
from omegaconf import OmegaConf

class BrainECHOTrainer(TrainingInterface):
    """
    Trainer for BrainECHO Stages 2 & 3
    
    Stage 2: Brain-audio alignment
    Stage 3: Whisper finetuning with AdaLoRA
    
    Unlike standard trainers, this wraps the complete BrainECHO training pipeline
    which includes two sequential stages with checkpoint management.
    """
    
    def __init__(self, model, loss_fn, optimizer, device="cpu", scheduler=None, metrics=None, cfg=None):
        """
        Initialize BrainECHO Trainer
        
        Args:
            model: BrainECHO model (can be None, will be created from cfg)
            loss_fn: Loss function (not used, BrainECHO has internal losses)
            optimizer: Optimizer (not used, created internally per stage)
            device: Device for training
            scheduler: Learning rate scheduler (not used, created internally)
            metrics: Metrics dict (not used)
            cfg: Hydra configuration with all BrainECHO parameters
        """
        # Don't call super().__init__() because we manage everything differently
        self.device = device
        self.cfg = cfg
        
        # Extract parameters from config
        self.dataset_name = cfg.dataset_name
        self.dataset_path = cfg.dataset_path
        self.save_path = cfg.save_path
        self.subject = cfg.get('subject', 'all')
        self.split_mel = cfg.get('split_mel', False)
        self.latent_loss = cfg.get('latent_loss', 'mse')
        self.split_len = cfg.get('split_len', 4) * 100
        self.codebook_size = cfg.get('cb_size', 2048)
        self.r = cfg.get('ratio', 4)
        self.exclude_the_present_stage3 = cfg.get('exclude_the_present_stage3', False)
        self.max_samples = cfg.get('max_samples', None)
        
        # Get dataset configuration
        self.dataset_config = get_dataset_config(self.dataset_name)
        
        # Determine mel length
        self.mel_len = self.dataset_config.get_mel_length(
            self.dataset_path, self.split_mel, self.split_len
        )
        
        # Initialize logger
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        init_logger(config_dict)
        self.logger = getLogger()
        
        print(f"\n{'='*80}")
        print(f"BrainECHO Trainer Initialized")
        print(f"Dataset: {self.dataset_config.name.upper()}")
        print(f"Description: {self.dataset_config.description}")
        print(f"Device: {self.device}")
        print(f"Mel length: {self.mel_len}")
        print(f"{'='*80}\n")
    
    def load_dataset(self, phase, exclude_the_present=False):
        """Load dataset for given phase"""
        dataset_kwargs = {"path": self.dataset_path + phase, "subject": self.subject}
        
        if self.dataset_config.name == "hbn":
            dataset_kwargs["exclude_the_present"] = exclude_the_present
        if self.max_samples is not None:
            dataset_kwargs["max_samples"] = self.max_samples
        
        return EEG_dataset_whisper(**dataset_kwargs)
    
    def train_step(self, batch):
        """
        Single training step (required by TrainingInterface but not used)
        BrainECHO handles training internally
        """
        raise NotImplementedError("BrainECHO uses custom training loop in fit()")
    
    def validation_step(self, batch):
        """
        Single validation step (required by TrainingInterface but not used)
        BrainECHO handles validation internally
        """
        raise NotImplementedError("BrainECHO uses custom training loop in fit()")
    
    def fit(self, train_loader=None, val_loader=None, epochs=None):
        """
        Train BrainECHO model through Stages 2 and 3
        
        Stage 2: Brain-audio alignment
        Stage 3: Whisper finetuning with AdaLoRA
        
        Args:
            train_loader: Ignored (data loaded internally)
            val_loader: Ignored (data loaded internally)
            epochs: Ignored (epochs from config)
            
        Returns:
            dict: Training metrics
        """
        print("\n" + "="*80)
        print("STARTING BRAINECHO TRAINING (STAGES 2 & 3)")
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
        
        # ====================================================================
        # STAGE 2: BRAIN-AUDIO ALIGNMENT
        # ====================================================================
        
        print("\n" + "="*80)
        print("STAGE 2: BRAIN-AUDIO ALIGNMENT")
        print("="*80 + "\n")
        
        # Load datasets for Stage 2
        print("[INFO] Loading Stage 2 datasets...")
        train_set = self.load_dataset("train", exclude_the_present=False)
        valid_set = self.load_dataset("valid", exclude_the_present=False)
        dataset_sizes = {"train": len(train_set), "dev": len(valid_set)}
        
        print(f"[INFO] Stage 2 train_set size: {len(train_set)}")
        print(f"[INFO] Stage 2 dev_set size: {len(valid_set)}")
        
        if len(train_set) == 0:
            raise ValueError("Training set is empty!")

        # Initialize model
        model_type = self.cfg.get('model_type', 'brainecho')
        if model_type == 'brainecho_biot':
            print(f"[INFO] Initializing BrainECHOBIOT model with {self.dataset_config.in_channels} channels...")
            model = BrainECHOBIOT(
                None,
                in_chan=self.dataset_config.in_channels,
                d_model=256,
                ffn_dim=1024,
                num_layers=4,
                mel_len=self.mel_len,
                codebook_size=self.codebook_size,
                r=self.r,
                split_mel=self.split_mel,
            )
        elif model_type == 'brainecho_stft':
            print(f"[INFO] Initializing BrainECHOSTFT model with {self.dataset_config.in_channels} channels...")
            model = BrainECHOSTFT(
                None,
                in_chan=self.dataset_config.in_channels,
                d_model=256,
                ffn_dim=1024,
                num_layers=4,
                mel_len=self.mel_len,
                codebook_size=self.codebook_size,
                r=self.r,
                split_mel=self.split_mel,
            )
        else:
            print(f"[INFO] Initializing BrainECHO model with {self.dataset_config.in_channels} channels...")
            model = BrainECHO(
                None,
                in_chan=self.dataset_config.in_channels,
                d_model=256,
                ffn_dim=1024,
                num_layers=4,
                mel_len=self.mel_len,
                codebook_size=self.codebook_size,
                r=self.r,
                split_mel=self.split_mel,
            )
        model.mel_vqvae.load_state_dict(torch.load(self.cfg.checkpoint, map_location=self.device))
        processor = WhisperProcessor.from_pretrained(self.dataset_config.whisper_model)
        model.to(self.device)
        model.task = "task1"
        
        # Training parameters for Stage 2
        num_epoch_finetune = self.cfg.num_epoch_finetune[0]
        lr_finetune = self.cfg.lr_finetune[0]
        batch_size = self.cfg.batch_size[0]
        T = self.cfg.T_max[0]
        
        print(f"[INFO] Stage 2 - Epochs: {num_epoch_finetune}, LR: {lr_finetune}, Batch: {batch_size}")
        
        # Dataloaders
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                     num_workers=4, pin_memory=True)
        valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, 
                                     num_workers=4, pin_memory=True)
        dataloaders = {"train": train_dataloader, "dev": valid_dataloader}
        
        # Optimizer and scheduler
        early_stopping = EarlyStopper(patience=4, min_delta=0.0)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        config_dict["lr_finetune"] = lr_finetune
        optimizer = build_optimizer(config_dict, model, mode="finetune")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
        
        # Checkpoint path
        save_name = f"b{batch_size}_{num_epoch_finetune}_{lr_finetune}_alignment"
        if self.subject != "all":
            save_name = f"sub{self.subject}_b{batch_size}_{num_epoch_finetune}_{lr_finetune}_alignment"
        checkpoint_path_stage2 = f"{self.save_path}/{save_name}.pt"
        print(f"[INFO] Stage 2 checkpoint: {checkpoint_path_stage2}")

        # Load mel features
        mel_features_path = self.dataset_path + self.dataset_config.mel_features_path_pattern
        with open(mel_features_path, "rb") as f:
            log_mel_features = pickle.load(f)
        if isinstance(log_mel_features, np.ndarray):
            log_mel_features = torch.from_numpy(log_mel_features).float()
        print(f"[INFO] Mel features shape: {tuple(log_mel_features.shape)}  dtype={log_mel_features.dtype}")
        
        # Noise samples for robustness
        actual_mel_len = log_mel_features.shape[2]
        noise_audio = [0] * int(actual_mel_len / 100 * 16000)
        noise_mel = processor(noise_audio, sampling_rate=16000, return_tensors="pt").input_features
        noise_mel = noise_mel[..., :actual_mel_len].to(self.device)
        if self.split_mel:
            noise_mel = torch.clip(noise_mel, min=-1)
        
        # Train Stage 2
        print("\n[INFO] Starting Stage 2 training...")
        self._show_require_grad_layers(model)
        stage2_metrics = self._train_stage(
            model, dataloaders, optimizer, scheduler, early_stopping,
            num_epoch_finetune, checkpoint_path_stage2, dataset_sizes,
            log_mel_features, noise_mel, stage=2
        )
        print("[INFO] Stage 2 training complete!")
        
        # ====================================================================
        # STAGE 3: WHISPER FINETUNING
        # ====================================================================
        
        print("\n" + "="*80)
        print("STAGE 3: WHISPER FINETUNING")
        print("="*80 + "\n")
        
        model.task = "task2"
        del noise_mel
        
        # Load best model from Stage 2
        print(f"[INFO] Loading best model from Stage 2: {checkpoint_path_stage2}")
        model.load_state_dict(torch.load(checkpoint_path_stage2, map_location=self.device), strict=True)
        
        # Training parameters for Stage 3
        num_epoch_finetune = self.cfg.num_epoch_finetune[1]
        lr_finetune = self.cfg.lr_finetune[1]
        batch_size = self.cfg.batch_size[1]
        T = self.cfg.T_max[1]
        
        print(f"[INFO] Stage 3 - Epochs: {num_epoch_finetune}, LR: {lr_finetune}, Batch: {batch_size}")
        
        # Reload datasets if needed
        if self.dataset_config.reload_dataset_stage3:
            print("[INFO] Reloading datasets for Stage 3...")
            if self.exclude_the_present_stage3:
                print("[INFO] Excluding ThePresent from Stage 3")
            train_set = self.load_dataset("train", exclude_the_present=self.exclude_the_present_stage3)
            valid_set = self.load_dataset("valid", exclude_the_present=self.exclude_the_present_stage3)
            dataset_sizes = {"train": len(train_set), "dev": len(valid_set)}
            print(f"[INFO] Stage 3 train_set size: {len(train_set)}")
            print(f"[INFO] Stage 3 dev_set size: {len(valid_set)}")
        
        # Dataloaders
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                     num_workers=4, pin_memory=True)
        valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, 
                                     num_workers=4, pin_memory=True)
        dataloaders = {"train": train_dataloader, "dev": valid_dataloader}
        
        # Optimizer and scheduler
        early_stopping = EarlyStopper(patience=4, min_delta=0.0)
        config_dict["lr_finetune"] = lr_finetune
        optimizer = build_optimizer(config_dict, model, mode="finetune")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
        
        # Checkpoint path
        save_name = f"b{batch_size}_{num_epoch_finetune}_{lr_finetune}_finetune"
        if self.subject != "all":
            save_name = f"sub{self.subject}_b{batch_size}_{num_epoch_finetune}_{lr_finetune}_finetune"
        checkpoint_path_stage3 = f"{self.save_path}/{save_name}.pt"
        print(f"[INFO] Stage 3 checkpoint: {checkpoint_path_stage3}")
        
        # Add pretrained Whisper
        print("[INFO] Loading pretrained Whisper model...")
        pretrained = WhisperForConditionalGeneration.from_pretrained(self.dataset_config.whisper_model).to(self.device)
        
        # Freeze modules
        if model_type == 'brainecho_biot':
            freeze_modules = ["biot_encoder", "temporal_pool"]
        elif model_type == 'brainecho_stft':
            freeze_modules = ["patch_embedding", "channel_tokens",
                              "positional_encoding", "transformer", "temporal_pool"]
        else:
            freeze_modules = ["TS_Conv", "transformer", "pos_embed_e"]
        for module in freeze_modules:
            for p in getattr(model, module).parameters():
                p.requires_grad = False
        for p in pretrained.parameters():
            p.requires_grad = False
        
        # Configure AdaLoRA
        total_step = math.ceil(dataset_sizes["train"] / batch_size) * num_epoch_finetune
        tinit, tfinal = self.dataset_config.get_adalora_params(total_step)
        tinit = min(tinit, max(1, total_step // 10))
        tfinal = min(tfinal, max(tinit + 1, total_step // 2))
        print(f"[INFO] AdaLoRA config: tinit={tinit}, tfinal={tfinal}")
        
        config = AdaLoraConfig(
            init_r=12, target_r=4, tinit=tinit, tfinal=tfinal, total_step=total_step,
            deltaT=10, lora_alpha=32, lora_dropout=0.05,
            target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        model.add_adalora_pretrained_model(pretrained, config)
        
        # Train Stage 3
        print("\n[INFO] Starting Stage 3 training...")
        self._show_require_grad_layers(model)
        stage3_metrics = self._train_stage(
            model, dataloaders, optimizer, scheduler, early_stopping,
            num_epoch_finetune, checkpoint_path_stage3, dataset_sizes,
            log_mel_features=None, noise_mel=None, stage=3
        )
        print("[INFO] Stage 3 training complete!")
        
        print("\n" + "="*80)
        print("BRAINECHO TRAINING COMPLETE!")
        print("="*80)
        print(f"Stage 2 checkpoint: {checkpoint_path_stage2}")
        print(f"Stage 3 checkpoint: {checkpoint_path_stage3}")
        print("="*80 + "\n")
        
        return {
            "stage2_metrics": stage2_metrics,
            "stage3_metrics": stage3_metrics,
            "stage2_checkpoint": checkpoint_path_stage2,
            "stage3_checkpoint": checkpoint_path_stage3,
        }
    
    def _train_stage(self, model, dataloaders, optimizer, scheduler, early_stopping,
                     num_epochs, checkpoint_path, dataset_sizes, log_mel_features=None, 
                     noise_mel=None, stage=2):
        """
        Train a single stage (Stage 2 or Stage 3)
        
        Args:
            model: BrainECHO model
            dataloaders: Dict with train and dev dataloaders
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            early_stopping: Early stopping object
            num_epochs: Number of epochs
            checkpoint_path: Path to save checkpoint
            dataset_sizes: Dict with dataset sizes
            log_mel_features: Mel features (Stage 2 only)
            noise_mel: Noise mel (Stage 2 only)
            stage: Stage number (2 or 3)
            
        Returns:
            dict: Training metrics
        """
        start_time = time.time()
        best_valid_loss = None
        step = 0
        stop = False
        phases = ("train", "dev")
        
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            self.logger.info("Epoch {}/{}\n".format(epoch, num_epochs - 1))
            print("-" * 10)
            
            for phase in phases:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                mel_running_loss = 0.0
                
                for rawEEG, input_masks, target_ids, target_mask, sent_ids, target_string, subject_ids in tqdm(
                    dataloaders[phase], ncols=80
                ):
                    b = rawEEG.shape[0]
                    
                    # Get EEG length
                    if self.dataset_config.eeg_length_mode == "dynamic":
                        max_eeg_length = rawEEG.shape[-1]
                    else:
                        max_eeg_length = self.dataset_config.fixed_eeg_length
                    
                    rawEEG_batch = rawEEG[..., :max_eeg_length].to(self.device).float()
                    
                    if self.split_mel:
                        rawEEG_batch = rearrange(rawEEG_batch, "b c (p t)->(b p) c t", t=self.mel_len * 2)
                    target_mask_batch = target_mask.to(self.device)
                    
                    if model.task == "task1":  # Stage 2
                        target_ids_batch = None
                        target_mel_feature_batch = log_mel_features.index_select(0, sent_ids).to(self.device)
                        if self.split_mel:
                            target_mel_feature_batch[
                                target_mel_feature_batch
                                == target_mel_feature_batch.reshape(b, -1).min(1).values.reshape(b, 1, 1)
                            ] = -1
                            target_mel_feature_batch = rearrange(
                                target_mel_feature_batch, "b d (p t)->(b p) d t", t=self.mel_len
                            )
                        if random.random() < 0.05:
                            noise = torch.randn_like(rawEEG_batch[:1])
                            rawEEG_batch = torch.cat([rawEEG_batch, noise])
                            target_mel_feature_batch = torch.cat([target_mel_feature_batch, noise_mel])
                    else:  # Stage 3
                        target_ids_batch = target_ids.to(self.device)
                        target_ids_batch[target_mask_batch == 0] = -100
                        target_mel_feature_batch = None
                    
                    optimizer.zero_grad()
                    
                    if phase == "train":
                        out = model(rawEEG_batch, target_ids_batch, target_mel_feature_batch, loss=self.latent_loss)
                    else:
                        with torch.no_grad():
                            out = model(rawEEG_batch, target_ids_batch, target_mel_feature_batch, loss=self.latent_loss)
                    
                    loss = out.loss
                    if model.task == "task1":
                        mel_loss = out.mel_loss
                        del target_mel_feature_batch
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        if model.task == "task2":
                            model.pretrained.model.encoder.update_and_allocate(step)
                            step += 1
                    
                    running_loss += loss.item() * rawEEG_batch.size()[0]
                    if model.task == "task1":
                        mel_running_loss += mel_loss.item() * rawEEG_batch.size()[0]
                
                if phase == "train":
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                print("{} Loss: {:.4f}".format(phase, epoch_loss))
                self.logger.info("{} Loss: {:.4f}".format(phase, epoch_loss))
                
                if model.task == "task1":
                    epoch_mel_loss = mel_running_loss / dataset_sizes[phase]
                    print("{} mel Loss: {:.4f}".format(phase, epoch_mel_loss))
                    self.logger.info("{} mel Loss: {:.4f}".format(phase, epoch_mel_loss))
                
                if phase == "dev":
                    save = False
                    if best_valid_loss is None or best_valid_loss[1] > epoch_loss:
                        best_valid_loss = (epoch, epoch_loss)
                        save = True
                    
                    if early_stopping.early_stop(epoch_loss):
                        print("Early stopping")
                        stop = True
                    
                    if save:
                        if model.task == "task1":
                            torch.save(model.state_dict(), checkpoint_path)
                        elif model.task == "task2":
                            temp_model = copy.deepcopy(model)
                            temp_model.pretrained.model.encoder = temp_model.pretrained.model.encoder.merge_and_unload()
                            torch.save(temp_model.state_dict(), checkpoint_path)
                        print(f"update best on dev checkpoint: {checkpoint_path}")
                        self.logger.info("update best on dev checkpoint:{}".format(checkpoint_path))
            
            print()
            if stop:
                break
        
        end_time = time.time()
        run_time = math.ceil(end_time - start_time)
        print(f"Stage {stage} time consuming: {datetime.timedelta(seconds=run_time)}")
        self.logger.info(f"Stage {stage} time consuming: {datetime.timedelta(seconds=run_time)}")
        
        return {
            "best_epoch": best_valid_loss[0] if best_valid_loss else None,
            "best_loss": best_valid_loss[1] if best_valid_loss else None,
            "training_time_seconds": run_time,
        }
    
    def _show_require_grad_layers(self, model):
        """Print layers that require gradient computation"""
        print()
        print("require_grad layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
