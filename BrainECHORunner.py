import os
import traceback
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from Runners.RunnerInterface import RunnerInterface
from BrainECHOTrainer import BrainECHOTrainer
from VQVAETrainer import VQVAETrainer
from eval_utils import evaluate_brainecho
from omegaconf import ListConfig


# ============================================================================
# TRAINER FACTORY
# ============================================================================

TRAINER_REGISTRY = {
    "brainecho":      BrainECHOTrainer,
    "brainecho_biot": BrainECHOTrainer,
    "brainecho_stft": BrainECHOTrainer,
}

EVAL_REGISTRY = {
    "brainecho":      evaluate_brainecho,
    "brainecho_biot": evaluate_brainecho,
    "brainecho_stft": evaluate_brainecho,
}


class BrainECHORunner(RunnerInterface):
    """
    Unified Runner for BrainECHO experiments.

    Dispatches to BrainECHOTrainer based on cfg.model_type:
        - brainecho       → standard BrainECHO (conv EEG encoder)
        - brainecho_biot  → BrainECHO with BIOT EEG encoder
        - brainecho_stft  → BrainECHO with STFT patch encoder

    Trivial baselines (random selection, Gaussian noise) require no
    separate trainer — they are controlled by eval flags:
        eval_random_select: true   → random sentence baseline
        eval_input_noise: true     → full Gaussian noise baseline
        eval_length_noise: true    → length-preserving noise baseline
    """

    def __init__(self):
        super().__init__()
        print("BrainECHO Runner initialized")

    def run(self, cfg: DictConfig):
        """
        Main orchestration method.

        Flow:
            1. Validate configuration
            2. Determine model type
            3. (Optional) Stage 1: VQVAE pretraining
            4. Stages 2/3: Brain-audio alignment + Whisper finetuning
            5. (Optional) Evaluate
            6. Save results
        """
        print("\n" + "="*80)
        print("BRAINECHO RUNNER - STARTING EXPERIMENT")
        print("="*80)

        try:
            # ================================================================
            # CONFIGURATION
            # ================================================================

            if hasattr(cfg, 'base_output_dir') and cfg.base_output_dir:
                base_output_dir = cfg.base_output_dir
            else:
                base_output_dir = HydraConfig.get().runtime.output_dir + "/"

            print(f"\n[INFO] Output directory: {base_output_dir}")
            print("\n" + "="*80)
            print("CONFIGURATION")
            print("="*80)
            print(OmegaConf.to_yaml(cfg))
            print("="*80 + "\n")

            self._validate_config(cfg)

            # ================================================================
            # MODEL TYPE
            # ================================================================

            model_type = cfg.get('model_type', 'brainecho').lower()
            print(f"[INFO] Model type: {model_type}")

            if model_type not in TRAINER_REGISTRY:
                raise ValueError(
                    f"Unknown model_type '{model_type}'. "
                    f"Available: {list(TRAINER_REGISTRY.keys())}"
                )

            # ================================================================
            # DEVICE SETUP
            # ================================================================

            import torch
            if torch.cuda.is_available():
                if isinstance(cfg.cuda, int):
                    device = f"cuda:{cfg.cuda}"
                else:
                    device = cfg.cuda
            else:
                device = "cpu"

            print(f"[INFO] Using device: {device}")

            # ================================================================
            # STAGE SELECTION
            # ================================================================

            stages_to_run = cfg.get('stages_to_run', [2, 3])
            if isinstance(stages_to_run, (list, ListConfig)):
                stages_to_run = list(stages_to_run)
            else:
                stages_to_run = [stages_to_run]

            print(f"[INFO] Stages to run: {stages_to_run}")

            stage1_checkpoint = None
            stage2_metrics = None
            stage3_metrics = None
            stage2_checkpoint = None
            stage3_checkpoint = None

            # ================================================================
            # STAGE 1: MEL-VQVAE PRETRAINING
            # ================================================================

            if 1 in stages_to_run:
                print("\n" + "="*80)
                print("INITIALIZING STAGE 1 TRAINER (VQVAE)")
                print("="*80)

                vqvae_trainer = VQVAETrainer(
                    model=None, loss_fn=None, optimizer=None,
                    device=device, scheduler=None, metrics=None, cfg=cfg
                )
                print("[INFO] VQVAE Trainer initialized successfully")

                print("\n" + "="*80)
                print("STARTING STAGE 1 TRAINING")
                print("="*80)

                stage1_results = vqvae_trainer.fit()
                stage1_checkpoint = stage1_results['checkpoint_path']

                print("\n[INFO] Stage 1 completed successfully!")
                print(f"[INFO] VQVAE checkpoint saved: {stage1_checkpoint}")

                if 2 in stages_to_run or 3 in stages_to_run:
                    cfg.checkpoint = stage1_checkpoint
                    print(f"[INFO] Will use this checkpoint for Stages 2 & 3")

            # ================================================================
            # STAGES 2 & 3: BRAIN-AUDIO ALIGNMENT + WHISPER FINETUNING
            # ================================================================

            if 2 in stages_to_run or 3 in stages_to_run:

                trainer_cls = TRAINER_REGISTRY[model_type]
                trainer_name = trainer_cls.__name__

                print("\n" + "="*80)
                print(f"INITIALIZING TRAINER: {trainer_name}")
                print("="*80)

                trainer = trainer_cls(
                    model=None, loss_fn=None, optimizer=None,
                    device=device, scheduler=None, metrics=None, cfg=cfg
                )
                print(f"[INFO] {trainer_name} initialized successfully")

                print("\n" + "="*80)
                print(f"STARTING TRAINING: {trainer_name}")
                print("="*80)

                metrics = trainer.fit()
                stage2_metrics = metrics.get("stage2_metrics")
                stage3_metrics = metrics.get("stage3_metrics")
                stage2_checkpoint = metrics.get("stage2_checkpoint")
                stage3_checkpoint = metrics.get("stage3_checkpoint")

                print(f"\n[INFO] {trainer_name} training completed!")

            # ================================================================
            # TRAINING RESULTS
            # ================================================================

            print("\n" + "="*80)
            print("TRAINING RESULTS")
            print("="*80)

            if 1 in stages_to_run and stage1_checkpoint:
                print(f"\nStage 1 (VQVAE Pretraining):")
                print(f"  Checkpoint: {stage1_checkpoint}")

            if stage2_metrics:
                print(f"\nStage 2 (Brain-Audio Alignment):")
                print(f"  Best Epoch: {stage2_metrics.get('best_epoch', 'N/A')}")
                print(f"  Best Loss: {stage2_metrics.get('best_loss', 'N/A'):.4f}")
                print(f"  Checkpoint: {stage2_checkpoint if stage2_checkpoint else 'N/A'}")

            if stage3_metrics:
                print(f"\nStage 3 (Whisper Finetuning):")
                print(f"  Best Epoch: {stage3_metrics.get('best_epoch', 'N/A')}")
                print(f"  Best Loss: {stage3_metrics.get('best_loss', 'N/A'):.4f}")
                print(f"  Checkpoint: {stage3_checkpoint if stage3_checkpoint else 'N/A'}")

            print("\n" + "="*80)
            print("EXPERIMENT COMPLETE!")
            print("="*80 + "\n")

            # ================================================================
            # EVALUATION (Optional)
            # ================================================================

            eval_results = None
            if hasattr(cfg, 'run_evaluation') and cfg.run_evaluation:
                print("\n" + "="*80)
                print("STARTING POST-TRAINING EVALUATION")
                print("="*80)

                if not hasattr(cfg, 'eval_checkpoint') or cfg.eval_checkpoint is None:
                    if stage3_checkpoint:
                        cfg.eval_checkpoint = stage3_checkpoint
                        print(f"[INFO] Using checkpoint for evaluation: {cfg.eval_checkpoint}")
                    else:
                        print("[WARNING] No checkpoint available for evaluation")
                        eval_results = None

                if cfg.eval_checkpoint:
                    try:
                        eval_fn = EVAL_REGISTRY[model_type]
                        eval_results = eval_fn(cfg, logger=None)
                        print("[INFO] Evaluation completed successfully!")
                        try:
                            metrics["evaluation"] = eval_results
                        except UnboundLocalError:
                            pass  # stages_to_run=[] means no training ran
                    except Exception as e:
                        print(f"[WARNING] Evaluation failed: {e}")
                        print("Training results are still available.")

            # ================================================================
            # SAVE RESULTS
            # ================================================================

            if hasattr(cfg, 'save_results') and cfg.save_results:
                import json
                results_name = f"training_results_{cfg.model_type}"
                if cfg.get('eval_input_noise', False):
                    results_name += "_noise"
                if cfg.get('eval_random_select', False):
                    results_name += "_random"
                if cfg.get('eval_length_noise', False):
                    results_name += "_length_noise"
                results_path = os.path.join(cfg.save_path, f"{results_name}.json")
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, 'w') as f:
                    json_metrics = {}

                    if stage1_checkpoint:
                        json_metrics["stage1"] = {"checkpoint": str(stage1_checkpoint)}

                    if stage2_metrics:
                        json_metrics["stage2"] = {
                            "best_epoch": int(stage2_metrics["best_epoch"]) if stage2_metrics["best_epoch"] is not None else None,
                            "best_loss": float(stage2_metrics["best_loss"]) if stage2_metrics["best_loss"] is not None else None,
                            "training_time_seconds": int(stage2_metrics["training_time_seconds"]),
                            "checkpoint": str(stage2_checkpoint) if stage2_checkpoint else None
                        }

                    if stage3_metrics:
                        json_metrics["stage3"] = {
                            "best_epoch": int(stage3_metrics["best_epoch"]) if stage3_metrics["best_epoch"] is not None else None,
                            "best_loss": float(stage3_metrics["best_loss"]) if stage3_metrics["best_loss"] is not None else None,
                            "training_time_seconds": int(stage3_metrics["training_time_seconds"]),
                            "checkpoint": str(stage3_checkpoint) if stage3_checkpoint else None
                        }

                    if eval_results is not None:
                        json_metrics["evaluation"] = {
                            k: float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in eval_results.items()
                        }

                    json.dump(json_metrics, f, indent=4)
                print(f"[INFO] Results saved to: {results_path}")

            combined_metrics = {}
            if stage1_checkpoint:
                combined_metrics["stage1_checkpoint"] = stage1_checkpoint
            if stage2_metrics:
                combined_metrics["stage2_metrics"] = stage2_metrics
                combined_metrics["stage2_checkpoint"] = stage2_checkpoint
            if stage3_metrics:
                combined_metrics["stage3_metrics"] = stage3_metrics
                combined_metrics["stage3_checkpoint"] = stage3_checkpoint
            if eval_results:
                combined_metrics["evaluation"] = eval_results

            return combined_metrics

        except Exception as e:
            print("\n" + "="*80)
            print("ERROR IN BRAINECHO RUNNER")
            print("="*80)
            print(f"Error: {e}")
            print("\nFull traceback:")
            print(traceback.format_exc())
            print("="*80 + "\n")
            raise

    def _validate_config(self, cfg: DictConfig):
        """Validate required configuration parameters"""
        required_params = {
            'dataset_name': 'Dataset name (e.g., brennan, hbn, neurasearch)',
            'dataset_path': 'Path to dataset directory',
            'save_path':    'Path to save checkpoints',
            'checkpoint':   'Path to VQVAE checkpoint (Stage 1 output)',
        }

        missing = []
        for param, description in required_params.items():
            if not hasattr(cfg, param) or cfg[param] is None:
                # checkpoint is only required when actually running stages 2/3
                if param == 'checkpoint':
                    stages = cfg.get('stages_to_run', [2, 3])
                    if isinstance(stages, (list, ListConfig)):
                        stages = list(stages)
                    if 2 not in stages and 3 not in stages:
                        continue
                missing.append(f"  - {param}: {description}")

        if missing:
            raise ValueError("Missing required configuration parameters:\n" + "\n".join(missing))

        print("[INFO] Configuration validation passed")
