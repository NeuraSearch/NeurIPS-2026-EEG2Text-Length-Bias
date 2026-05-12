from torch.utils.data import Dataset
import os
import pickle
import torch


class EEG_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), "rb") as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample["input_embeddings"],
            input_sample["input_attn_mask"],
            input_sample["input_attn_mask_invert"],
            input_sample["target_ids"],
            input_sample["target_mask"],
            input_sample["target_string"],
        )


class EEG_dataset_whisper(Dataset):
    def __init__(self, path, subject="all", exclude_the_present=False, max_samples=None):
        """EEG dataset for Brennan, and HBN

        Args:
            path (str): path to dataset
            subject (str, optional): subject ID. Defaults to "all".
            exclude_the_present (bool, optional): Exclude ThePresent movie samples (HBN only). Defaults to False.
            max_samples (int, optional): Limit dataset to first N samples. Defaults to None (all samples).
        """
        self.path = path
        self.files = os.listdir(path)

        def sort_key(filename):
            parts = filename.split("-")

            if "stim-lppHK" in filename:  # LPPHK format: sub-HK001-run-1-stim-lppHK-sent-000.pickle
                subject = parts[1]  # "HK001"
                run = int(parts[3])  # 1
                sent = int(filename.split("-sent-")[1].replace(".pickle", ""))  # 0
                return (subject, run, sent)
            elif "stim-spanish" in filename:  # Spanish format: sub-002-rep-00-stim-spanish-sent-000.pickle
                subject = parts[1]
                rep = int(parts[3])
                sent = int(filename.split("-sent-")[1].replace(".pickle", ""))
                return (subject, rep, sent)
            elif "stim-" in filename:  # SparrKULee format: sub-001-stim-audiobook_5_1-sent-42.pickle
                subject = parts[1]
                sent = filename.split("-sent-")[1].replace(".pickle", "")
                return (subject, 0, int(sent))
            elif "_trial-" in filename:  # SeLis format
                subject = parts[1].split("_")[0]
                trial = filename.split("_trial-")[1].split("_")[0]
                sent = filename.split("_sent-")[1].replace(".pickle", "")
                return (subject, int(trial), int(sent))
            elif "task" in filename:  # HBN format
                subject = parts[1]
                task = parts[3]
                segment = int(parts[5].replace(".pickle", ""))
                return (subject, task, segment)
            else:  # Brennan format: sub-1-sent-5.pickle
                return (int(parts[1]), 0, int(parts[3].replace(".pickle", "")))

        self.files.sort(key=sort_key)

        if subject != "all":
            self.files = [file for file in self.files if f"sub{subject}-" in file or f"sub-{subject}-" in file]

        # Filter out ThePresent movie samples if requested (for HBN Stage 3)
        if exclude_the_present:
            print(f"[INFO]Filtering ThePresent samples from {len(self.files)} files...")
            filtered_files = []
            for file in self.files:
                with open(os.path.join(self.path, file), "rb") as handle:
                    sample = pickle.load(handle)
                    # Keep sample only if it's not from ThePresent movie
                    if sample.get("task") != "ThePresent":
                        filtered_files.append(file)

            removed_count = len(self.files) - len(filtered_files)
            self.files = filtered_files
            print(f"[INFO]Removed {removed_count} ThePresent samples, {len(self.files)} samples remaining")

        # Limit number of samples for quick testing
        if max_samples is not None and len(self.files) > max_samples:
            self.files = self.files[:max_samples]
            print(f"[INFO]Limited to {max_samples} samples for testing")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), "rb") as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample["rawEEG"],
            input_sample["input_attn_mask"],
            input_sample["target_ids"],
            input_sample["target_mask"],
            input_sample["sent_id"],
            input_sample["target_string"],
            input_sample.get("subject", "unknown"),
        )


class MEG_dataset_whisper(Dataset):
    def __init__(self, path, split="train"):
        """GWilliams EEG dataset (unused)

        Args:
            path (str): path to dataset
            split (str, optional): train, valid or test. Defaults to "train".
        """        
        path = os.path.join(path, split)
        files = os.listdir(path)
        self.input_samples = []
        for file in files:
            with open(os.path.join(path, file), "rb") as handle:
                input_sample = pickle.load(handle)
                self.input_samples.extend(input_sample)

    def __len__(self):
        return len(self.input_samples)

    def __getitem__(self, idx):
        input_sample = self.input_samples[idx]
        return (
            input_sample["MEG_start"],
            input_sample["MEG_end"],
            input_sample["target_ids"],
            input_sample["target_mask"],
            input_sample["audio_file"],
            input_sample["audio_start"],
            input_sample["audio_end"],
            input_sample["target_string"],
            input_sample["subject"],
            input_sample["session"],
            input_sample["task"],
        )
