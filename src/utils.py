import os 
import math
import torch 
import random
import logging 
import argparse  
import datetime 
import numpy as np
import pandas as pd

from pathlib import Path    
from torch.optim.lr_scheduler import _LRScheduler  


def seed_all(seed_value, cuda_deterministic=False):
    random.seed(seed_value)  
    os.environ["PYTHONHASHSEED"] = str(seed_value)  
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value) 

    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed_value)  
        torch.cuda.manual_seed_all(seed_value)  

    if cuda_deterministic:  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  
    else:  
        torch.backends.cudnn.deterministic = False  
        torch.backends.cudnn.benchmark = True  


def str_to_bool(value): 
    if isinstance(value, bool):  
        return value  

    normalized = value.strip().lower()  
    if normalized in {"true", "1", "yes", "y"}: 
        return True 
    if normalized in {"false", "0", "no", "n"}:  
        return False

    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as a boolean.")


def format_seconds(seconds):
    seconds = int(seconds)  
    hours, remainder = divmod(seconds, 3600)  
    minutes, seconds = divmod(remainder, 60) 

    if hours > 0:  
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:  
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def gpu_memory_string(device):  
    """
    Build a compact CUDA memory summary for logs
    """
    if device.type != "cuda":  
        return "cpu"

    # Convert CUDA memory to GB
    allocated = torch.cuda.memory_allocated(device) / 1024**3  
    reserved = torch.cuda.memory_reserved(device) / 1024**3  
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"GPU memory allocated={allocated:.2f}GB; reserved={reserved:.2f}GB; peak={peak:.2f}GB"  


def set_logger(log_dir, name="train"):
    """
    Create a logger that writes to both terminal and file
    """
    log_dir = Path(log_dir)  
    log_dir.mkdir(parents=True, exist_ok=True)  

    # Build timestamp to avoid overwriting old logs
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  
    log_path = log_dir / f"{name}_{timestamp}.log"  

    logger = logging.getLogger(name)  
    logger.setLevel(logging.INFO)  # keep informative training messages
    logger.handlers.clear()  
    logger.propagate = False 

    formatter = logging.Formatter(  
        fmt="[%(asctime)s] %(levelname)s - %(message)s",  # time, severity, and actual message
        datefmt="%H:%M:%S",  
    )

    # handler that writes to disk
    file_handler = logging.FileHandler(log_path)  
    file_handler.setFormatter(formatter)  

    # handler that writes to terminal
    stream_handler = logging.StreamHandler()  
    stream_handler.setFormatter(formatter)  

    logger.addHandler(file_handler) 
    logger.addHandler(stream_handler)  
    # Record where the log file is stored
    logger.info(f"Logging to {log_path}")  

    return logger 


def log(message, logger=None):  
    """
    Send a message through the logger if available, otherwise print it
    """
    if logger is None:  
        print(message, flush=True)  
    else: 
        logger.info(message)  


def unwrap_model(model):  
    return model.module if hasattr(model, "module") else model  


def strip_module_prefix(state_dict):
    return {  
        key.replace("module.", "", 1): value 
        for key, value in state_dict.items() 
    }


def load_state_dict_from_checkpoint(path, map_location="cpu"):
    """
    Load either a full checkpoint or a raw state dict
    """
    # Load the checkpoint file on the requested device
    checkpoint = torch.load(path, map_location=map_location)  
    state_dict = checkpoint.get("model_state_dict", checkpoint)  
    # Normalize keys in case the model was saved from DataParallel/DDP (as scBERT)
    state_dict = strip_module_prefix(state_dict) 
    # Return both the full checkpoint object and the normalized model weights 
    return checkpoint, state_dict 


def save_checkpoint(model, optimizer, scheduler, epoch, args, stats=None, history=None):
    ckpt_dir = Path(args.ckpt_dir)  
    ckpt_dir.mkdir(parents=True, exist_ok=True)  

    # Collect info to resume training or run inference
    checkpoint = {  
        "epoch": epoch + 1,                                     # epochs
        "model_state_dict": unwrap_model(model).state_dict(),   # model weights
        "optimizer_state_dict": optimizer.state_dict(),         # AdamW and optimizer configuration
        "scheduler_state_dict": scheduler.state_dict(),         # learning-rate scheduler state
        "args": vars(args),                                     # training arguments so generation/inference can inspect how the model was trained
    }

    # Save current epoch metrics when the caller provides them
    if stats is not None:  
        checkpoint["stats"] = stats  
    # Save the full metric history when the caller provides it
    if history is not None:  
        checkpoint["history"] = history  

    # checkpoint filename
    ckpt_path = ckpt_dir / f"{args.model_name}_{epoch + 1}.pth"  
    torch.save(checkpoint, ckpt_path)  
    return ckpt_path  


class CosineAnnealingWarmupRestarts(_LRScheduler):  
    """
    Cosine annealing with warmup and optional restarts
    """  

    def __init__(  
        self, 
        optimizer: torch.optim.Optimizer,  
        first_cycle_steps: int,  
        cycle_mult: float = 1.0,  
        max_lr: float = 0.1,  
        min_lr: float = 0.001,  
        warmup_steps: int = 0,  
        gamma: float = 1.0,  
        last_epoch: int = -1, 
    ):
        assert warmup_steps < first_cycle_steps  

        self.first_cycle_steps = first_cycle_steps 
        self.cycle_mult = cycle_mult  
        self.base_max_lr = max_lr  
        self.max_lr = max_lr  
        self.min_lr = min_lr  
        self.warmup_steps = warmup_steps  
        self.gamma = gamma  
        self.cur_cycle_steps = first_cycle_steps  
        self.cycle = 0  
        self.step_in_cycle = last_epoch 

        super().__init__(optimizer, last_epoch)  
        self.init_lr()  

    def init_lr(self): 
        """
        Initialize every parameter group to the minimum learning rate
        """
        self.base_lrs = []  
        for param_group in self.optimizer.param_groups: 
            param_group["lr"] = self.min_lr     # Set starting learning rate
            self.base_lrs.append(self.min_lr)   # Remember base learning rate

    def get_lr(self):
        """
        Compute the learning rate for the current scheduler state
        """
        # Before any scheduler step, return base learning rates
        if self.step_in_cycle == -1:  
            return self.base_lrs  

        # During warmup, increase linearly
        if self.step_in_cycle < self.warmup_steps: 
            return [  
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr  # Linear warmup formula
                for base_lr in self.base_lrs 
            ]

        # After warmup, use cosine annealing
        return [base_lr +
                (self.max_lr - base_lr)
                * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)))
                / 2 for base_lr in self.base_lrs]
        

    def step(self, epoch=None):  
        """
        Advance the scheduler by one step or jump to a specific epoch
        """
        # Move one step forward
        if epoch is None:  
            epoch = self.last_epoch + 1  
            self.step_in_cycle += 1  

            if self.step_in_cycle >= self.cur_cycle_steps:  
                self.cycle += 1  
                self.step_in_cycle -= self.cur_cycle_steps  
                self.cur_cycle_steps = int(  
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult 
                ) + self.warmup_steps  

        # Jump to a requested epoch
        else:  
            if epoch >= self.first_cycle_steps:  
                if self.cycle_mult == 1.0:  
                    self.step_in_cycle = epoch % self.first_cycle_steps  
                    self.cycle = epoch // self.first_cycle_steps  
                else:  
                    n = int(  
                        math.log(  
                            epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1,  
                            self.cycle_mult,  
                        )
                    )
                    self.cycle = n  
                    self.step_in_cycle = epoch - int(  
                        self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)  
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult**n  
            else: 
                self.cur_cycle_steps = self.first_cycle_steps 
                self.step_in_cycle = epoch  

        # Decay the maximum learning rate by cycle number
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)  
        # Update the scheduler epoch counter
        self.last_epoch = math.floor(epoch)  

        # Apply the computed learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):  
            param_group["lr"] = lr  


def construct_marker_weights(
    marker_df: pd.DataFrame,
    min_logfc: float = 1.0,
    max_extra_weight: float = 2.0
) -> pd.DataFrame:
    
    """
    Ordinary genes have weight 1.

    Retained markers have weight in:
        [1, 1 + max_extra_weight]
    """

    df = marker_df.copy()

    # Keep reliable positive markers
    df = df[
        (df["scores"] > 0)
        & (df["logfc"] >= min_logfc)
        & (df["adj_p_val"] < 0.01)
    ].copy()

    weighted_groups = []

    for cell_type, group in df.groupby("cell_type"):
        group = group.copy()

        logfc = group["logfc"].to_numpy(dtype=np.float64)

        # Robust normalization within this cell type
        lower = np.percentile(logfc, 10)
        upper = np.percentile(logfc, 90)

        if upper <= lower:
            importance = np.ones_like(logfc)

        else:
            importance = (logfc - lower) / (upper - lower)
            importance = np.clip(importance, 0.0, 1.0)

        group["marker_importance"] = importance
        group["loss_weight"] = (
            1.0 + max_extra_weight * importance
        )

        weighted_groups.append(group)

    return pd.concat(weighted_groups, ignore_index=True)


def build_marker_weight_vectors(
    weighted_markers: pd.DataFrame,
    cell_type_to_id: dict,
    num_genes: int,
    device: torch.device
):
    
    marker_weights = {}

    for cell_type_name, cell_type_id in cell_type_to_id.items():

        weights = torch.ones(
            num_genes,
            dtype=torch.float32,
            device=device
        )

        rows = weighted_markers[
            weighted_markers["cell_type"] == cell_type_name
        ]

        indices = torch.as_tensor(
            rows["gene_idx"].to_numpy(),
            dtype=torch.long,
            device=device
        )

        values = torch.as_tensor(
            rows["loss_weight"].to_numpy(),
            dtype=torch.float32,
            device=device
        )

        weights[indices] = values
        marker_weights[int(cell_type_id)] = weights

    return marker_weights


def sample_specific_weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    celltype_ids: torch.Tensor,
    marker_weights: dict,       # for each cell type we have the info of its markers
    ignore_index=-100
):
    """
    Logits: [B, K, G]
    labels: [B, G]
    """

    per_gene_loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        reduction="none"
    )

    # Take only those which are masked
    valid = labels != ignore_index
    weights = torch.ones_like(per_gene_loss)    # [B, G]

    for sample_idx, cell_type_id in enumerate(celltype_ids.tolist()):
        weights[sample_idx] = marker_weights[int(cell_type_id)]
    
    # Multiply each sample loss with its marker importance
    return (
        per_gene_loss[valid] * weights[valid]
    ).sum() / weights[valid].sum().clamp_min(1.0)
