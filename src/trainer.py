"""
Finetuning scBERT using diffusion-style reconstruction
"""  

import json
import math
import time
import torch
import random  
import argparse 
import numpy as np
import pandas as pd

from pathlib import Path  
from torch import nn  
from torch.optim import AdamW  
from torch.utils.data import DataLoader 
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm 
except ImportError:
    tqdm = None  

from model import DiffScBERT 
from dataset import DiffscBERTDataset
from visualize import MarkerPlotAccumulator, plot_accumulated_markers
from ablation import load_marker_data
from utils import ( 
    CosineAnnealingWarmupRestarts,  
    format_seconds,
    gpu_memory_string, 
    load_state_dict_from_checkpoint, 
    log,  
    save_checkpoint, 
    seed_all, 
    set_logger, 
    str_to_bool,
    sample_specific_weighted_ce,
    construct_marker_weights,
    build_marker_weight_vectors
)


# Tell CrossEntropyLoss to ignore positions that were not masked
IGNORE_INDEX = -100 
NUM_CLASSES = 7 # 5 bins + CLS + dummy bin


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma=1.5,
        ignore_index=-100,
        reduction="mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer(
                "alpha",
                torch.as_tensor(alpha, dtype=torch.float32),
            )

    def forward(self, logits, targets):
        """
        logits:  [B, K, G]
        targets: [B, G]
        """

        valid = targets != self.ignore_index

        if not valid.any():
            return logits.sum() * 0.0

        safe_targets = targets.clone()
        safe_targets[~valid] = 0

        log_probs = F.log_softmax(logits, dim=1)  # [B, K, G]
        log_pt = log_probs.gather(dim=1, index=safe_targets.unsqueeze(1)).squeeze(1)  # [B, G]

        pt = log_pt.exp()

        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = -focal_weight * log_pt  # [B, G]

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[safe_targets]  # [B, G]
            loss = loss * alpha_t

        loss = loss[valid]

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            return loss
        

def get_mask_subset_with_prob(mask, prob):
    """
    Build a boolean mask that selects about "prob" valid tokens per sequence
    """
    # if requested probability is zero or negative, do not mask anything (boolean mask full of False values)
    if prob <= 0:  
        return torch.zeros_like(mask, dtype=torch.bool)

    batch_size, seq_len = mask.shape 
    max_masked = max(1, math.ceil(prob * seq_len))  
    num_valid_tokens = mask.sum(dim=-1, keepdim=True)  # num positions actually maskable in each sample

    mask_excess = torch.arange(max_masked, device=mask.device).unsqueeze(0)  # slot indices from 0 to max_masked-1
    mask_excess = mask_excess >= torch.ceil(num_valid_tokens * prob)  # mark slots that exceed the desired number for each sample

    # Create a mask of random numbers and fill invalid positions with 1e-9
    rand = torch.rand((batch_size, seq_len), device=mask.device).masked_fill(~mask, -1e9)

    # Sample topk values (k = number of tokens to mask)
    _, sampled_indices = rand.topk(max_masked, dim=-1)  # positions with highest random scores

    # sampled_indices = (sampled_indices + 1).masked_fill(mask_excess, 0)  # reserve index 0 as a dummy slot
    sampled_indices = sampled_indices.masked_fill(mask_excess, 0)
    new_mask = torch.zeros((batch_size, seq_len), device=mask.device, dtype=torch.long)  # create a all false mask
    new_mask.scatter_(-1, sampled_indices, 1)  # substitute with true
    return new_mask.bool()  # convert the result to a boolean mask


class DiffscBERTCollator: 
    """
    Collate function passed to the DataLoader. It:
    1. takes the tokenized data produced in dataset.py
    2. appends the CLS token at the end
    3. chooses the current masking-probability range from the epoch
    4. samples one masking probability for the whole batch
    5. masks the batch with that probability
    """

    def __init__( 
        self, 
        cls_token_id,  
        mask_token_id,  
        total_epochs,  
        p_start=0.1,                    # min masking probability at the beginning of training
        p_end=1.0,                      # max masking probability at the end of training
        max_zero_mask_fraction = 0.6,
        num_probability_bins=3,         # number of probability intervals spanning p_start to p_end
        final_random_mask_pct=0.0 
    ):
        self.cls_token_id = cls_token_id  
        self.mask_token_id = mask_token_id  
        self.total_epochs = total_epochs
        self.p_start = p_start
        self.p_end = p_end
        self.max_zero_mask_fraction = max_zero_mask_fraction
        self.num_probability_bins = num_probability_bins
        self.final_random_mask_pct = final_random_mask_pct 
        self.current_epoch = 0 

        if not (0.0 <= self.final_random_mask_pct < 1.0):
            raise ValueError("final_random_mask_pct must be in [0.0, 1.0).")

        # Build interval boundaries between the start and end probs, and convert consecutive boundaries into prob ranges such as (0.1, 0.4)
        edges = np.linspace(p_start, p_end, num_probability_bins + 1, dtype=np.float32)  
        self.mask_ranges = list(zip(edges[:-1], edges[1:])) 

    def set_epoch(self, epoch):  
        """
        Update the epoch index before the DataLoader starts producing batches
        """
        self.current_epoch = epoch  


    def get_current_range(self): 
        """
        Determine the masking-probability interval for the current epoch.
        First phase: Curriculum over num_probability_bins intervals.
        Final phase: Random full-range masking from [p_start, p_end].

        For instance:
            --epochs 100 --p_start 0.1 --p_end 0.5 --num_mask_steps 5 --final_random_mask_pct 0.10''
                - epochs (1:90) -> 5 curriculum intervals
                - epochs (91:100)-> random mask_prob from 0.1 to 0.5

        """
        final_random_epochs = int(round(self.total_epochs * self.final_random_mask_pct))
        curriculum_epochs = self.total_epochs - final_random_epochs

        # Phase 1: normal masking range
        if self.current_epoch < curriculum_epochs:
            if len(self.mask_ranges) == 1 or curriculum_epochs <= 1:
                step_id = 0
            else:
                progress = self.current_epoch / max(curriculum_epochs - 1, 1)
                step_id = min(
                    int(progress * len(self.mask_ranges)),
                    len(self.mask_ranges) - 1,
                )

            mask_range = self.mask_ranges[step_id]
            schedule_mode = "curriculum"

            return step_id, mask_range, schedule_mode

        # Phase 2: random full-range
        else:
            step_id = None 
            mask_range = (self.p_start, self.p_end)
            schedule_mode = "random_full_range"

            return step_id, mask_range, schedule_mode 


    def mask_prob_to_step_id(self, mask_prob):
        """
        Convert a sampled mask probability into the matching discrete step id.
        Used during the final random full-range phase, where mask_prob
        can come from anywhere in [p_start, p_end].
        """
        width = (self.p_end - self.p_start) / self.num_probability_bins

        step_id = int((mask_prob - self.p_start) / width)

        step_id = max(0, step_id)
        step_id = min(step_id, self.num_probability_bins - 1)

        return step_id


    def __call__(self, batch):
        
        # Stack tokenized gene sequences from all samples in the batch and the target cell-type ids for the whole batch
        full_tokens = torch.cat([sample["data"].long() for sample in batch], dim=0)  
        celltype_ids = torch.cat([sample["label"].long().view(-1) for sample in batch], dim=0)  

        # Create one CLS token per sample and append it at the end of each gene sequence
        # cls_column = torch.full((tokens.size(0), 1), self.cls_token_id, dtype=torch.long)  
        # full_tokens = torch.cat((tokens, cls_column), dim=1)  

        # Look up the current masking bin and its prob interval and sample one masking prob for the entire batch
        step_id, mask_range, schedule_mode = self.get_current_range() 
        mask_prob = random.uniform(*mask_range)  

        if schedule_mode == "random_full_range":
            step_id = self.mask_prob_to_step_id(mask_prob)

        B, L = full_tokens.shape

        # Exclude CLS token
        gene_positions = torch.ones_like(full_tokens, dtype=torch.bool)
        gene_positions[:, -1] = False

        is_zero = full_tokens == 0
        is_nonzero = gene_positions & ~is_zero
        is_zero = gene_positions & is_zero

        mask = torch.zeros_like(full_tokens, dtype=torch.bool)

        p = mask_prob  # or however you compute mask_prob
        max_zero_fraction = self.max_zero_mask_fraction  # e.g. 0.60

        num_gene_tokens = L - 1
        num_to_mask = max(1, int(round(p * num_gene_tokens)))

        for i in range(B):
            zero_idx = torch.where(is_zero[i])[0]
            nonzero_idx = torch.where(is_nonzero[i])[0]

            # Desired split
            max_zero_to_mask = int(round(max_zero_fraction * num_to_mask))
            nonzero_to_mask = num_to_mask - max_zero_to_mask

            # But cannot sample more nonzeros than available
            actual_nonzero_to_mask = min(nonzero_to_mask, nonzero_idx.numel())

            # If not enough nonzeros, give remaining quota back to zeros
            actual_zero_to_mask = min(
                num_to_mask - actual_nonzero_to_mask,
                zero_idx.numel(),
            )

            if actual_nonzero_to_mask > 0:
                perm = torch.randperm(nonzero_idx.numel(), device=full_tokens.device)
                selected_nonzero = nonzero_idx[perm[:actual_nonzero_to_mask]]
                mask[i, selected_nonzero] = True

            if actual_zero_to_mask > 0:
                perm = torch.randperm(zero_idx.numel(), device=full_tokens.device)
                selected_zero = zero_idx[perm[:actual_zero_to_mask]]
                mask[i, selected_zero] = True

        # scBERT uses token 0 both for zero-expression genes and for CLS
        # To stay consistent with that behavior, token 0 is never masked

        masked_tokens = full_tokens.clone() 
        masked_tokens[mask] = self.mask_token_id  # Replace the selected positions with the MASK token id

        labels = full_tokens.clone()  
        # Ignore all positions not masked because they do not contribute to the loss, and also ignore the CLS pos
        labels[~mask] = IGNORE_INDEX  
        labels[:, -1] = IGNORE_INDEX  

        # Step id corresponds to a masking interval
        # e.g. (0.1, 0.4) -> 1, (0.4, 0.7) -> 2, ...
        step_ids = torch.full((full_tokens.size(0),), step_id, dtype=torch.long)  

        return {  
            "input_ids": masked_tokens,                                 # Send masked sequence to the model
            "labels": labels,                                           # Send original tokens only at masked positions as reconstruction targets
            "celltype_ids": celltype_ids,                               # Send the conditioning cell-type ids
            "step_ids": step_ids,                                       # Send the diffusion-step ids corresponding to the current masking bin
            "mask_prob": torch.tensor(mask_prob, dtype=torch.float32),  # Keep the sampled masking probability for logging
            "schedule_mode": schedule_mode,
        }


def build_pretrained_encoder(num_tokens, seq_len, args):
    """
    Create the scBERT Performer encoder and optionally load pretrained weights
    """

    try:
        # import the scBERT PerformerLM implementation 
        from performer_pytorch.performer_pytorch import PerformerLM
    except ImportError as exc:
        raise ImportError(
            "Could not import PerformerLM from the local performer_pytorch folder. "
        ) from exc

    # create the same PerformerLM backbone architecture
    encoder = PerformerLM(
        num_tokens=num_tokens,
        dim=args.dim,
        depth=args.depth,
        max_seq_len=seq_len,
        heads=args.heads,
        local_attn_heads=args.local_attn_heads,
        g2v_position_emb=args.pos_embed,
    )

    # optionally load pretrained scBERT weights .pth
    if args.model_path:
        model_path = Path(args.model_path)

        # load the checkpoint and normalize possible DataParallel/DDP key prefixes
        _, state_dict = load_state_dict_from_checkpoint(
            model_path,
            map_location="cpu",
        )

        # load compatible pretrained parameters into the Performer encoder
        encoder.load_state_dict(state_dict, strict=False)

    return encoder


def composite_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    celltype_ids: torch.Tensor,
    marker_weights: dict,
    loss_fn,
    lambda_sparse: float = 0.0,
    lambda_marker: float = 0.5,
    zero_bin_id: int = 0,
):
    """
    logits: [B, K, G]
    labels: [B, G]

    Returns:
        composite_loss, recon_loss, sparsity_loss
    """

    # Reconstruction loss -> Focal loss
    recon_loss = loss_fn(logits, labels)

    # Markers specific loss
    marker_loss = sample_specific_weighted_ce(
        logits,
        labels,
        celltype_ids,
        marker_weights,
    )

    valid = labels != IGNORE_INDEX

    # Real sparsity
    real_sparsity = (labels[valid] == zero_bin_id).float().mean()

    # Predicted sparsity
    probs = logits.softmax(dim=1)
    zero_probs = probs[:, zero_bin_id, :]
    pred_sparsity = zero_probs[valid].mean()

    # Sparsity loss
    sparsity_loss = torch.abs(real_sparsity - pred_sparsity)

    # Composite loss
    composite = recon_loss + lambda_sparse * sparsity_loss + lambda_marker * marker_loss

    return composite, recon_loss, sparsity_loss, marker_loss


def train_one_epoch(
        model, dataloader, optimizer, loss_fn, 
        device, grad_accum_steps, epoch, total_epochs, 
        log_every, lambda_sparse, lambda_marker,
        num_cell_types, idx_to_cell_type, marker_weights, 
        marker_indices, marker_names, outdir,
        logger=None
    ):
    """
    Run one full pass over the training set
    """
    model.train() 
    optimizer.zero_grad(set_to_none=True) 

    if device.type == "cuda": 
        torch.cuda.reset_peak_memory_stats(device)

    # Accumulate info for logging
    running_loss = 0.0  # composite loss
    running_recon_loss = 0.0
    running_sparsity_loss = 0.0
    running_marker_loss = 0.0
    running_acc = 0.0  # masked-token accuracy
    running_mask_prob = 0.0  
    num_batches = max(len(dataloader), 1)  

    # Initialize accumulator for markers
    marker_plot_accumulator = MarkerPlotAccumulator(
        marker_idxs=marker_indices,
        marker_names=marker_names,
        ignore_index=-100,
    )

    start_time = time.time()  

    progress = dataloader  
    if tqdm is not None:  
        progress = tqdm(
            dataloader,
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            dynamic_ncols=True,
            leave=True,
        )

    
    # Batch-local per-cell-type per-bin counters
    ct_bin_correct = torch.zeros(num_cell_types, NUM_CLASSES, dtype=torch.long)
    ct_bin_total = torch.zeros(num_cell_types, NUM_CLASSES, dtype=torch.long)
    ct_bin_pred_total = torch.zeros(num_cell_types, NUM_CLASSES, dtype=torch.long)

    running_bin_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    running_bin_total = torch.zeros(NUM_CLASSES, dtype=torch.long)
    running_pred_bin_tot = torch.zeros(NUM_CLASSES, dtype=torch.long)

    # Kwargs for plotting
    plot_kwargs = {}
    plot_kwargs["outdir"] = f"{outdir}/images"
    plot_kwargs["celltype_names"] = idx_to_cell_type

    for step, batch in enumerate(progress, start=1):          

        # Batch-local per-cell-type counters
        ct_correct = torch.zeros(num_cell_types, dtype=torch.long)
        ct_total = torch.zeros(num_cell_types, dtype=torch.long)
        ct_nonzero_correct = torch.zeros(num_cell_types, dtype=torch.long)
        ct_nonzero_total = torch.zeros(num_cell_types, dtype=torch.long)

        input_ids = batch["input_ids"].to(device, non_blocking=True)  
        labels = batch["labels"].to(device, non_blocking=True)  
        celltype_ids = batch["celltype_ids"].to(device, non_blocking=True)  
        step_ids = batch["step_ids"].to(device, non_blocking=True)  

        ## Forward pass
        # input_ids: tokens, already masked
        # celltype_ids: needed for celltype encoding
        # step_ids: where I am in the generating process
        logits = model(input_ids, celltype_ids, step_ids)   # [B, L, K]

        # Exclude the CLS token from the reconstruction loss
        gene_logits = logits[:, :-1, :]  
        gene_labels = labels[:, :-1]  

        loss, recon_loss, sparsity_loss, marker_loss = composite_loss(
            logits=gene_logits.transpose(1,2),
            labels=gene_labels,
            celltype_ids=celltype_ids,
            marker_weights=marker_weights,
            loss_fn=loss_fn,
            lambda_sparse=lambda_sparse,
            lambda_marker=lambda_marker,
            zero_bin_id=0,
        )

        (loss / grad_accum_steps).backward()                      

        did_optimizer_step = False  
        if step % grad_accum_steps == 0 or step == len(dataloader): 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            optimizer.step() 
            optimizer.zero_grad(set_to_none=True)  
            did_optimizer_step = True  # optimizer-update step

        with torch.no_grad():

            marker_plot_accumulator.update(
                logits=gene_logits,
                labels=gene_labels,
                celltype_ids=celltype_ids
            )

            if step % log_every == 0:

                # Update plot kwargs
                plot_kwargs["step"] = step
                plot_kwargs["epoch"] = epoch

                # Plot markers (kwargs)
                plot_accumulated_markers(accumulator=marker_plot_accumulator, plot_kwargs=plot_kwargs)
                marker_plot_accumulator.reset()
            
            pred_bins = gene_logits.argmax(dim=-1)      # [B, G]
            valid = gene_labels != IGNORE_INDEX         # [B, G]
            masked_tokens = int(valid.sum().item())

            # Per-cell-type masked-token accuracy
            pred_cpu = pred_bins.detach().cpu()
            labels_cpu = gene_labels.detach().cpu()
            valid_cpu = valid.detach().cpu()
            celltype_cpu = celltype_ids.detach().cpu()

            # Loop in every cell type
            for ct in range(num_cell_types):

                # Create mask for current cell type
                sample_mask = celltype_cpu == ct

                # If no ct cell type is present, skip
                if not sample_mask.any():
                    continue

                # [B, G] ct x masked tokens mask. [B, G] & [B, 1]
                ct_valid = valid_cpu & sample_mask.unsqueeze(1)
                this_total = int(ct_valid.sum().item())

                # Check that we have some valid tokens
                if this_total == 0:
                    continue

                this_correct = int((pred_cpu[ct_valid] == labels_cpu[ct_valid]).sum().item())
                # print(f"Total accuracy cell_type {ct}: {(this_correct / this_total):.4f}")

                # Update running tensors
                ct_total[ct] += this_total
                ct_correct[ct] += this_correct

                # Filter labels and predictions by cell type
                ct_true_bins = labels_cpu[ct_valid]
                ct_pred_bins = pred_cpu[ct_valid]

                for bin_id in range(NUM_CLASSES):
                    this_bin_mask = ct_true_bins == bin_id
                    this_bin_total = int(this_bin_mask.sum().item())

                    if this_bin_total > 0:
                        # How many correct predictions on current bin_id
                        this_bin_correct = int(
                            (ct_pred_bins[this_bin_mask] == bin_id).sum().item()
                        )
                    else:
                        this_bin_correct = 0
                    
                    # How many predictions on current bin_id
                    this_bin_pred_total = int((ct_pred_bins == bin_id).sum().item())

                    # Update running sums
                    ct_bin_total[ct, bin_id] += this_bin_total
                    ct_bin_correct[ct, bin_id] += this_bin_correct
                    ct_bin_pred_total[ct, bin_id] += this_bin_pred_total

            if valid.any():
                correct = pred_bins[valid] == gene_labels[valid]
                acc = correct.float().mean().item()

                true_bins = gene_labels[valid]          # [num_masked]
                pred_valid = pred_bins[valid]           # [num_masked]

                # Count how many predictions fall into each bin
                pred_counts = torch.bincount(
                    pred_valid.detach().cpu(),
                    minlength=NUM_CLASSES,
                )

                running_pred_bin_tot += pred_counts

                for bin_id in range(NUM_CLASSES):
                    bin_mask = true_bins == bin_id
                    bin_total = int(bin_mask.sum().item())

                    if bin_total > 0:
                        bin_correct = int((pred_valid[bin_mask] == bin_id).sum().item())
                    else:
                        bin_correct = 0

                    running_bin_total[bin_id] += bin_total
                    running_bin_correct[bin_id] += bin_correct


        # store logging info
        running_loss += loss.item()  # composite loss
        running_recon_loss += recon_loss.item()
        running_sparsity_loss += sparsity_loss.item()
        running_marker_loss += marker_loss.item()
        running_acc += acc  
        running_mask_prob += float(batch["mask_prob"])  

        avg_loss = running_loss / step  # avg composite loss
        avg_recon_loss = running_recon_loss / step
        avg_sparsity_loss = running_sparsity_loss / step
        avg_marker_loss = running_marker_loss / step
        avg_acc = running_acc / step  
        avg_mask_prob = running_mask_prob / step  
        elapsed = time.time() - start_time  
        batches_per_second = step / max(elapsed, 1e-8) 
        eta_seconds = (num_batches - step) / max(batches_per_second, 1e-8)  

        bin_acc = running_bin_correct.float() / running_bin_total.clamp_min(1).float()
        bin_acc_str = " ".join([f"bin{bin_id}={bin_acc[bin_id].item():.3f}" for bin_id in range(NUM_CLASSES)])
        bin_total_str = " ".join([f"n{bin_id}={int(running_bin_total[bin_id].item())}" for bin_id in range(NUM_CLASSES)])
        pred_bin_str = " ".join([f"pred{bin_id}={int(running_pred_bin_tot[bin_id].item())}" for bin_id in range(NUM_CLASSES)])
        # update tqdm progress bar with stored info
        if tqdm is not None:  
            progress.set_postfix(
                loss=f"{avg_loss:.4f}",                     # composite loss
                recon=f"{avg_recon_loss:.4f}",              # reconstruction loss
                sparse=f"{avg_sparsity_loss:.4f}",          # sparsity loss
                marker=f"{avg_marker_loss:.4f}",            # marker specific loss
                acc=f"{avg_acc:.4f}",
                mask=f"{avg_mask_prob:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                gpu=gpu_memory_string(device) if device.type == "cuda" else "cpu",
            )

        # per-cell-type log string
        ct_acc_parts = []

        for ct in range(num_cell_types):
            total_ct = int(ct_total[ct].item())

            if total_ct == 0:
                continue

            acc_ct = ct_correct[ct].item() / total_ct
            nonzero_total_ct = int(ct_nonzero_total[ct].item())

            if nonzero_total_ct > 0:
                nonzero_acc_ct = ct_nonzero_correct[ct].item() / nonzero_total_ct
            else:
                nonzero_acc_ct = float("nan")

            ct_name = idx_to_cell_type.get(ct, f"celltype_{ct}")
            ct_acc_parts.append(f"{ct_name}:acc={acc_ct:.3f},nonzero_acc={nonzero_acc_ct:.3f},n={total_ct},nonzero_n={nonzero_total_ct}")

        ct_acc_str = " | ".join(ct_acc_parts)

        # print periodic logs
        if step == 1 or step % log_every == 0 or step == num_batches: 

            # Print cell type specific information
            cell_type_acc = ct_bin_correct / ct_bin_total
            cell_type_prec = ct_bin_correct / ct_bin_pred_total

            print(f"Cell type Accuracy: {cell_type_acc}")
            print(f"Cell type precision: {cell_type_prec}")

            # Reset to zero
            ct_bin_correct = ct_bin_correct.zero_()
            ct_bin_total = ct_bin_total.zero_()
            ct_bin_pred_total = ct_bin_pred_total.zero_()

            log(
                f"[epoch {epoch + 1}/{total_epochs} step {step}/{num_batches}] "
                f"composite_loss={avg_loss:.4f} "
                f"reconstruction_loss={avg_recon_loss:.4f} "
                f"sparsity_loss={avg_sparsity_loss:.4f} "
                f"marker_loss={avg_marker_loss:.4f} "
                f"acc={avg_acc:.4f} "
                f"mask_prob={avg_mask_prob:.3f} "
                f"masked_tokens={masked_tokens} "
                f"{bin_acc_str} "
                f"{bin_total_str} "
                f"{pred_bin_str} "
                f"celltype_acc=[{ct_acc_str}] "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"opt_step={did_optimizer_step} "
                f"speed={batches_per_second:.2f} batch/s "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta_seconds)} "
                f"{gpu_memory_string(device)}",
                logger=logger,
            )

            # Reset running tensors for acc
            running_bin_correct = running_bin_correct.zero_()
            running_bin_total = running_bin_total.zero_()
            running_pred_bin_tot = running_pred_bin_tot.zero_()

    celltype_acc = {}
    celltype_nonzero_acc = {}

    for ct in range(num_cell_types):
        total_ct = int(ct_total[ct].item())

        if total_ct > 0:
            acc_ct = ct_correct[ct].item() / total_ct
        else:
            acc_ct = float("nan")

        nonzero_total_ct = int(ct_nonzero_total[ct].item())

        if nonzero_total_ct > 0:
            nonzero_acc_ct = ct_nonzero_correct[ct].item() / nonzero_total_ct
        else:
            nonzero_acc_ct = float("nan")

        ct_name = idx_to_cell_type.get(ct, f"celltype_{ct}")
        celltype_acc[ct_name] = float(acc_ct)
        celltype_nonzero_acc[ct_name] = float(nonzero_acc_ct)

    return {  
    "loss": running_loss / num_batches,  
    "recon_loss": running_recon_loss / num_batches,
    "sparsity_loss": running_sparsity_loss / num_batches,
    "marker_loss": running_marker_loss / num_batches,
    "acc": running_acc / num_batches,  
    "mask_prob": running_mask_prob / num_batches,  
    "elapsed": time.time() - start_time,
    "last_batch_celltype_acc": celltype_acc,
    "last_batch_celltype_nonzero_acc": celltype_nonzero_acc,
    }

def parse_weights(weights: str, expected_num_classes: int = NUM_CLASSES, atol: float = 1e-6):
    """
    Parse comma-separated class weights/probabilities.

    It expects:
    - exactly expected_num_classes values
    - all values must be non-negative
    - values must sum to 1
    - the last two values must be exactly 0

    Example:
        --penalties "0.04,0.06,0.20,0.30,0.40,0.0,0.0"
    """

    try:
        values = [float(x.strip()) for x in weights.split(",")]
    except ValueError as e:
        raise ValueError(
            f"Could not parse --penalties='{weights}'. "
            "Please provide comma-separated numbers, e.g. "
            "'0.04,0.06,0.20,0.30,0.40,0.0,0.0'."
        ) from e

    if len(values) != expected_num_classes:
        raise ValueError(
            f"--penalties must contain exactly {expected_num_classes} values, "
            f"but {len(values)} were given: {values}"
        )

    if any(v < 0 for v in values):
        raise ValueError(
            f"--penalties must contain only non-negative values, "
            f"but got: {values}"
        )

    if values[-1] != 0.0:
        raise ValueError(
            f"The last two probabilities must be 0. "
            f"Got: {values[-2:]}"
        )

    return torch.tensor(values, dtype=torch.float32)


def build_arg_parser():  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed h5ad dataset.")  
    parser.add_argument("--model_path", type=str, default=None, help="Optional pretrained scBERT checkpoint.")  
    parser.add_argument("--cell_types", nargs="+", default=None, help="Optional list of cell types to keep. If omitted, all cell types are used.")  
    parser.add_argument("--max_cells_per_celltype", type=int, default=None, help="Maximum number of cells to sample per selected cell types. If omitted, use all cells.")
    parser.add_argument("--epochs", type=int, default=50)  
    parser.add_argument("--batch_size", type=int, default=8)  
    parser.add_argument("--learning_rate", type=float, default=1e-4)  
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)  
    parser.add_argument("--warmup_epochs", type=int, default=5) 
    parser.add_argument("--grad_accum_steps", type=int, default=1) 
    parser.add_argument("--log_every", type=int, default=100, help="Print a detailed training line every N batches.") 
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes.") 
    parser.add_argument("--seed", type=int, default=2021, help="Set the random seed for reproducibility.") 
    parser.add_argument("--dim", type=int, default=200, help="Set the embedding dimension of the backbone. By default it uses the one of scBERT.") 
    parser.add_argument("--depth", type=int, default=6, help="Define the number of transformer blocks.") 
    parser.add_argument("--heads", type=int, default=10, help="Set the number of attention heads.")  
    parser.add_argument("--local_attn_heads", type=int, default=0)  
    parser.add_argument("--pos_embed", type=str_to_bool, default=True, help="Enable gene2vec positional embeddings.")  
    parser.add_argument("--p_start", type=float, default=0.1, help="Define the initial masking probability range lower bound.")  
    parser.add_argument("--p_end", type=float, default=1.0, help="Define the final masking probability range upper bound.") 
    parser.add_argument("--num_mask_steps", type=int, default=3, help="Define how many masking-probability bins are used across training.")
    parser.add_argument("--markers", type=str, required=True, help="Path to the tsv marker file.")
    parser.add_argument("--lambda_marker", type=float, default=0.5, help="Weight for the markers loss.")
    parser.add_argument( "--final_random_mask_pct", type=float, default=0.0,
    help=(
        "Fraction of training spent sampling mask probability from the full "
        "[p_start, p_end] range after curriculum. "
        "Example: 0.10 means last 10 percent of epochs use full-range random masking."))
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts", help="Path to the directory where the checkpoints and logs will be written.")  
    parser.add_argument("--model_name", type=str, default="diffscbert", help="Define the prefix used for checkpoint filenames.")  
    parser.add_argument("--penalties", type=str, default=None, help="Optional class weights for reconstruction loss. Not used when running unweighted FocalLoss.")
    parser.add_argument("--max_zero_mask_fraction", type=float, default=0.60, help="Maximum fraction of masked gene tokens allowed to come from zero-expression bin.")
    parser.add_argument("--lambda_sparse", type=float, default=0.1, help="Weight of the sparsity-preservation loss.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser  


def main():  
    args = build_arg_parser().parse_args()  
    seed_all(args.seed) 

    if args.device.startswith("cuda") and not torch.cuda.is_available(): 
        device = torch.device("cpu") 
    else: 
        device = torch.device(args.device) 

    logger = set_logger(Path(args.ckpt_dir) / "logs", name=args.model_name)

    log("=" * 80, logger=logger)  
    log("DIFFscBERT training start", logger=logger)  
    training_start_time = time.time()  
    history = []  
    log(f"device={device}", logger=logger)
    if device.type == "cuda":
        log(f"cuda_visible_device_count={torch.cuda.device_count()}", logger=logger)
        log(f"cuda_device_name={torch.cuda.get_device_name(device)}", logger=logger)
    log(f"args={vars(args)}", logger=logger)
    log("Loading dataset...", logger=logger)
    dataset_load_start = time.time()

    # Load the dataset and keep only the selected cell types if requested and max cells per cell type
    dataset = DiffscBERTDataset(args.data_path, cell_types=args.cell_types, max_cells_per_celltype=args.max_cells_per_celltype,seed=args.seed)  

    log(f"Dataset loaded in {format_seconds(time.time() - dataset_load_start)}", logger=logger)
    log(f"Selected cell types={dataset.cell_types}", logger=logger)
    log(f"Number of selected cell types={dataset.num_cell_types}", logger=logger)
    log(f"Max cells per cell type={args.max_cells_per_celltype}", logger=logger)
    log(f"Number of selected cells={len(dataset):,}", logger=logger)
    log(f"Cell-type mapping={dataset.mapping}", logger=logger)
    log(f"Raw count matrix shape={dataset.count_matrix.shape}", logger=logger)
    
    mapping_path = Path(args.ckpt_dir) / f"{args.model_name}_cell_type_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mapping_path, "w") as f:
        json.dump(
            {
                "cell_types": dataset.cell_types,
                "cell_type_to_idx": dataset.mapping,
                "idx_to_cell_type": dataset.inverse_mapping,
                "num_cell_types": dataset.num_cell_types,
            },
            f,
            indent=2,
        )

    log(f"Saved cell-type mapping to {mapping_path}", logger=logger)

    # Build the collate function that appends CLS and applies masking
    collator = DiffscBERTCollator(  
        cls_token_id=dataset.cls_token_id,  
        mask_token_id=dataset.mask_token_id,  
        total_epochs=args.epochs,  
        p_start=args.p_start,  
        p_end=args.p_end,  
        max_zero_mask_fraction=args.max_zero_mask_fraction,
        num_probability_bins=args.num_mask_steps,
        final_random_mask_pct=args.final_random_mask_pct,
    )

    # Create the PyTorch dataloader that yields masked training batches
    train_loader = DataLoader(  
        dataset,  
        batch_size=args.batch_size,  
        shuffle=True,  
        collate_fn=collator,  
        num_workers=args.num_workers,  
        pin_memory=device.type == "cuda"
    )

    seq_len = dataset.count_matrix.shape[1] + 1                                 # number of genes + CLS token
    num_tokens = dataset.mask_token_id + 1                                      # vocabulary size == highest token id + 1 
    effective_batch_size = args.batch_size * args.grad_accum_steps              # compute the effective batch size after accumulation
    log(f"Sequence length={seq_len:,} tokens including CLS", logger=logger)
    log(f"Vocabulary size={num_tokens:,}", logger=logger)
    log(f"Batch size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, effective_batch_size={effective_batch_size}", logger=logger)
    log(f"Batches per epoch={len(train_loader):,}", logger=logger)
    log(f"Optimizer updates per epoch approximately={math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)):,}", logger=logger)
    log("Building/loading pretrained encoder...", logger=logger)
    encoder_start = time.time()

    # Load the scBERT encoder
    pretrained_encoder = build_pretrained_encoder(num_tokens, seq_len, args) 
    log(f"Pretrained encoder ready in {format_seconds(time.time() - encoder_start)}", logger=logger)

    # Wrap the pretrained encoder with DIFFscBERT conditioning layers
    model = DiffScBERT(  
        pretrained_encoder=pretrained_encoder,          # scBERT backbone
        # num_celltypes=len(dataset.mapping),           # number of distinct cell types
        num_celltypes=dataset.num_cell_types,
        num_steps=len(collator.mask_ranges),            # number of masking bins
    ).to(device) 
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    log(f"Model moved to {device}", logger=logger)
    log(f"Trainable parameters={trainable_params:,} / total parameters={total_params:,}", logger=logger)
    log(gpu_memory_string(device), logger=logger)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate) 

    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))  # ensures warmup never exceeds the total number of epochs
    scheduler = CosineAnnealingWarmupRestarts(  
        optimizer,  
        first_cycle_steps=max(args.epochs, 1),  
        max_lr=args.learning_rate,  
        min_lr=args.min_learning_rate,  
        warmup_steps=warmup_epochs,  
        gamma=0.9,  
    )

    # Compute reconstruction loss only on masked gene positions
    # weights = parse_weights(args.penalties).to(device)
    if num_tokens != NUM_CLASSES:
        raise ValueError(
            f"NUM_CLASSES={NUM_CLASSES}, but dataset/model vocabulary has num_tokens={num_tokens}. "
            "These must match because CrossEntropyLoss(weight=...) needs one weight per output class."
        )

    alpha = torch.tensor([0.20, 1.2, 1.50, 4.4, 10, 10, 0.0], device=device)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0, ignore_index=IGNORE_INDEX, reduction="mean").to(device)

    # Load markers for marker specific loss
    marker_df = pd.read_csv(args.markers, sep="\t")
    weights_df = construct_marker_weights(
        marker_df=marker_df,
        max_extra_weight=5.0 # score given to the best marker + 1
    )
    marker_weights = build_marker_weight_vectors(
        weights_df,
        cell_type_to_id=dataset.mapping,
        num_genes=seq_len-1,    # no CLS
        device=device,
    )

    # Extract marker idxs and names
    marker_indices, marker_names = load_marker_data(
        args.markers,
        cell_type_to_id=dataset.mapping,
        num_genes=seq_len - 1 # No CLS
    )


    for epoch in range(args.epochs): 
        # tell the collator which epoch we are in in order to choose the masking range 
        collator.set_epoch(epoch)  
        step_id, (range_start, range_end), schedule_mode = collator.get_current_range()  
        log("-" * 80, logger=logger)
        log(  
            f"Starting epoch {epoch + 1}/{args.epochs} | "
            f"Schedule_mode={schedule_mode} | "
            f"step_bin={step_id if step_id is not None else 'derived_from_mask_prob'} | "
            f"mask_range=({range_start:.3f}, {range_end:.3f})",
            logger=logger,  
        )

        # Train for one epoch
        stats = train_one_epoch( 
            model=model,  
            dataloader=train_loader,  
            optimizer=optimizer,  
            loss_fn=loss_fn, 
            device=device,
            grad_accum_steps=args.grad_accum_steps,  
            epoch=epoch,  
            total_epochs=args.epochs,  
            log_every=args.log_every, 
            lambda_sparse=args.lambda_sparse,
            lambda_marker=args.lambda_marker,
            num_cell_types=dataset.num_cell_types,
            idx_to_cell_type=dataset.inverse_mapping,
            marker_weights=marker_weights,
            marker_indices=marker_indices,
            marker_names=marker_names,
            outdir=args.ckpt_dir,
            logger=logger,  
        )

        scheduler.step()  
        # Epoch summary to store in the checkpoint
        epoch_record = {  
            "epoch": epoch + 1,
            "schedule_mode": schedule_mode,
            "step_bin": step_id if step_id is not None else "derived_from_mask_prob",
            "mask_range": [float(range_start), float(range_end)],  
            "sampled_mask_prob": float(stats["mask_prob"]), 
            "loss": float(stats["loss"]),
            "recon_loss": float(stats["recon_loss"]),
            "sparsity_loss": float(stats["sparsity_loss"]),  
            "marker_loss": float(stats["marker_loss"]),
            "acc": float(stats["acc"]),
            "last_batch_celltype_acc": stats["last_batch_celltype_acc"],
            "last_batch_celltype_nonzero_acc": stats["last_batch_celltype_nonzero_acc"],
            "elapsed": float(stats["elapsed"]),  
        }
        history.append(epoch_record) 
        ckpt_path = save_checkpoint(model, optimizer, scheduler, epoch, args, stats=epoch_record, history=history) 
        log(f"Saved checkpoint to {ckpt_path}", logger=logger) 

        ct_summary = " | ".join(
            [
                f"{ct}:acc={stats['last_batch_celltype_acc'][ct]:.4f},nonzero_acc={stats['last_batch_celltype_nonzero_acc'][ct]:.4f}"
                for ct in stats["last_batch_celltype_acc"]
            ]
        )

        log(  
            f"Epoch {epoch + 1}/{args.epochs} | "  
            f"schedule_mode={schedule_mode} | "
            f"step_bin={step_id if step_id is not None else 'derived_from_mask_prob'} | " 
            f"mask_range=({range_start:.3f}, {range_end:.3f}) | "  
            f"sampled_mask_prob={stats['mask_prob']:.3f} | " 
            f"loss={stats['loss']:.4f} | "
            f"recon={stats['recon_loss']:.4f} | "
            f"sparse={stats['sparsity_loss']:.4f} | "
            f"marker={stats['marker_loss']:.4f} | "
            f"acc={stats['acc']:.4f} | "
            f"last_batch_celltype_acc=[{ct_summary}] | "
            f"elapsed={format_seconds(stats['elapsed'])} | "
            f"{gpu_memory_string(device)}",
            logger=logger,
        )

    total_training_time = time.time() - training_start_time  
    log("=" * 80, logger=logger) 
    log(f"Training completed in {format_seconds(total_training_time)}", logger=logger) 
    log("=" * 80, logger=logger)  
    return 0 


if __name__ == "__main__": 
    main()  
