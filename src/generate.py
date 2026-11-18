"""
Generate gene-expression matrices with DIFFscBERT.
"""

import torch
import json
import argparse

from model import DiffScBERT
from pathlib import Path
from tqdm import tqdm
from performer_pytorch.performer_pytorch import PerformerLM  

NUM_TOKENS = 7
SEQLEN = 16907 # with CLS
DIM = 200
DEPTH = 6
HEADS = 10
MASK_TOKEN_ID = 6
P_START = 0.1
P_END = 1.0
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_weights(model_path: str):

    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']

    return state_dict


def load_ct_mapping(json_path: str):

    with open(json_path, "r") as fopen:
        ct_mapping = json.load(fopen)
    
    return ct_mapping


def build_base_encoder(): 
    # Create the base scBERT Performer encoder architecture
    return PerformerLM( 
        num_tokens=NUM_TOKENS,  
        dim=DIM,  
        depth=DEPTH,  
        max_seq_len=SEQLEN,  
        heads=HEADS, 
        local_attn_heads=0,  
        g2v_position_emb=True,  
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to the trained model .pth")
    parser.add_argument("--ct_mapping", required=True, type=str, help="Path to the JSON file containing the cell-types mapping.")
    parser.add_argument("--cell_type", required=True, type=str, help="Name of the cell type to generate.")
    parser.add_argument("--num_samples", required=True, type=int, help="Number of samples to generate.")
    parser.add_argument("--outdir", default="generation/", help="Path to the output directory.")
    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    # Create the backbone
    encoder = build_base_encoder()

    # Load ct mapping
    ct_mapping = load_ct_mapping(args.ct_mapping)
    
    # Load state dict
    state_dict = load_weights(args.model_path)
    num_steps = state_dict['step_emb.weight'].shape[0]

    # Create model
    model = DiffScBERT(  
        pretrained_encoder=encoder,      
        num_celltypes=ct_mapping['num_cell_types'],
        num_steps=num_steps,        # number of masking bins
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    mask_probs = torch.linspace(
        P_START,
        P_END,
        steps=num_steps,
        device=DEVICE,
    )

    cell_type_to_idx = ct_mapping['cell_type_to_idx'][args.cell_type]
    out_tensor = torch.empty(args.num_samples, SEQLEN - 1, dtype=torch.long, device = "cpu") # remove the cls

    with torch.no_grad():
        for b in tqdm(range(0, args.num_samples, BATCH_SIZE), desc="Generating..."):
            # Define input_ids (masked tokens) as a tensor of all 6 (==MASK_TOKEN_ID)
            input_ids = torch.full((BATCH_SIZE, SEQLEN), MASK_TOKEN_ID, dtype=torch.long, device=DEVICE)
            input_ids[:, -1] = 0 # CLS is always 0

            # Define cell type ids from the json generated during training
            # celltype_ids is a tensor that tells, for each sample/cell to generate, which cell type should it be
            celltype_ids = torch.full((input_ids.shape[0],), cell_type_to_idx, dtype=torch.long, device=DEVICE)

            current_batch_size = min(BATCH_SIZE, args.num_samples - b)

            for step_id in reversed(range(num_steps)):
            # Define step_ids: where I am in the generating process
            # For each sample, in what step ids am I?
            # mask_prob = 1.000 -> step id = 7
                step_ids = torch.full((input_ids.shape[0],), step_id, dtype=torch.long, device=DEVICE)

                # Forward pass
                logits = model(input_ids, celltype_ids, step_ids)   # [B, L, K] = sample, gene, token

                # Unmask those tokens with the lowest uncertainty from the model so we can switch to a lower step id (lower masking range)
                num_gene_positions = SEQLEN - 1
                current_num_masked = int(
                    (input_ids[0] == MASK_TOKEN_ID).sum().item()
                )

                if step_id == 0:
                    target_num_masked = 0
                else:
                    target_mask_prob = float(mask_probs[step_id - 1].item())
                    target_num_masked = round(target_mask_prob * num_gene_positions)
                
                num_tokens_unmask_per_step = (
                    current_num_masked - target_num_masked
                )

                num_tokens_unmask_per_step = max(
                    0, 
                    min(num_tokens_unmask_per_step, current_num_masked)
                )

                # Set the logits relative to the mask token to -inf, because in the training
                # we did not penalize/consider the mask token
                logits[:, :, MASK_TOKEN_ID] = -float('inf')
                probs = logits.softmax(dim=-1) # prob distribution that does not depend on the mask token
                
                # Decide which gene to unmask, unmask those for which the model is more sure
                # if we have a distribution that is far from being uniform, choose that one
                unif_prob = 1/(NUM_TOKENS-1)
                tot_var_dist = 0.5 * (
                    probs[:,:,:MASK_TOKEN_ID] - unif_prob
                ).abs().sum(dim=-1)

                # Define which tokens should not be unmasked (those with token id != 6)
                forbidden_unmask = input_ids != MASK_TOKEN_ID # those who do not need to be unmasked because they have already been unmasked
                
                # Put to -inf the forbidden_unmask tokens to not consider them
                tot_var_dist[forbidden_unmask] = -float('inf')

                # Choose top K, do not consider those which has already been unmasked
                _, idx_to_unmask = tot_var_dist.topk(num_tokens_unmask_per_step,dim=-1)

                # Extract all those positions to unmask
                probs_to_unmask = torch.gather(probs, -2,  idx_to_unmask.unsqueeze(-1).expand(-1,-1,probs.size(-1)))

                # Extract tokens to unmask
                token_to_unmask = torch.multinomial(probs_to_unmask.reshape(-1, probs_to_unmask.size(-1)), num_samples=1)
                token_to_unmask = token_to_unmask.reshape(probs_to_unmask.size(0),probs_to_unmask.size(1))

                # Put the predicted tokens in their unmasked positions
                input_ids = input_ids.scatter(-1, idx_to_unmask, token_to_unmask)
            
            # Store the generated cells into a tensor
            out_tensor[b:(b+current_batch_size), :] = input_ids[:,:-1].cpu() # without cls

    # Save matrix
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    celltype = args.cell_type.replace(" ", "_")
    torch.save(out_tensor, outdir / f"{celltype}_{args.num_samples}_DIFFscBERT.pt")


if __name__ == "__main__":
    main()
    


