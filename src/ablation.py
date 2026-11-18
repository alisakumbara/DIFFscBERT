"""
Test CLS ablation to see influence on generated cells.
"""

import argparse
import torch
import itertools
import pandas as pd
import json

from generate import build_base_encoder, load_ct_mapping, load_weights, SEQLEN
from model import DiffScBERT
from dataset import DiffscBERTDataset
from collections import defaultdict
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_TOKEN_ID = 6
P_START = 0.1
P_END = 1.0


def extract_cell_type_mappings(ct_mapping_f):

    with open(ct_mapping_f, "r") as fopen:
        ct_mapping = json.load(fopen)

    cell_type_to_id = ct_mapping["cell_type_to_idx"]
    idx_to_cell_type = ct_mapping["idx_to_cell_type"]

    return cell_type_to_id, idx_to_cell_type


def load_marker_data(
    marker_file: str,
    cell_type_to_id: dict,
    num_genes: int,
):
    marker_df = pd.read_csv(marker_file, sep="\t")
    marker_indices = {}
    marker_names = {}

    for cell_type_name, cell_type_id in cell_type_to_id.items():
        rows = marker_df[marker_df["cell_type"] == cell_type_name]
        indices = torch.as_tensor(rows["gene_idx"].astype(int).to_numpy(), dtype=torch.long, device=DEVICE)

        if indices.numel() == 0:
            raise ValueError(f"No markers found for {cell_type_name}")

        if indices.min() < 0 or indices.max() >= num_genes:
            raise ValueError(f"Marker index outside gene-logit range for {cell_type_name}")

        marker_indices[cell_type_id] = indices
        marker_names[cell_type_id] = rows["gene"].tolist()

    return marker_indices, marker_names


def expected_expression(logits):
    """
    logits: [B, G, K]

    Returns:
        Expected expression bin [B, G]
    """

    probabilities = logits.softmax(dim=-1)

    bin_values = torch.arange(
        logits.shape[-1],
        dtype=probabilities.dtype,
        device=logits.device,
    )

    return (probabilities * bin_values).sum(dim=-1)


def summarize_pair(
    logits_a,
    logits_b,
    markers_a,
    markers_b,
    plot_kwargs,
    global_i,
):
    from visualize import plot_marker_shift

    probabilities_a = logits_a.softmax(dim=-1)
    probabilities_b = logits_b.softmax(dim=-1)

    expected_a = expected_expression(logits_a)
    expected_b = expected_expression(logits_b)

    predicted_bins_a = logits_a.argmax(dim=-1)
    predicted_bins_b = logits_b.argmax(dim=-1)

    union_markers = torch.unique(torch.cat([markers_a, markers_b]))

    metrics = {}

    # Global sensitivity
    metrics["global_logit_mae"] = (logits_a - logits_b).abs().mean().item()
    metrics["global_probability_mae"] = (probabilities_a - probabilities_b).abs().mean().item()
    metrics["global_expected_bin_mae"] = (expected_a - expected_b).abs().mean().item()

    # Marker-specific sensitivity
    metrics["marker_probability_mae"] = (probabilities_a[:, union_markers] - probabilities_b[:, union_markers]).abs().mean().item()
    metrics["marker_expected_bin_mae"] = (expected_a[:, union_markers] - expected_b[:, union_markers]).abs().mean().item()
    metrics["marker_bin_change_fraction"] = (predicted_bins_a[:, union_markers] != predicted_bins_b[:, union_markers]).float().mean().item()

    # Plot marker shift
    plot_marker_shift(
        expected_a,
        expected_b,
        markers_a,
        markers_b,
        plot_kwargs,
        idx=global_i
    )

    # Directional effects:
    # Positive means cell type A predicts higher values for A markers
    # than cell type B predicts for those same markers
    metrics["a_marker_direction"] = (
        expected_a[:, markers_a]
        - expected_b[:, markers_a]
    ).mean().item()

    # Positive means condition B predicts higher values for B markers
    # than condition A predicts for those same markers.
    metrics["b_marker_direction"] = (
        expected_b[:, markers_b]
        - expected_a[:, markers_b]
    ).mean().item()

    return metrics


def test_cls_marker_ablation(
    model,
    num_steps,
    idx_to_cell_type,
    marker_indices,
    dataset,
    outdir,
    marker_names,
    num_repeats=50,
):
    model.eval()

    mask_probs = torch.linspace(P_START, P_END, steps=num_steps, device=DEVICE)
    cell_pairs = list(itertools.combinations(sorted(idx_to_cell_type.keys()), 2))

    all_results = defaultdict(list)
    plot_kwargs = {} # required for later plotting
    plot_kwargs["outdir"] = outdir

    with torch.inference_mode():
        for step in reversed(range(num_steps)):
            p_masked = float(mask_probs[step].item())

            for repeat in tqdm(range(num_repeats), desc=f"Step {step}"):
                # Real input
                sampled_idx = torch.randint(0, len(dataset), (1,)).item()
                batch = dataset[sampled_idx]
                input_ids, _ = batch["data"], batch["label"]
                input_ids = input_ids.to(torch.long).to(DEVICE)

                is_masked = (torch.rand((1, SEQLEN), device=DEVICE) < p_masked)

                # Assumes CLS is the first sequence position
                is_masked[:, -1] = False
                input_ids[is_masked] = MASK_TOKEN_ID
                input_ids[:, -1] = 0

                step_ids = torch.full((1,), fill_value=step, dtype=torch.long, device=DEVICE)

                for cell_a, cell_b in cell_pairs:
                    cell_a = int(cell_a)
                    cell_b = int(cell_b)

                    ids_a = torch.full((1,), fill_value=cell_a, dtype=torch.long, device=DEVICE)
                    ids_b = torch.full((1,), fill_value=cell_b, dtype=torch.long, device=DEVICE)

                    logits_a = model(input_ids, ids_a, step_ids)
                    logits_b = model(input_ids, ids_b, step_ids)

                    logits_a = logits_a[:, :-1, :]
                    logits_b = logits_b[:, :-1, :]

                    # This assumes model output is [B, G, K] and does not contain a CLS prediction
                    if logits_a.shape[1] != SEQLEN - 1:
                        raise ValueError(
                            f"Expected logits to contain only gene positions. Input length={SEQLEN}, "
                            f"logit length={logits_a.shape[1]}. Check whether CLS is included in logits.")

                    # Define plot kwargs
                    plot_kwargs["step"] = step
                    plot_kwargs["cell_a_idx"] = str(cell_a)
                    plot_kwargs["cell_b_idx"] = str(cell_b)
                    plot_kwargs["idx_to_cell"] = idx_to_cell_type
                    plot_kwargs["marker_names"] = marker_names

                    metrics = summarize_pair(
                        logits_a=logits_a,
                        logits_b=logits_b,
                        markers_a=marker_indices[cell_a],
                        markers_b=marker_indices[cell_b],
                        plot_kwargs=plot_kwargs,
                        global_i=repeat,
                    )

                    key = (step, cell_a, cell_b)

                    for metric_name, value in metrics.items():
                        all_results[(key, metric_name)].append(value)

    print("\nCLS marker ablation results")

    for step in reversed(range(num_steps)):
        print(f"\nStep {step} | mask probability={mask_probs[step].item():.3f}")

        for cell_a, cell_b in cell_pairs:
            cell_a = int(cell_a)
            cell_b = int(cell_b)
            print(f"\n{idx_to_cell_type[str(cell_a)]} vs {idx_to_cell_type[str(cell_b)]}")

            key = (step, cell_a, cell_b)

            metric_order = [
                "global_probability_mae",
                "global_expected_bin_mae",
                "marker_probability_mae",
                "marker_expected_bin_mae",
                "marker_bin_change_fraction",
                "a_marker_direction",
                "b_marker_direction",
            ]

            for metric_name in metric_order:
                dictionary_key = (
                    (step, cell_a, cell_b),
                    metric_name,
                )

                values_list = all_results.get(dictionary_key)

                if not values_list:
                    print(f"  {metric_name}: no values found for key {dictionary_key}")
                    continue

                values = torch.tensor(values_list, dtype=torch.float32)

                print(
                    f"  {metric_name}: "
                    f"{values.mean().item():.6e} "
                    f"+/- {values.std(unbiased=False).item():.6e} "
                    f"(n={values.numel()})"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to the trained model .pth")
    parser.add_argument("--ct_mapping", required=True, type=str, help="Path to the JSON file containing the cell-types mapping.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed dataset.")
    parser.add_argument("--cell_types", nargs="+", required=False, default=None, help="List of cell types for which to find markers.")  
    parser.add_argument("--marker_file", type=str, required=True, help="Where to find marker data.")
    parser.add_argument("--num_repeats", type=int, default=None, help="How many times run the test.")
    parser.add_argument("--outdir", type=str, default=".", help="Where to save images.")
    args = parser.parse_args()

    # Load the model
    encoder = build_base_encoder()

    # Load ct mapping
    ct_to_idx, idx_to_cell_type = extract_cell_type_mappings(args.ct_mapping)
    
    # Load state dict
    state_dict = load_weights(args.model_path)
    num_steps = state_dict['step_emb.weight'].shape[0]
    num_celltypes = state_dict["celltype_emb.weight"].shape[0]

    # Load dataset
    dataset = DiffscBERTDataset(
        args.data_path,
        cell_types=args.cell_types,
        max_cells_per_celltype=100
    )

    # Create model
    model = DiffScBERT(  
        pretrained_encoder=encoder,      
        num_celltypes=num_celltypes,
        num_steps=num_steps,        # number of masking bins
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    num_genes = SEQLEN - 1

    marker_indices, marker_names = load_marker_data(
        marker_file=args.marker_file,
        cell_type_to_id=ct_to_idx,
        num_genes=num_genes,
    )

    print("Loaded markers:")

    for cell_type_id, names in marker_names.items():
        print(f"  {idx_to_cell_type[str(cell_type_id)]}: {len(names)} markers")

    test_cls_marker_ablation(
        model=model,
        num_steps=num_steps,
        idx_to_cell_type=idx_to_cell_type,
        marker_indices=marker_indices,
        dataset=dataset,
        outdir=args.outdir,
        marker_names=marker_names,
        num_repeats=args.num_repeats,
    )
    

if __name__ == "__main__":

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    main()