"""
Generate visualizations utilities for plotting markers' predictions
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from pathlib import Path
from ablation import expected_expression

CELLS_PER_ROW = 4

from collections import defaultdict

import torch


class MarkerPlotAccumulator:
    """
    Accumulate expected predicted expression for marker genes across batches.

    For each cell type and marker gene, this stores:
        - the sum of expected predictions at masked positions
        - the number of masked observations

    It does not retain logits, so memory usage remains small.
    """

    def __init__(
        self,
        marker_idxs: dict,
        marker_names: dict,
        ignore_index: int = -100,
    ):
        self.marker_idxs = marker_idxs
        self.marker_names = marker_names
        self.ignore_index = ignore_index

        self.marker_sums = {}
        self.marker_counts = {}
        self.cell_counts = defaultdict(int)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        celltype_ids: torch.Tensor,
    ):
        """
        Add one batch to the accumulator.

        Parameters:
        - logits: Gene logits with shape [B, G, K]
        - labels: Gene labels with shape [B, G]. Unmasked positions must contain `ignore_index`
        - celltype_ids: Cell-type IDs with shape [B]
        """

        # Compute expected values before moving to CPU if desired
        # Output shape: [B, G]
        expected_values = expected_expression(logits.detach())
        expected_values = expected_values.cpu()
        labels = labels.detach().cpu()
        celltype_ids = celltype_ids.detach().cpu().reshape(-1)

        if expected_values.shape != labels.shape:
            raise ValueError(
                "`expected_expression(logits)` and `labels` must have "
                f"the same shape. Got {expected_values.shape} and "
                f"{labels.shape}."
            )

        unique_celltypes = torch.unique(celltype_ids).tolist()

        for celltype_id in unique_celltypes:
            if (celltype_id not in self.marker_idxs or celltype_id not in self.marker_names):
                continue

            ct_rows = celltype_ids == celltype_id
            ct_expected = expected_values[ct_rows]  # [B_ct, G]
            ct_labels = labels[ct_rows]             # [B_ct, G]

            self.cell_counts[celltype_id] += int(ct_rows.sum().item())

            marker_indices = torch.as_tensor(self.marker_idxs[celltype_id], dtype=torch.long)
            marker_indices = marker_indices.cpu()

            if marker_indices.numel() == 0:
                continue

            marker_values = ct_expected[:, marker_indices]  # [B_ct, M]

            # True only at positions used for reconstruction
            marker_mask = (ct_labels[:, marker_indices] != self.ignore_index)  # [B_ct, M]
            batch_counts = marker_mask.sum(dim=0).to(torch.long)
            batch_sums = (marker_values * marker_mask.to(marker_values.dtype)).sum(dim=0)

            if celltype_id not in self.marker_sums:
                self.marker_sums[celltype_id] = torch.zeros_like(batch_sums, dtype=torch.float64)
                self.marker_counts[celltype_id] = torch.zeros_like(batch_counts, dtype=torch.long)

            self.marker_sums[celltype_id] += batch_sums.to(torch.float64)
            self.marker_counts[celltype_id] += batch_counts

    def reset(self):
        """Remove all accumulated values"""
        self.marker_sums.clear()
        self.marker_counts.clear()
        self.cell_counts.clear()

    def has_data(self):
        """Return True when at least one cell type has accumulated data"""
        return bool(self.marker_sums)

def plot_marker_shift(
    expected_a: np.ndarray,
    expected_b: np.ndarray,
    markers_a: np.ndarray,
    markers_b: np.ndarray,
    plot_kwargs: dict,
    idx: int,
):
    """
    Plot marker prediction shift from ct A to ct B
    (when I change my cell type from A to B, does marker predictions move accordingly?)
    """

    if expected_a.ndim > 1:
        expected_a = expected_a.squeeze(0)

    if expected_b.ndim > 1:
        expected_b = expected_b.squeeze(0)
    # Check to pass only one sample, and that shapes are adjusted
    if expected_a.ndim != 1 or expected_b.ndim != 1:
        print(f"expected_a shape: {expected_a.shape}")
        print(f"expected_b shape: {expected_b.shape}")
        raise ValueError(
            "`expected_a` and `expected_b` must be one-dimensional arrays "
            "containing one expected value per gene."
        )

    if expected_a.shape != expected_b.shape:
        raise ValueError(
            "`expected_a` and `expected_b` must have the same shape. "
            f"Received {expected_a.shape} and {expected_b.shape}."
        )

    # Extract kwargs with default
    step = plot_kwargs.get("step", None)
    outdir = plot_kwargs.get("outdir", ".")
    idx_to_cell_type = plot_kwargs.get("idx_to_cell", None)
    cell_a = plot_kwargs.get("cell_a_idx", None)
    cell_b = plot_kwargs.get("cell_b_idx", None)
    marker_names = plot_kwargs.get("marker_names", None)

    n_genes = expected_a.shape[0]

    # Ensure inputs are 1D
    markers_a = markers_a.squeeze().cpu().numpy()
    markers_b = markers_b.squeeze().cpu().numpy()
    union_markers = np.concatenate((markers_a, markers_b))

    expected_a = expected_a.squeeze().cpu().numpy()
    expected_b = expected_b.squeeze().cpu().numpy()

    # Scale the figure according to the number of markers in each panel
    # width_a = max(6.0, 0.45 * markers_a.size)
    # width_b = max(6.0, 0.45 * markers_b.size)
    panel_width = max(6.0, 0.45 * union_markers.size)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        constrained_layout=True,
        # figsize=(2*panel_width, 6),
        figsize=(12,6)
    )

    bar_width = 0.38

    def draw_panel(
        ax,
        expected: np.ndarray,
        title: str
    ):
        x = np.arange(union_markers.size)
        values = expected[union_markers]

        # Half of the markers are 
        ax.bar(x[:markers_a.size], values[:markers_a.size], width = bar_width, color="red")
        ax.bar(x[markers_a.size:], values[markers_a.size:], width=bar_width, color="blue")

        if marker_names is None:
            gene_labels = [str(index) for index in union_markers]
        else:
            gene_labels = marker_names[int(cell_a)] + marker_names[int(cell_b)]

        ax.set_xticks(x)
        ax.set_xticklabels(gene_labels, rotation=90, ha="center")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title(title)
        ax.set_xlabel("Marker gene index")
        ax.set_ylabel("Mean expected predicted bin")
        ax.grid(axis="y", alpha=0.3)
        # ax.legend()

    title_a = (
        "Cell type A expression" if cell_a is None and idx_to_cell_type is None
        else f"{idx_to_cell_type[cell_a]} expression"
    )
    title_b = (
        "Cell type A expression" if cell_b is None and idx_to_cell_type is None
        else f"{idx_to_cell_type[cell_b]} expression"
    )
    draw_panel(ax=axes[0], expected=expected_a, title= title_a)
    draw_panel(ax=axes[1], expected=expected_b, title=title_b)

    all_values = np.concatenate([expected_a[union_markers], expected_b[union_markers]])
    y_min = min(0.0, float(np.nanmin(all_values)))
    y_max = float(np.nanmax(all_values))
    margin = max(0.1, 0.05 * (y_max - y_min))

    for ax in axes:
        ax.set_ylim(y_min - margin, y_max + margin)

    output_directory = Path(outdir)
    output_directory.mkdir(parents=True, exist_ok=True)

    suffix = f"_step_{step}_{idx}_{cell_a}_{cell_b}" if step is not None else ""
    output_path = output_directory / f"marker_shift{suffix}.png"

    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Marker-shift plot saved to: {output_path}")


def plot_markers(
    logits: torch.Tensor,
    labels: torch.Tensor,
    celltype_ids: torch.Tensor,
    marker_idxs: dict,
    marker_names: dict,
    plot_kwargs: dict,
    ignore_index: int = -100
):

    # Port everything to cpu
    celltype_ids = celltype_ids.detach().cpu()
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()

    # Calculate expected values per gene
    expected_values = expected_expression(logits)
    expected_values = expected_values.detach().cpu()

    # Find celltypes in batch
    unique_celltypes = torch.unique(celltype_ids).tolist()

    # Keep only cell types for which marker information is available
    celltypes_to_plot = [
        ct
        for ct in unique_celltypes
        if ct in marker_idxs and ct in marker_names
    ]

    if not celltypes_to_plot:
        print("No cell types with marker information found in this batch.")
        return

    cells_per_row = plot_kwargs.get("cells_per_row", 4)
    num_plots = len(celltypes_to_plot)
    num_rows = math.ceil(num_plots / cells_per_row)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=cells_per_row,
        figsize=(4.5 * cells_per_row, 4.5 * num_rows),
        squeeze=False,
    )

    axes = axes.ravel()
    celltype_names = plot_kwargs.get("celltype_names", {})


    def draw_barplot(
        ax,
        values: np.ndarray,
        observed_mask: np.ndarray,
        marker_gene_names,
        title: str,
    ):
        x = np.arange(len(values))

        # Bars are plotted only for markers observed as masked at least once
        display_values = np.where(observed_mask, values, 0.0)
        ax.bar(x, display_values)

        # Mark genes that were never masked for this cell type
        for marker_position, was_observed in enumerate(observed_mask):
            if not was_observed:
                ax.text(
                    marker_position,
                    0.02,
                    "not\nmasked",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(marker_gene_names, rotation=90, ha="center")
        ax.set_xlabel("Marker genes")
        ax.set_ylabel("Mean expected predicted bin")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)


    for plot_idx, celltype_id in enumerate(celltypes_to_plot):
        ax = axes[plot_idx]

        # Select only samples belonging to this cell type
        ct_rows = celltype_ids == celltype_id
        ct_expected = expected_values[ct_rows]  # [B_ct, G]
        ct_labels = labels[ct_rows]             # [B_ct, G]

        # True where a gene was masked and therefore predicted
        ct_masked = ct_labels != ignore_index   # [B_ct, G]

        marker_indices = torch.as_tensor(
            marker_idxs[celltype_id],
            dtype=torch.long,
        )

        # Move marker indices to cpu
        marker_indices = marker_indices.cpu()
        marker_gene_names = list(marker_names[celltype_id])

        if marker_indices.numel() != len(marker_gene_names):
            raise ValueError(
                f"Cell type {celltype_id} has "
                f"{marker_indices.numel()} marker indices but "
                f"{len(marker_gene_names)} marker names."
            )

        if marker_indices.numel() == 0:
            ax.set_title(f"Cell type {celltype_id}: no markers")
            ax.axis("off")
            continue

        if marker_indices.min() < 0 or marker_indices.max() >= logits.shape[1]:
            raise IndexError(
                f"Marker indices for cell type {celltype_id} are outside "
                f"the valid gene range [0, {logits.shape[1] - 1}]."
            )

        # Restrict directly to marker genes
        marker_values = ct_expected[:, marker_indices]  # [B_ct, M]
        marker_mask = ct_masked[:, marker_indices]      # [B_ct, M]

        # Number of masked observations available for each marker gene
        marker_counts = marker_mask.sum(dim=0)          # [M]

        # Sum only values corresponding to masked positions
        marker_sums = (
            marker_values
            * marker_mask.to(marker_values.dtype)
        ).sum(dim=0)

        # Avoid division by zero. Values with no observations are handled
        # separately through 'observed_mask'
        mean_marker_values = (
            marker_sums
            / marker_counts.clamp_min(1).to(marker_values.dtype)
        )

        observed_mask = marker_counts > 0
        title_name = celltype_names.get(celltype_id, f"Cell type {celltype_id}")

        draw_barplot(
            ax=ax,
            values=mean_marker_values.numpy(),
            observed_mask=observed_mask.numpy(),
            marker_gene_names=marker_gene_names,
            title=title_name,
        )

    # Hide unused subplot cells
    for unused_idx in range(num_plots, len(axes)):
        axes[unused_idx].axis("off")

    step = plot_kwargs.get("step")
    epoch = plot_kwargs.get("epoch")

    suffix_parts = []

    if step is not None:
        suffix_parts.append(f"step_{step}")

    if epoch is not None:
        suffix_parts.append(f"epoch_{epoch}")

    suffix = ("_" + "_".join(suffix_parts) if suffix_parts else "")
    outdir = Path(plot_kwargs.get("outdir", "."))
    outdir.mkdir(parents=True, exist_ok=True)

    outname = outdir / f"celltype_markers{suffix}.png"

    fig.tight_layout()
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Markers figure saved in {outname}")


def plot_accumulated_markers(
    accumulator: MarkerPlotAccumulator,
    plot_kwargs: dict,
):
    """
    Plot marker predictions accumulated across multiple batches.
    """

    celltypes_to_plot = sorted(accumulator.marker_sums.keys())

    if not celltypes_to_plot:
        print("No accumulated marker predictions available.")
        return

    cells_per_row = plot_kwargs.get("cells_per_row", 4)
    num_plots = len(celltypes_to_plot)
    num_rows = math.ceil(num_plots / cells_per_row)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=cells_per_row,
        figsize=(4.5 * cells_per_row, 4.5 * num_rows),
        squeeze=False,
    )

    axes = axes.ravel()

    celltype_names = plot_kwargs.get("celltype_names", {})

    for plot_idx, celltype_id in enumerate(celltypes_to_plot):
        ax = axes[plot_idx]

        marker_sums = accumulator.marker_sums[celltype_id]
        marker_counts = accumulator.marker_counts[celltype_id]

        observed_mask = marker_counts > 0
        mean_marker_values = (marker_sums / marker_counts.clamp_min(1).to(marker_sums.dtype))

        values = mean_marker_values.numpy()
        observed = observed_mask.numpy()

        marker_gene_names = list(accumulator.marker_names[celltype_id])

        x = np.arange(len(values))

        # Markers never masked during the accumulation window are displayed
        # at zero, but explicitly annotated.
        display_values = np.where(observed, values, 0.0)

        ax.bar(x, display_values)

        for marker_position, was_observed in enumerate(observed):
            if not was_observed:
                ax.text(
                    marker_position,
                    0.02,
                    "not\nmasked",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            marker_gene_names,
            rotation=90,
            ha="center",
        )

        celltype_name = celltype_names.get(
            celltype_id,
            f"Cell type {celltype_id}",
        )

        number_of_cells = accumulator.cell_counts[celltype_id]
        number_of_predictions = int(marker_counts.sum().item())

        ax.set_title(
            f"{celltype_name}\n"
            f"{number_of_cells} cells, "
            f"{number_of_predictions} masked marker observations"
        )

        ax.set_xlabel("Marker genes")
        ax.set_ylabel("Mean expected predicted bin")

        ax.set_ylim([0, 5])             # 5 = max bin
        ax.grid(axis="y", alpha=0.3)

    for unused_idx in range(num_plots, len(axes)):
        axes[unused_idx].axis("off")

    step = plot_kwargs.get("step")
    epoch = plot_kwargs.get("epoch")

    suffix_parts = []

    if step is not None:
        suffix_parts.append(f"step_{step}")

    if epoch is not None:
        suffix_parts.append(f"epoch_{epoch}")

    suffix = (
        "_" + "_".join(suffix_parts)
        if suffix_parts
        else ""
    )

    outdir = Path(plot_kwargs.get("outdir", "."))
    outdir.mkdir(parents=True, exist_ok=True)

    outname = outdir / f"accumulated_celltype_markers{suffix}.png"

    fig.tight_layout()
    fig.savefig(
        outname,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Accumulated marker figure saved in {outname}")

