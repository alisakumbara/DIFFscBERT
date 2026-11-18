"""
Find cell-type specific markers.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import argparse

from scipy import sparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed h5ad dataset.")
    parser.add_argument("--cell_types", nargs="+", required=False, default=None, help="List of cell types for which to find markers.")  
    parser.add_argument("--num_cells_per_type", type=int, required=False, default=1000, help="How many cells to consider per cell type.")
    parser.add_argument("--num_markers", type=int, default=10, help="How many markers per cell type.")
    parser.add_argument("--outdir", type=str, default=".", help="Where to save img.")
    args = parser.parse_args()

    # Load real preprocessed data
    print("Reading data...")
    data = sc.read_h5ad(args.data_path)
    print("Data loaded!")

    n_cells_per_type = args.num_cells_per_type
    cell_types = args.cell_types
    random_seed = 42
    
    # Check that the requested cell types are present
    available_cell_types = set(data.obs["cell_type"].astype(str))
    
    if cell_types is None:
        cell_types = available_cell_types

    missing_cell_types = [
        cell_type
        for cell_type in cell_types
        if cell_type not in available_cell_types
    ]

    if missing_cell_types:
        raise ValueError(
            f"The following cell types are not present in the dataset: {missing_cell_types}"
        )

    # Sample cells per cell type
    rng = np.random.default_rng(random_seed)
    sampled_indices = []

    cell_type_array = data.obs["cell_type"].astype(str).to_numpy()

    for cell_type in cell_types:
        changed = False
        cell_indices = np.flatnonzero(
            cell_type_array == cell_type
        )

        print(f"{cell_type}: {len(cell_indices)} available cells")

        if len(cell_indices) < n_cells_per_type:
            print(
                f"{cell_type!r} contains only {len(cell_indices)} cells, "
                f"but {n_cells_per_type} cells were requested."
                f"Taking {len(cell_indices)} samples"
            )
            tmp = n_cells_per_type
            changed = True
            n_cells_per_type = len(cell_indices)

        selected = rng.choice(cell_indices, size=n_cells_per_type, replace=False)
        if changed:
            n_cells_per_type = tmp

        sampled_indices.extend(selected.tolist())


    # Subset the real cells
    adata_real = data[sampled_indices].copy()

    # Make cell-type labels categorical and preserve the desired order
    adata_real.obs["cell_type"] = pd.Categorical(
        adata_real.obs["cell_type"].astype(str),
        categories=cell_types,
        ordered=True,
    )

    print("\nSampled AnnData:")
    print(adata_real)

    print("\nSampled cells per cell type:")
    print(adata_real.obs["cell_type"].value_counts(sort=False))

    if sparse.issparse(adata_real.X):
        real_matrix = adata_real.X.toarray()
    else:
        real_matrix = np.asarray(adata_real.X)

    real_matrix = np.asarray(real_matrix, dtype=np.float32)

    # Create a fresh AnnData object, as for the generated data
    adata_real_processed = sc.AnnData(X=real_matrix)
    adata_real_processed.obs["cell_type"] = pd.Categorical(
        adata_real.obs["cell_type"].astype(str),
        categories=cell_types, ordered=True)
    adata_real_processed.obs_names = [f"real_cell_{i}" for i in range(adata_real_processed.n_obs)]

    # Preserve the original gene names and gene order
    adata_real_processed.var_names = data.var_names.astype(str).copy()
    adata_real_processed.var_names_make_unique()

    print("\nReal-cell matrix used for UMAP:")
    print(adata_real_processed)

    # Remove genes with no variation across sampled real cells
    gene_variance = np.var(adata_real_processed.X, axis=0)
    variable_gene_mask = gene_variance > 0

    print(
        "\nGenes with non-zero variance: "
        f"{variable_gene_mask.sum()} / "
        f"{adata_real_processed.n_vars}"
    )

    print("Calculating diff expressed genes...")
    sc.tl.rank_genes_groups(adata_real_processed, groupby="cell_type", method="wilcoxon")
    print("Calculated rankings.")
    f = sc.pl.rank_genes_groups_dotplot(
        adata_real_processed, groupby="cell_type", 
        standard_scale="var", n_genes=args.num_markers,
        return_fig=True
    )

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    f.savefig(outdir / "markers_all.png", dpi=300)
    f.savefig(outdir / "markers_all.svg", format="svg", dpi=300)

    # Plot only positive markers
    f = sc.pl.rank_genes_groups_dotplot(
        adata_real_processed, groupby="cell_type", 
        standard_scale="var", n_genes=args.num_markers,
        return_fig=True, min_logfoldchange=1.0,
        values_to_plot="logfoldchanges"
    )
    f.savefig(outdir / "markers_positive.png", dpi=300)
    f.savefig(outdir / "markers_positive.svg", format = "svg", dpi=300)

    # Get gene indexes
    genes = adata_real_processed.var.index.tolist()

    # Save marker list
    marker_data = []

    for cell_type in cell_types:

        cell_type_markers = sc.get.rank_genes_groups_df(
            adata_real_processed,
            group=cell_type,
        ).sort_values("pvals_adj")

        cell_type_markers = cell_type_markers[cell_type_markers["logfoldchanges"] >= 1].head(args.num_markers)

        # Keep only genes present in the AnnData object
        cell_type_markers = cell_type_markers[cell_type_markers["names"].isin(adata_real_processed.var_names)].copy()
        marker_genes = cell_type_markers["names"].tolist()
        cell_type_mask = (adata_real_processed.obs["cell_type"] == cell_type)
        marker_expression = adata_real_processed[cell_type_mask, marker_genes].X

        # Mean over cells -> one mean per marker gene
        mean_expressions = np.asarray(marker_expression.mean(axis=0)).ravel()

        for (_, row), mean_expression in zip(
            cell_type_markers.iterrows(),
            mean_expressions,
        ):
            gene = row["names"]
            marker_data.append(
                {
                    "gene": gene,
                    "scores": float(row["scores"]),
                    "gene_idx": int(
                        adata_real_processed.var_names.get_loc(gene)
                    ),
                    "logfc": float(row["logfoldchanges"]),
                    "adj_p_val": float(row["pvals_adj"]),
                    "cell_type": cell_type,
                    "mean_expression": float(mean_expression),
                }
            )
            
    marker_df = pd.DataFrame(marker_data)
    marker_df.to_csv(outdir / "markers.tsv", sep="\t", index=False, header=True)

if __name__ == "__main__":
    main()








