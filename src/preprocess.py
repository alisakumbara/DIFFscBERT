"""
Preprocess finetuning dataset.
This scripts aligns the breast cancer dataset to the genetic space of the dataset used to train the scBERT performer. 
At the end, the output dataset will have:
    - the same cells as the breast cancer dataset, except cells removed by filtering
    - the same genes as the Panglao reference dataset
    - genes ordered exactly like in the reference dataset
    - zero values for reference genes that are missing from the breast cancer dataset
"""
import scanpy as sc
import numpy as np
import anndata as ad

from scipy import sparse
from pathlib import Path
 
# Load reference dataset and the breast cancer dataset used for finetuning
PROJECT_ROOT = Path.cwd()
OUTPUT_FILE = PROJECT_ROOT.parent.parent / "DATASET" / "preprocessed_data.h5ad"

panglao = sc.read_h5ad(PROJECT_ROOT.parent.parent / "DATASET" / "panglao_human.h5ad") # --ref
data = sc.read_h5ad(PROJECT_ROOT.parent.parent / "DATASET" / "breast_cancer.h5ad") # --data

# Extract raw counts, not log-normalized
data = data.raw.to_adata() 
 
print(f"Input data shape: {data.shape}")
print(f"Panglao shape: {panglao.shape}")
 
# Use gene symbols instead of Ensembl IDs
data.var_names = data.var["feature_name"].astype(str)
data.var_names_make_unique()

ref = panglao.var_names.tolist() 
obj = data.var_names.tolist() 
 
matches = len(set(ref) & set(obj))
print(f"Matched genes: {matches}/{len(ref)}")
 
counts = sparse.lil_matrix((data.n_obs, len(ref)),dtype=np.float32)
 
# Gene matching
gene_to_idx = {g: i for i, g in enumerate(obj)}
 
for i, gene in enumerate(ref):
    if gene in gene_to_idx:
        counts[:, i] = data.X[:, gene_to_idx[gene]]
 
counts = counts.tocsr()

# Create a new AnnData object using the aligned expression matrix
new = ad.AnnData(X=counts)
# Assign reference gene names to the new dataset
new.var_names = ref
# Preserve original cell names from the breast cancer dataset
new.obs_names = data.obs_names
new.obs = data.obs.copy()
new.uns = panglao.uns.copy()
 
print("Shape before filtering:", new.shape)

# Remove low-quality cells expressing less than min_genes
sc.pp.filter_cells(new, min_genes=200)
 
print("Shape after filtering:", new.shape)
 
if new.n_obs == 0:
    raise ValueError(
        "All cells were removed by filter_cells(). "
    )

# Normalize counts per cell
sc.pp.normalize_total(new, target_sum=1e4)
sc.pp.log1p(new, base=2)
 
new.write_h5ad(OUTPUT_FILE)