"""
Dataset wrapper for scBERT.
"""

import scanpy as sc
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import List, Optional

NUM_CLASS = 7

class DiffscBERTDataset(Dataset):

    def __init__(
        self,
        path: str,                                      # path to the preprocessed .h5ad file
        cell_types: Optional[List[str]] = None,         # if None, all cell types in the dataset are used
        max_cells_per_celltype: Optional[int] = None,   # max number of cells to keep per selected cell type. If None, all cells are used
        seed: int = 2021,
        debug: bool = False
    ):

        self.debug = debug

        # Read H5AD dataset
        data = sc.read_h5ad(path)

        if "cell_type" not in data.obs.columns:
            raise ValueError(
                "'cell_type' column was not found in data.obs. "
                f"Available columns are: {list(data.obs.columns)}"
            )

        # Decide which cell types to use
        available_cell_types = sorted(
            data.obs["cell_type"].astype(str).unique().tolist()
        )

        if cell_types is None:
            cell_types = available_cell_types
        else:
            non_existing_ct = [
                c for c in cell_types
                if c not in available_cell_types
            ]

            if len(non_existing_ct) > 0:
                raise ValueError(
                    f"Specified cell type(s) {non_existing_ct} not found in data. "
                    f"Available cell types are: {available_cell_types}"
                )

        self.cell_types = cell_types
        self.max_cells_per_celltype = max_cells_per_celltype

        # Filter by cell type and optionally subsample each cell type
        rng = np.random.default_rng(seed)

        selected_indices = []

        for ct in cell_types:
            ct_indices = np.where(
                data.obs["cell_type"].astype(str).values == ct
            )[0]

            if (
                max_cells_per_celltype is not None
                and len(ct_indices) > max_cells_per_celltype
            ):
                ct_indices = rng.choice(
                    ct_indices,
                    size=max_cells_per_celltype,
                    replace=False,
                )

            selected_indices.extend(ct_indices.tolist())

        selected_indices = np.array(selected_indices)

        # Shuffle selected cells so the dataset is not ordered by cell type
        rng.shuffle(selected_indices)

        filtered_data = data[selected_indices].copy()

        
        print("=" * 80)
        print("DiffscBERTDataset summary")
        print(f"Original dataset shape: {data.shape}")
        print(f"Filtered dataset shape: {filtered_data.shape}")
        print(f"Number of selected cell types: {len(cell_types)}")
        print(f"Selected cell types: {cell_types}")
        print(f"max_cells_per_celltype: {max_cells_per_celltype}")
        print("Selected cells per cell type:")

        for ct in cell_types:
            n = int(
                (filtered_data.obs["cell_type"].astype(str).values == ct).sum()
            )
            print(f"    {ct}: {n}")

        print("=" * 80)


        # Mapping from cell type name to integer label
        # Example:
        #   "fibroblast" -> 0
        #   "macrophage" -> 1
        #   "malignant cell" -> 2
        self.mapping = {
            ct: i for i, ct in enumerate(cell_types)
        }

        # Optional reverse mapping
        self.inverse_mapping = {
            i: ct for ct, i in self.mapping.items()
        }

        self.num_cell_types = len(self.cell_types)

        # Store count matrix -> shape: number of selected cells x number of genes
        self.count_matrix = filtered_data.X

        # Binning and tokenization        
        self.bin_num = 5
        # Token 0 is used for zero expression, as in scBERT
        self.cls_token_id = 0

        # MASK token is placed after the expression-bin tokens.
        # With bin_num = 5:        
        #   0 = zero-expression / CLS token
        #   1 = expression bin 1
        #   2 = expression bin 2
        #   3 = expression bin 3
        #   4 = expression bin 4
        #   5 = expression bin 5
        #   6 = MASK token
        self.mask_token_id = self.bin_num + 1

        
        # Store labels -> the integer cell-type conditions used by the model
        self.labels = filtered_data.obs["cell_type"].astype(str).apply(
            lambda x: self.mapping[x]
        )

        # Store filtered AnnData 
        self.filtered_data = filtered_data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        selected_rows = self.count_matrix[index, :].toarray()

        if self.debug:
            import pandas as pd
            print(f"Expression values")
            print(pd.DataFrame(selected_rows.T).value_counts())

        # Return tokenized version
        selected_rows[selected_rows > (NUM_CLASS - 2)] = NUM_CLASS - 2
        selected_rows = torch.from_numpy(selected_rows).long()

        num_rows = selected_rows.shape[0]
        selected_rows = torch.cat((selected_rows, torch.zeros(num_rows, 1)), dim=-1)

        if self.debug:
            print("After Tokenization")
            print(pd.DataFrame(selected_rows.T).value_counts())

        # Take labels
        selected_cells = self.labels.iloc[index]

        if isinstance(selected_cells, (int, np.integer)):
            selected_cells = [selected_cells]
        else:
            selected_cells = selected_cells.tolist()

        return {
            "data": selected_rows,
            "label": torch.tensor(selected_cells, dtype=torch.long),
        }

    def __len__(self):
        # Number of selected cells
        return self.count_matrix.shape[0]

if __name__ == "__main__":

    ds_path = "../../DATASET/preprocessed_data.h5ad"

    # multiple cell types, max 2000 cells per type
    ds_balanced = DiffscBERTDataset(
        ds_path,
        cell_types=[
            "CD4-positive helper T cell",
            "fibroblast",
            "macrophage",
            "cycling stromal cell",
            "mast cell",
        ],
        max_cells_per_celltype=2000,
        seed=2021,
        debug=True
    )

    print(f"The balanced dataset has {len(ds_balanced)} samples")
    print(ds_balanced[1])
    # print(ds_balanced[0])
    # print(ds_balanced[0:16])
    # print(ds_balanced[torch.tensor([0, 1, 2, 10])])