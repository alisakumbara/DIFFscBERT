"""
DiffscBERT model architecture
"""

from torch import nn

class DiffScBERT(nn.Module):
    """ """

    def __init__(self, pretrained_encoder, num_celltypes, num_steps):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        dim = pretrained_encoder.token_emb.embedding_dim
        self.softmax = nn.Softmax(dim=-1)

        # Embedding of the desired cell type
        self.celltype_emb = nn.Embedding(num_celltypes, dim) 

        # Embedding of the chosen masking level
        self.step_emb = nn.Embedding(num_steps, dim)  # num_mask_steps in trainer.py

    def forward(self, x, celltype_ids, step_ids):
        # Pretrained scBERT token embeddings.
        x = self.pretrained_encoder.token_emb(x)

        # Original scBERT gene positional embeddings
        x += self.pretrained_encoder.pos_emb(x)

        # Add the diffusion-step embedding to all positions
        x += self.step_emb(step_ids).unsqueeze(1)

        # Add the condition embedding only to the last special slot, to avoid
        # shifting gene positions and preserves the pretrained alignment

        # x[batch_size, #(g1,g2,...,gN), dim(in performer dim=200) ]
        # x[:, -1, :] = self.celltype_emb(celltype_ids) 
        # add celltype emb to each token
        x += self.celltype_emb(celltype_ids).unsqueeze(1)

        # scBERT forward path
        x = self.pretrained_encoder.dropout(x)
        layer_pos_emb = self.pretrained_encoder.layer_pos_emb(x)
        x = self.pretrained_encoder.performer(x, pos_emb=layer_pos_emb)
        x = self.pretrained_encoder.norm(x)
        x = self.pretrained_encoder.to_out(x)
        return x  # [B, L, NUM_CLASSES]

    def choose_bin(self, logits):
        # During training we keep the raw logits for CrossEntropyLoss.
        # When we want to choose the predicted bin, we first apply softmax
        # and then take the argmax over the token dimension.
        probs = self.softmax(logits)
        chosen_bins = probs.argmax(dim=-1)
        return probs, chosen_bins


    def predict_bins(self, x, celltype_ids, step_ids):
        logits = self.forward(x, celltype_ids, step_ids)
        return self.choose_bin(logits)