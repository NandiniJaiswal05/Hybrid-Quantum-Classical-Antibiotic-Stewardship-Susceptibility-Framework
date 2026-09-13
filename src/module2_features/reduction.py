import torch
import torch.nn as nn
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Enforce Chronological order: from Baseline to Acute Disruptions
TIME_STEPS = ['ALL', '180', '90', '30', '14', '7']
TIME_STEP_SET = set(TIME_STEPS)


def parse_temporal_features(df: pd.DataFrame):
    """
    Slices the flat clinical dataframe into sequential temporal groups.

    BUGFIX: the previous implementation matched window tokens with
    `f" {t}" in col or col.endswith(str(t))`. `col.endswith(str(t))` in
    particular is a bare substring/suffix check, so for t='7' *any* column
    whose name happens to end in the digit 7 (e.g. a static feature like
    "comorbidity - condition_37" or "demographics - zip_87") would be
    silently misfiled into the 7-day temporal group instead of
    static_features. That's inconsistent with how app.py's own tokenizer
    (split_window) classifies the same columns -- it splits on whitespace
    and checks for an *exact* token match -- and it can corrupt which
    columns get fed into the sequential MPS layer vs. treated as static,
    degrading the model in a way that would show up as poor calibration /
    persistently high uncertainty, not just a crash.

    This version tokenizes the same way (splitting on whitespace, treating
    "-" as a separator) and only assigns a column to a temporal group when
    a window token appears as a standalone token, matching split_window's
    behavior exactly.
    """
    feature_groups = {t: [] for t in TIME_STEPS}
    static_features = []

    for col in df.columns:
        tokens = col.replace("-", " ").split()
        matched_step = next((t for t in tokens if t in TIME_STEP_SET), None)

        if matched_step is not None:
            feature_groups[matched_step].append(col)
        else:
            static_features.append(col)

    return feature_groups, static_features


class MatrixProductStateLayer(nn.Module):
    """
    Classical Tensor Network: Iteratively contracts temporal features,
    maintaining a hidden memory state passed forward through time.
    """
    def __init__(self, input_dims, bond_dim=16, output_dim=8):
        super().__init__()
        self.time_steps = len(input_dims)
        self.output_dim = output_dim
        self.cells = nn.ModuleList()

        for i, in_dim in enumerate(input_dims):
            # The first time step takes only its own features; subsequent steps include memory bond
            in_features = in_dim if i == 0 else in_dim + bond_dim
            # The final contraction outputs exactly 8 dimensions for the Qubits
            out_features = bond_dim if i < self.time_steps - 1 else output_dim

            self.cells.append(nn.Sequential(
                nn.Linear(in_features, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, out_features)
            ))

    def forward(self, x_seq):
        """
        x_seq: list of PyTorch tensors corresponding to chronological slices.
        Returns: Dense 8D tensor representation.
        """
        # BUGFIX (defensive): if x_seq is empty (no temporal groups found for
        # the current feature set -- e.g. after a categorization fix changes
        # which columns count as temporal) the old code returned `memory`
        # while it was still None, which then blew up later as
        # `torch.cat([None, x_static], ...)` with a confusing TypeError.
        # Fail clearly here instead.
        if not x_seq:
            raise ValueError(
                "MatrixProductStateLayer received no temporal feature groups. "
                "This model architecture requires at least one temporal window; "
                "check that parse_temporal_features is finding window-suffixed "
                "columns in the current feature set."
            )

        memory = None
        for i, x_t in enumerate(x_seq):
            if i == 0:
                memory = self.cells[i](x_t)
            else:
                combined = torch.cat([x_t, memory], dim=-1)
                memory = self.cells[i](combined)
        return memory