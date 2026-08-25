"""
Splitters for crossvalidation in the pipeline scarcity

Following the generic interface of sklearn (GroupKFold, StratifiedGroupKFold) :
.split(X, y, groups) -> generator of (train_idx, test_idx) in integer positions.
"""
import numpy as np
import pandas as pd


class LeaveOneTargetYearOut:
    """
    Takes out one TY (target year) at a time from the train. All weather scenarios (WS)
    remain in training for the remaining TY including the 16 target ones
    """

    def __init__(self, ty_values: list | None = None):
        self.ty_values = ty_values

    def _resolve_ty_values(self, X: pd.DataFrame) -> list:
        ty_values = self.ty_values if self.ty_values is not None else sorted(pd.unique(X['TY'].values))
        if len(ty_values) < 2:
            raise ValueError(
                f"LeaveOneTargetYearOut needs 2 TY at least {ty_values} "
                "With a single TY there is nothing to retain for the test set "
                "Use GroupKFold(by WS) while waiting to have more TY in the dataset"
            )
        return ty_values

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """ groups is ignored (only accepted for compatibility with GroupKFold/StratifiedGroupKFold)"""

        ty_col = X['TY'].values
        for held_out_ty in self._resolve_ty_values(X):
            test_idx = np.where(ty_col == held_out_ty)[0]
            train_idx = np.where(ty_col != held_out_ty)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X: pd.DataFrame, y=None, groups=None) -> int:
        return len(self._resolve_ty_values(X))
