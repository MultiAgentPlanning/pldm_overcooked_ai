import pandas as pd
import torch
from torch.utils.data import random_split
from typing import Optional
from solution.pldm.data_processor import OvercookedDataset  # Replace with actual import


def get_validation_dataframe(data_path: str,
                             seed = 101,
                             state_encoder_type: str = 'grid',
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.1,
                             max_samples: Optional[int] = None,
                             return_terminal: bool = False) -> pd.DataFrame:
    """
    Return the validation split of the Overcooked dataset as a DataFrame.
    """
    dataset = OvercookedDataset(data_path, state_encoder_type, max_samples, return_terminal)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty! No transitions were loaded from the data file.")

    test_size = max(1, int(test_ratio * len(dataset)))
    val_size = max(1, int(val_ratio * len(dataset)))
    train_size = len(dataset) - val_size - test_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Return raw dataframe rows corresponding to val indices
    val_indices = val_dataset.indices
    original_data = pd.read_csv(data_path)

    if max_samples is not None:
        original_data = original_data.iloc[:max_samples]

    val_df = original_data.iloc[val_indices].reset_index(drop=True)
    return val_df


def main():
    val_df = get_validation_dataframe(
        data_path="data/raw/2020_hh_trials.csv"
    )

    val_df.to_csv("2020_samples.csv", index=False)

if __name__ == "__main__":
    main()
