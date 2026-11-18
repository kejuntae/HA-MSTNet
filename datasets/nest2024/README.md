# NEST UMAR data preparation

This directory provides data preprocessing code only. It does not distribute
the original NEST dataset or its supplementary materials. Download the required
files yourself from the official [UMAR dataset on Figshare (version 2)](https://doi.org/10.6084/m9.figshare.24551992.v2).

## Download the input files

Create `datasets/nest2024/UMAR/` and place these four CSV files there without
renaming them:

- `umar_2019-07-01_2020-07-01_wide_descId.csv`
- `umar_2020-07-01_2021-07-01_wide_descId.csv`
- `umar_2021-07-01_2022-07-01_wide_descId.csv`
- `umar_2022-07-01_2023-07-01_wide_descId.csv`

No occupancy spreadsheet or data from DFAB or SolAce is required.

## Generate the training inputs

From the repository root, install the dependencies in `requirements.txt`, then run:

```bash
python datasets/nest2024/preprocessing_UMAR.py --save-all
python datasets/nest2024/generate_occupancy.py
```

The first command cleans the raw data, estimates room-level HVAC power,
resamples to 15-minute intervals, and writes `UMAR_2024.csv` and
`UMAR_2024_room{272,273,274}.csv` to `datasets/data_preprocessed/NEST2024/`.
It overwrites existing files with those names. The second command adds the
`occupancy` column to those four files. This schedule-based proxy is 0 on
weekdays from 08:00 (inclusive) to 18:00 (exclusive), and 1 otherwise. It is
not measured occupancy.

`generate_room_subsets.py` can regenerate the room tables from an existing
wide table.

## Source and license

Cite the [NEST Open Building Data collection](https://doi.org/10.6084/m9.figshare.c.7178787.v1)
and the [accompanying data paper](https://doi.org/10.1038/s41597-024-03292-2).
The upstream UMAR release is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
