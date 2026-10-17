# Gotham IoT intrusion detection

Class-imbalance-aware threat classification on the **Gotham Dataset 2025**
(78 emulated IoT devices, MQTT / CoAP / RTSP traffic, packet-level CSVs),
built to run on the free tier of Google Colab:

* the 23.8 GB archive is read directly from Google Drive, never unpacked;
* every long step is split into small units whose progress is saved on Drive,
  so a disconnect costs at most the unit that was running;
* every run leaves a structured log, a manifest and progress state behind, so
  results can be explained without writing new code.

## Status

| Step | What it does | State |
|---|---|---|
| 0. Explore | list the archive, profile every CSV, label/time/protocol/duplicate analysis | **ready** |
| 1. Ingest | CSV -> typed Parquet parts with stream features | after reviewing step 0 |
| 2. Profile + compact | EDA, de-duplication with counts, leakage-free split | after step 1 |
| 3. Train | imbalance-aware XGBoost / Random Forest with checkpoints | after step 2 |
| 4. Evaluate + explain | per-class metrics, confusion, SHAP, dashboard tables | after step 3 |

## Run step 0 in Colab

1. Open the notebook:
   <https://colab.research.google.com/github/madhumithakk01/ds/blob/main/notebooks/00_explore_dataset.ipynb>
2. Runtime -> Change runtime type -> **CPU** (keep GPU time for training).
3. In cell 1 set `ZIP_PATH` if `GothamDataset2025.zip` is not in the root of My Drive.
   If the zip is in "Shared with me", add a shortcut to My Drive first.
4. Run all cells. If Colab disconnects, reconnect and run them again:
   finished files are skipped and only the interrupted one is read again.
5. Copy the text printed by cell 7 (`SUMMARY.md`) to plan the next steps.

## Commands

```
python -m gids env                 # hardware, disk, GPU, library versions, zip found?
python -m gids selftest            # end-to-end check on a small synthetic archive
python -m gids inventory           # archive listing, CSV headers, first rows, read speed
python -m gids explore             # full scan of every CSV (resumable) -> SUMMARY.md
python -m gids explore --max-files 5        # quick look at the five smallest files
python -m gids explore --max-minutes 60     # stop cleanly after an hour
python -m gids status              # progress, lock, recent runs (finished / killed)
python -m gids trace --errors      # warnings and errors with tracebacks
python -m gids unlock --yes        # remove a lock left by a dead session
```

Global options go before the command: `--work-dir`, `--set key=value`
(for example `--set explore.block_mb=16`), `--quiet`, `--force`.

## Outputs (in the work directory, default `MyDrive/gotham_ids`)

```
reports/explore/
  INVENTORY.md, inventory.csv/json   archive contents, CSV heads, read speed
  SUMMARY.md, summary.json           the overview to read first
  files.csv                          rows, size, capture segments, labels per file
  columns.csv                        fill rate, numeric/hex/repeated values, ranges, top values
  labels_*.csv                       label counts in total, per device and per device type
  label_mapping_preview.csv          raw label -> coarse class
  time_by_label.csv, time_by_device_type_label.csv, timeline.png
  protocols_by_label.csv
  duplicates_by_label.csv / _by_class.csv
                                     packets vs distinct header vectors, share of packets whose
                                     exact header vector also appears under another label, and
                                     the best recall reachable on header fields alone
  shared_vectors_*_pairs.csv         which labels share identical packets
  metadata/                          copies of the archive's metadata JSON files
  *.png                              charts
data/explore/                        per-file scan results + progress ledger
logs/<run>.jsonl, logs/<run>.log     every event of every run
runs/<run>/manifest.json             command, config, code commit, environment, final status
state/                               lock, heartbeat, error tracebacks
```

## How resuming works

Each unit of work (here: one CSV file) is recorded in an append-only ledger
only after its results are safely written; files are written to a temporary
name and renamed into place.  On restart, units whose results exist are
skipped.  Because Google Drive can lose the last writes of a VM that dies,
the ledger is never trusted alone - the result files must also be present.
A lock in `state/lock.json` stops two sessions from writing at the same
time; a lock from a dead session is taken over automatically after 5 minutes.

## Configuration

Defaults are in `configs/default.yaml`.  To change settings without editing
the repository, create `config.override.yaml` in the work directory on Drive,
or pass `--set` on the command line.

## Repository layout

```
gids/            pipeline package (python -m gids ...)
configs/         default settings
notebooks/       Colab notebooks, one per step
```

## Dataset

Belarbi et al., *Gotham Dataset 2025: A Reproducible Large-Scale IoT Network
Dataset for Intrusion Detection and Security Research*, Zenodo,
doi:10.5281/zenodo.14502760 (CC BY 4.0).
