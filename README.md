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
| 0. Explore | list the archive, profile every CSV, label/time/protocol/duplicate analysis | done |
| 1. Ingest | CSV -> typed Parquet parts with context features; time-blocked train/val/test split | done |
| 2. Prepare | per-segment device IP, classes, split + episode tags, merged rows with weights, shortcut check | **ready** |
| 3. Train | imbalance-aware XGBoost / Random Forest with checkpoints | after step 2 |
| 4. Evaluate + explain | per-class metrics, confusion, SHAP, dashboard tables | after step 3 |

## Run step 0 in Colab

1. Open the notebook:
   <https://colab.research.google.com/github/madhumithakk01/ds/blob/main/notebooks/00_explore_dataset.ipynb>
2. Runtime -> Change runtime type -> **CPU** (keep GPU time for training).
3. The dataset is expected at `MyDrive/Ds/GothamDataset2025.zip` (`Ds` is a shortcut to the
   shared folder: in "Shared with me", right-click it -> Organize -> Add shortcut -> My Drive).
   If the file name differs, cell 2 lists the folder; set `ZIP_PATH` in cell 1 to match.
   Outputs go to `MyDrive/gotham_ids`, in your own Drive, not in the shared folder.
4. Run all cells. If Colab disconnects, reconnect and run them again:
   finished files are skipped and only the interrupted one is read again.
5. Copy the text printed by cell 7 (`SUMMARY.md`) to plan the next steps.

## Run step 1 in Colab

1. Open <https://colab.research.google.com/github/madhumithakk01/ds/blob/main/notebooks/01_ingest.ipynb>
2. CPU runtime, same `ZIP_PATH` / `WORK_DIR` as step 0, run all cells (about 30-60 min, ~1 GB on Drive).
3. After a disconnect run cells 2 and 4 again: it continues from the last checkpoint.
4. Copy the text printed by cell 6 (`INGEST.md`) into the chat.

What step 1 decides (agreed after step 0):

* **Classes:** six coarse classes (Normal, DoS, Network Scanning, Brute Force, C&C Communication,
  Infection); the raw label `Unknown` is left out (`labels.drop_raw`). The 17 raw labels stay in the
  data for a finer second evaluation.
* **Split:** per raw label, in time order. Each attack episode (and benign traffic) is cut into
  blocks of at most 1 h; in every block the first 60 % of packets are train, the next 10 % validation,
  the last 30 % test, with a short gap left out in between. A random split would put near-identical
  packets of the same flood into train and test; a plain early/late split would leave whole attack
  types out of one side. Details: `gids/split.py`.
* **Unit:** one prediction per packet, with context features (`ctx_*`) that only look *back* in time
  within the same capture: packets/bytes per source, destination, IP pair and 5-tuple flow in the last
  1 s / 10 s, inter-arrival times, new destination ports per source. IPs, MACs, device and time are
  kept for analysis but are not features.

## Run step 2 in Colab

1. Open <https://colab.research.google.com/github/madhumithakk01/ds/blob/main/notebooks/02_prepare.ipynb>
2. CPU runtime, run all cells (about 15-30 min). After a disconnect run cells 1, 2 and 4 again.
3. Copy the text printed by cell 6 (`PREPARE.md`) into the chat.

Step 1 showed that some "-1" files (`combined-cycle-1`, `cooler-motor-1`, `predictive-maintenance-1`, ...)
are several captures joined end to end, so step 2 finds the device IP per capture segment, not per file.

## Commands

```
python -m gids env                 # hardware, disk, GPU, library versions, zip found?
python -m gids selftest            # end-to-end check on a small synthetic archive (crash + resume included)
python -m gids inventory           # archive listing, CSV headers, first rows, read speed
python -m gids explore             # full scan of every CSV (resumable) -> SUMMARY.md
python -m gids explore --max-files 5        # quick look at the five smallest files
python -m gids explore --max-minutes 60     # stop cleanly after an hour
python -m gids ingest              # step 1: CSV -> Parquet + context features, then the split (resumable)
python -m gids ingest --max-files 5         # quick look at the five smallest files
python -m gids prepare             # step 2: classes, split/episode tags, merged rows, shortcut check (resumable)
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
reports/ingest/
  INGEST.md, summary.json            split per class / raw label, features, files, things to check
  split_by_label.csv, split_by_class.csv, split_blocks.csv, split_episodes.csv
  features.csv, files.csv            feature ranges and fill rates; per-file rows, parts, device IP
reports/prepare/
  PREPARE.md, summary.json           rows per class/split, shortcut check, value tables, segments
  shortcut_check.csv, values_*.csv, medians_by_class.csv, rows_by_class.csv, rows_by_label.csv,
  segments.csv                       device IP, share, time range and labels of every capture segment
data/explore/                        per-file scan results + progress ledger
data/ingest-<hash>/                  Parquet parts (one or more per CSV), per-file metadata,
                                     split-<hash>.json, _state/ checkpoints of half-done files
data/prepare-<hash>/rows/            one Parquet table per CSV: features, split, cls, label, episode,
                                     weight (identical packets merged), device_type; vocab.json
logs/<run>.jsonl, logs/<run>.log     every event of every run
runs/<run>/manifest.json             command, config, code commit, environment, final status
state/                               lock, heartbeat, error tracebacks
```

## How resuming works

Each unit of work (one CSV file; inside a file, every 128 MB of ingested text) is recorded in an append-only ledger
only after its results are safely written; files are written to a temporary
name and renamed into place.  On restart, units whose results exist are
skipped.  Because Google Drive can lose the last writes of a VM that dies,
the ledger is never trusted alone - the result files must also be present.
A lock in `state/lock.json` stops two sessions from writing at the same
time; a lock from a dead session is taken over automatically after 5 minutes.

## Configuration

Defaults are in `configs/default.yaml`.  Output folders are named after a hash of the settings
that shaped them (`data/ingest-<hash>`), so changing a setting never mixes old and new results.  To change settings without editing
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
