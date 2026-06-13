# LUCID - Agent Development Guide

## Architecture Overview (v3.0)

LUCID is a unified pipeline for SISR dataset filtering. It runs as a **single CLI command**:

```bash
python -m lucid --input /path/to/raw_images --output /path/to/filtered_tiles
```

### Pipeline Flow

```
Input images
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  INGEST STAGE                                              │
│  1. Signal filter (entropy, blockiness, gradient, etc.)     │
│  2. Extract 512px tiles (ICNet training size), write to workspace/raw_tiles/│
│  3. ICNet GPU scoring, keep passing tiles in workspace/scored│
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  DEDUPE STAGE                                            │
│  1. ResNet18 feature fingerprints on GPU                   │
│  2. Cosine similarity against unique pool                  │
│  3. Move duplicates to workspace/redundant/                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  FINALIZE STAGE                                          │
│  1. Sequential naming (00001_dataset.png)                │
│  2. Write master_lineage.csv for provenance                │
│  3. Output to user-specified --output folder              │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

- `lucid/config.py` - Singleton configuration (CLI args, defaults)
- `lucid/pipeline.py` - Main orchestrator, runs all stages
- `lucid/state.py` - SQLite-based resume/crash recovery
- `lucid/core/metrics.py` - Signal quality metrics (entropy, Laplacian, etc.)
- `lucid/core/validation.py` - Corrupt image detection
- `lucid/core/filtering.py` - Tile extraction + ICNet scoring
- `lucid/core/dedupe.py` - GPU-based deduplication
- `lucid/core/finalize.py` - Final consolidation + lineage CSV
- `lucid/io.py` - Buffered, throttled disk writes
- `lucid/resources.py` - VRAM query, disk space estimation
- `lucid/platform.py` - Cross-platform helpers (process priority)
- `lucid/gui.py` - Optional Tkinter GUI wrapper

### Output Structure

When run with `--output /workspace` and `--input /raw/{photos}`:

```
/workspace/
├── tiles/                          # Final filtered tiles
├── master_lineage.csv              # Tile -> source mapping
└── lucid_workspace_photos/         # Internal scratch workspace (auto-cleaned with --cleanup)
    ├── raw_tiles/                  # Intermediate: signal-passing tiles
    ├── scored/                     # Intermediate: ICNet-passing tiles
    │   └── redundant/              # Tiles removed as duplicates
    ├── scores/
    │   └── ingest_scores.csv       # Per-tile ICNet complexity score
    └── .lucid/
        └── lucid_state.db          # SQLite state DB for --resume
```

The scratch workspace is named `lucid_workspace_<input-folder>` so multi-dataset
runs never collide inside the same `--output`. Pass `--workspace PATH` to put the
scratch directory on a fast drive independently of the final output destination.

### Resume / Crash Recovery

- State is tracked in `<workspace>/.lucid/lucid_state.db` (SQLite)
- Source images: `pending` → `extracted` → `scored` → `finalized`
- Resume with `--resume` flag
- Disk space check before processing (estimates needed space)

### Large Image Handling

- Images larger than 8192px on any dimension are auto-resized before tiling
- Batch size auto-detects from available VRAM (via `recommend_batch_size()`)
- I/O throttling detects removable drives and adds 5ms sleep between writes

### CLI Arguments

```
--input PATH          Required: Input folder with images
--output PATH         Required: Output folder for final tiles
--workspace PATH      Optional: Scratch workspace (default: lucid_workspace_<input> inside output)
--icnet PATH          Optional: ICNet weights path (default: auto-download)
--tile-size INT       Tile dimension (default: 512, matches ICNet training size)
--threshold FLOAT     ICNet complexity threshold (default: 0.45)
--dedupe-threshold    Cosine similarity threshold (default: 0.96)
--workers INT         CPU workers (default: cpu_count - 2)
--batch INT           GPU batch size (default: auto from VRAM)
--multiscale          Enable multi-scale tiling
--io-throttle-ms FLOAT I/O sleep duration
--strict              Abort on first corrupt image
--resume              Resume previous run
--stage {all,ingest,dedupe,finalize} Run specific stage
--dry-run             Validate only, no writes
--verbose             Enable debug logging
```

### Testing

```bash
# Install
pip install -e .

# Run on test folder
python -m lucid --input ./test_images --output ./filtered --tile-size 256 --dry-run

# GUI (if tkinter available)
lucid-gui
```

### Legacy Scripts

The individual scripts `01_ingest.py`, `02_dedupe.py`, `03_finalize.py` are deprecated but still work. They forward to the new unified pipeline.