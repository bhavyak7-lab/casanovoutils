# casanovoutils

Utility CLI tools for **evaluating, visualizing, and manipulating peptide-spectrum match (PSM) data**, designed to work cleanly with **Casanovo**, **mzTab**, and **MGF** files.

This package provides:

* Precision–coverage (Prec–Cov) curve plotting with AUPC
* Amino-acid–level and peptide-level evaluation
* MGF downsampling by peptide
* Residue mass table utilities

All functionality is exposed via a **Fire-based CLI** entrypoint.

## CLI Overview

```bash
casanovoutils <command> [options]
```

Available commands:

* `graph-prec-cov` — plot precision–coverage curves
* `downsample-ms` — downsample MGF files by peptide
* `dump-residues` — export the default residue mass table

You can inspect help for any command via:

```bash
casanovoutils <command> --help
```

## Precision–Coverage Plotting (`graph-prec-cov`)

Plot and compare **precision–coverage curves** for peptide or amino-acid–level predictions, with **AUPC** reported directly in the legend.

### Key features

* Supports **mzTab** + optional **MGF** ground truth
* Peptide-level or amino-acid–level evaluation
* Multiple datasets on a single plot
* Uses Casanovo’s `aa_match_batch` evaluator internally

### Basic example (peptide-level)

```bash
casanovoutils graph-prec-cov \
  add-peptides \
    --mztab_path results.mztab \
    --name Casanovo \
    --ground_truth_col true_sequence \
  save prec_cov.png
```

### Multiple datasets in one figure

```bash
casanovoutils graph-prec-cov \
  --fig_width 6 \
  --fig_height 4 \
  --legend_location upper right \
  add-peptides \
    --mztab_path modelA.mztab \
    --name ModelA \
    --ground_truth_col peptide \
  add-peptides \
    --mztab_path modelB.mztab \
    --name ModelB \
    --ground_truth_mgf truth.mgf \
  save comparison.png
```

Because this is a **single Fire object**, state is preserved across chained calls.

### Amino-acid–level precision–coverage

If your mzTab contains per-AA scores (e.g. Casanovo output):

```bash
casanovoutils graph-prec-cov \
  add-amino-acids \
    --mztab_path results.mztab \
    --name Casanovo-AA \
    --scores_col opt_ms_run[1]_aa_scores \
    --ground_truth_col peptide \
  save aa_prec_cov.png
```

## MGF Downsampling (`downsample-ms`)

Downsample one or more MGF files by **sampling up to *k* spectra per peptide**.

Useful for:

* Dataset balancing
* Reducing redundancy
* Faster benchmarking

### Example

```bash
casanovoutils downsample-ms \
  input1.mgf input2.mgf \
  --outfile sampled.mgf \
  --k 2
```

Options:

* `--k` — max PSMs per peptide
* `--shuffle` — shuffle output spectra (default: true)
* `--random_seed` — reproducible sampling

## Residue Mass Tables

### Dump the default residue table

```bash
casanovoutils dump-residues residues.yaml
```

You can edit this YAML file and pass it back into plotting commands:

```bash
--residues_path custom_residues.yaml
```

This is useful for:

* Custom modifications
* Non-standard residues
* Mass tweaks for experimental work
