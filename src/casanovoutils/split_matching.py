import argparse
import os
import random
import logging
from pyteomics import mgf
from collections import Counter
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "output.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def read_baseline_peptides(baseline_path):
    """
    Read baseline mgf and return a set of peptide sequences found.
    
    I had previously read all of the baseline files and stored them as .txt files, so I will 
    be opening those instead of the original .mgf files.
    """
    peptides = set()
    with open(baseline_path, "r", encoding="utf-8") as f:
        for line in f:
            peptides.add(line.strip())
    return peptides

def parse_charge(charge_info):
    """
    Parse charge information
    """
    if charge_info is None:
        return None
    if isinstance(charge_info, str):
        m = re.search(r'(\d+)', charge_info)
        if m:
            return int(m.group(1))
        return None
    if isinstance(charge_info, (list, tuple)):
        try:
            return parse_charge(charge_info[0])
        except Exception:
            return None
    try:
        return int(charge_info)
    except Exception:
        return None

def safe_title(params):
    """
    Create a safe title or get title.
    """
    if not isinstance(params, dict):
        return None
    t = params.get("title")
    if t:
        return t
    if "filename" in params and "scan" in params:
        return f"{params.get('filename')}:scan:{params.get('scan')}"
    return None

def collect_mgf_spectra_from_dir(input_dir):
    """
    Walks input directory, gets valid spectra, and adds them to dictionary 
    mapping sequences to their associated spectra.
    """
    distinct = {}
    num_skipped_no_seq = 0
    skipped_no_title = 0
    skipped_high_charge = 0
    num_skipped_peaks = 0

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(".mgf"): 
                continue
            file_path = os.path.join(root, file)
            logging.info(f"Currently processing {file_path}")
            try:
                for spectrum in mgf.read(file_path):
                    params = spectrum.get("params", {}) or {}
                    seq = params.get("seq")

                    if "U" in seq: #this check was explicitly done because there are peptides with "U" in Michael's data
                        logging.info("Peptide with U found") 
                        num_skipped_no_seq += 1
                        continue

                    if seq is None:
                        num_skipped_no_seq += 1
                        continue

                    mz_arr = spectrum.get("m/z array")
                    int_arr = spectrum.get("intensity array")

                    if mz_arr is None or int_arr is None:
                        num_skipped_peaks += 1
                        continue

                    if len(mz_arr) != len(int_arr):
                        num_skipped_peaks += 1
                        logging.info("Found malformed spectra with not equal m/z and intensity")
                        continue

                    if len(mz_arr) < 20:
                        num_skipped_peaks += 1
                        logging.info("Found spectra with less than 20 m/z")
                        continue

                    if len(int_arr) < 20: 
                        num_skipped_peaks += 1
                        logging.info("Found spectra with less than 20 intenisty")
                        continue

                    title = safe_title(params)
                    if title is None:
                        skipped_no_title += 1
                        continue
                    charge_info = params.get("charge")
                    charge_val = parse_charge(charge_info)
                    if charge_val > 3:
                        skipped_high_charge += 1
                        continue

                    if seq not in distinct:
                        distinct[seq] = []
                    distinct[seq].append(spectrum)

            except Exception as e:
                logging.warning(f"Failed to read {file_path}: {e}")
    return distinct, num_skipped_no_seq, skipped_no_title, skipped_high_charge, num_skipped_peaks

def main():
    parser = argparse.ArgumentParser(description="Train-test-validation split a mgf dataset at the peptide level")
    parser.add_argument("--input_path", required=True, help="Path to directory containing mgf files (recursively searched)")
    parser.add_argument("--train_split", type=float, required=True, help="Fraction for training")
    parser.add_argument("--test_split", type=float, required=True, help="Fraction for testing")
    parser.add_argument("--validation_split", type=float, required=True, help="Fraction for validation")
    parser.add_argument("--output_dir", required=True, help="Directory to save files in")
    parser.add_argument("--train_file", required=True, help="Path to baseline training .txt file")
    parser.add_argument("--test_file", required=True, help="Path to baseline testing .txt file")
    parser.add_argument("--val_file", required=True, help="Path to baseline validation .txt file")

    args = parser.parse_args()

    input_dir = args.input_path
    train_split = args.train_split
    test_split = args.test_split
    val_split = args.validation_split
    output_dir = args.output_dir
    baseline_train = args.train_file
    baseline_test = args.test_file
    baseline_val = args.val_file

    setup_logging(output_dir)

    logging.info(f"Processing input dir: {input_dir}")
    logging.info(f"Train/Test/Val splits (peptide-level): {train_split}/{test_split}/{val_split}")
    logging.info(f"Output directory: {output_dir}")

    # Read baselines
    train_baseline_peps = read_baseline_peptides(baseline_train)
    test_baseline_peps = read_baseline_peptides(baseline_test)
    val_baseline_peps = read_baseline_peptides(baseline_val)
    logging.info(f"{len(train_baseline_peps)} peptides in train baseline, {len(test_baseline_peps)} in test baseline, {len(val_baseline_peps)} in val baseline")

    # Collect spectra from input dir recursively
    distinct_peptide_and_spectra, num_skipped_no_seq, skipped_no_title, skipped_high_charge, skipped_peaks = collect_mgf_spectra_from_dir(input_dir)
    logging.info(f"Found {len(distinct_peptide_and_spectra)} distinct peptides in input dir")
    logging.info(f"{skipped_peaks} spectra for having too little peaks")
    logging.info(f"{num_skipped_no_seq} spectra skipped for missing seq; {skipped_no_title} skipped for missing title")
    logging.info(f"{skipped_high_charge} spectra with a charge > 3")
    
    # Sort peptides into train/test/val using baseline sets first
    train_peps = []
    test_peps = []
    val_peps = []
    extra_peps = []
    
    for peptide in distinct_peptide_and_spectra.keys():
        if peptide in train_baseline_peps:
            train_peps.append(peptide)
        elif peptide in test_baseline_peps:
            test_peps.append(peptide)
        elif peptide in val_baseline_peps:
            val_peps.append(peptide)
        else:
            extra_peps.append(peptide)

    logging.info(f"{len(train_peps)} training peptides before filling, {len(test_peps)} test, {len(val_peps)} val, {len(extra_peps)} extras")

    # Shuffle extras and fill based on requested fractions (peptide-level)
    random.seed(42)
    random.shuffle(extra_peps)
    num_peptides = len(distinct_peptide_and_spectra)
    n_train_target = int(train_split * num_peptides)
    n_val_target = int(val_split * num_peptides)
    n_test_target = int(test_split * num_peptides)

    # Fill training
    if len(train_peps) < n_train_target:
        need = min(n_train_target - len(train_peps), len(extra_peps))
        train_peps.extend(extra_peps[:need])
        extra_peps = extra_peps[need:]

    # Fill testing
    if len(test_peps) < n_test_target:
        need = min(n_test_target - len(test_peps), len(extra_peps))
        test_peps.extend(extra_peps[:need])
        extra_peps = extra_peps[need:]

    # Fill validation
    if len(val_peps) < n_val_target:
        need = min(n_val_target - len(val_peps), len(extra_peps))
        val_peps.extend(extra_peps[:need])
        extra_peps = extra_peps[need:]

    logging.info(f"{len(train_peps)} training peptides after fill (target {n_train_target})")
    logging.info(f"{len(test_peps)} testing peptides after fill (target {n_test_target})")
    logging.info(f"{len(val_peps)} validation peptides after fill (target {n_val_target})")
    logging.info(f"{len(extra_peps)} peptides remaining unassigned")

    # Helper to gather spectra for a split and compute split counters
    def gather_split_spectra(peptide_list):
        specs = []
        peptide_seqs = []
        for pep in peptide_list:
            peptide_seqs.append(pep)
            spectra_list = distinct_peptide_and_spectra.get(pep, [])
            for spec in spectra_list:
                specs.append(spec)

        return specs, peptide_seqs

    train_specs, peptide_seqs_train = gather_split_spectra(train_peps)
    test_specs, peptide_seqs_test = gather_split_spectra(test_peps)
    val_specs, peptide_seqs_val = gather_split_spectra(val_peps)
    extra_specs, peptide_seqs_extra = gather_split_spectra(extra_peps)
    
    # Write mgf output files (write by peptide for efficiency)
    train_file = os.path.join(output_dir, "train_all_processed.mgf")
    test_file = os.path.join(output_dir, "test_all_processed.mgf")
    val_file = os.path.join(output_dir, "validation_all_processed.mgf")
    extra_file = os.path.join(output_dir, "extra_all_processed.mgf")

    def write_mgf(peptide_list, out_path, type):
        written = 0
        curr_pep = 0
        with open(out_path, "w") as f_out:
            for pep in peptide_list:
                specs = distinct_peptide_and_spectra.get(pep, [])
                if specs:
                    written += len(specs)
                    curr_pep += 1
                    mgf.write(specs, f_out)
                    logging.info(f"Currently wrote {written} spectra for {type}")
                    logging.info(f"Currently written {curr_pep} out of {len(peptide_list)} for {type}")
        logging.info(f"Wrote {written} spectra (from {len(peptide_list)} peptides) to {out_path} for {type}")
    
    def write_peps_list(peptide_seqs, type): 
        path = os.path.join(output_dir, f"{type}.txt")
        with open(path, "w", encoding="utf-8") as f: 
            for sequence in peptide_seqs:
                f.write(sequence + "\n")
    
    logging.info("Starting to write peptide lists")
    write_peps_list(peptide_seqs_train, "training")
    write_peps_list(peptide_seqs_test, "testing")
    write_peps_list(peptide_seqs_val, "validation")
    write_peps_list(peptide_seqs_extra, "extra")

    logging.info("Starting to writing mgf files")
    write_mgf(train_peps, train_file, "training")
    write_mgf(test_peps, test_file, "testing")
    write_mgf(val_peps, val_file, "validation")
    write_mgf(extra_peps, extra_file,"extra")

    logging.info("Finished writing train/test/validation MGF files")
    logging.info(f"Training file at: {train_file}")
    logging.info(f"Testing file at: {test_file}")
    logging.info(f"Validation file at: {val_file}")

if __name__ == "__main__":
    main()

