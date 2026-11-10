#!/usr/bin/env python3
"""
Convert 1000 Genomes PED file to individual JSON metadata files per sample.

INTERNAL_VERSION = 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# 1000 Genomes superpopulation codes
SUPERPOPULATION_DESCRIPTIONS: dict[str, str] = {
    "AFR": "African",
    "AMR": "Ad Mixed American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}

# 1000 Genomes population codes
POPULATION_DESCRIPTIONS: dict[str, str] = {
    "ACB": "African Caribbean in Barbados",
    "ASW": "African Ancestry in Southwest US",
    "BEB": "Bengali in Bangladesh",
    "CDX": "Chinese Dai in Xishuangbanna",
    "CEU": "Utah Residents with Northern and Western European Ancestry",
    "CHB": "Han Chinese in Beijing",
    "CHS": "Han Chinese South",
    "CLM": "Colombian in Medellin",
    "ESN": "Esan in Nigeria",
    "FIN": "Finnish in Finland",
    "GBR": "British in England and Scotland",
    "GIH": "Gujarati Indians in Houston",
    "GWD": "Gambian in Western Division",
    "IBS": "Iberian Populations in Spain",
    "ITU": "Indian Telugu in the UK",
    "JPT": "Japanese in Tokyo",
    "KHV": "Kinh in Ho Chi Minh City",
    "LWK": "Luhya in Webuye, Kenya",
    "MSL": "Mende in Sierra Leone",
    "MXL": "Mexican Ancestry in Los Angeles",
    "PEL": "Peruvian in Lima",
    "PJL": "Punjabi in Lahore",
    "PUR": "Puerto Rican in Puerto Rico",
    "STU": "Sri Lankan Tamil in the UK",
    "TSI": "Toscani in Italy",
    "YRI": "Yoruba in Ibadan, Nigeria",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PED file samples to individual JSON metadata files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i samples.ped -o /output_dir/
  %(prog)s -i samples.ped -o /output_dir/ --dir-per-sample
        """,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input PED file (whitespace-separated with header)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--dir-per-sample",
        action="store_true",
        help="Write JSON to existing subdirectory named after sample ID "
             "(e.g., output_dir/NA12775/NA12775_metadata.json)",
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create sample subdirectories if they don't exist (only with --dir-per-sample)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_false",
        dest="pretty",
        help="Disable pretty-printing (compact JSON)",
    )

    return parser.parse_args()


def read_ped_file(ped_path: Path) -> list[dict]:
    """
    Read a PED file and return a list of sample dictionaries.

    Parameters
    ----------
    ped_path : Path
        Path to the PED file.

    Returns
    -------
    list[dict]
        List of dictionaries, one per sample, with keys from the header.
    """
    samples: list[dict] = []

    with open(ped_path, "r") as f:
        header_line = f.readline().strip()
        if not header_line:
            raise ValueError("Empty PED file or missing header")

        columns = header_line.split()

        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != len(columns):
                print(
                    f"Warning: Line {line_num} has {len(fields)} fields, "
                    f"expected {len(columns)}. Skipping.",
                    file=sys.stderr,
                )
                continue

            sample_dict = dict(zip(columns, fields))

            # Convert Sex to descriptive value
            if "Sex" in sample_dict:
                sex_code = sample_dict["Sex"]
                sample_dict["Sex"] = int(sex_code)
                sample_dict["Sex_label"] = {1: "Male", 2: "Female"}.get(
                    int(sex_code), "Unknown"
                )

            # Add Population description
            if "Population" in sample_dict:
                pop_code = sample_dict["Population"]
                sample_dict["Population_description"] = POPULATION_DESCRIPTIONS.get(
                    pop_code, "Unknown"
                )

            # Add Superpopulation description
            if "Superpopulation" in sample_dict:
                superpop_code = sample_dict["Superpopulation"]
                sample_dict["Superpopulation_description"] = SUPERPOPULATION_DESCRIPTIONS.get(
                    superpop_code, "Unknown"
                )

            # Convert 0 values in FatherID/MotherID to null
            for parent_field in ["FatherID", "MotherID"]:
                if parent_field in sample_dict and sample_dict[parent_field] == "0":
                    sample_dict[parent_field] = None

            samples.append(sample_dict)

    return samples


def write_sample_json(
    sample: dict,
    output_dir: Path,
    dir_per_sample: bool,
    create_dirs: bool,
    pretty: bool,
) -> Optional[Path]:
    """
    Write a single sample's metadata to a JSON file.

    Parameters
    ----------
    sample : dict
        Sample metadata dictionary.
    output_dir : Path
        Base output directory.
    dir_per_sample : bool
        If True, write to output_dir/SampleID/SampleID_metadata.json.
    create_dirs : bool
        If True and dir_per_sample is True, create missing directories.
    pretty : bool
        If True, pretty-print the JSON.

    Returns
    -------
    Optional[Path]
        Path to the written file, or None if skipped.
    """
    sample_id = sample.get("SampleID")
    if sample_id is None:
        print("Warning: Sample missing SampleID, skipping.", file=sys.stderr)
        return None

    if dir_per_sample:
        sample_dir = output_dir / sample_id
        if not sample_dir.exists():
            if create_dirs:
                sample_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(
                    f"Warning: Directory {sample_dir} does not exist, "
                    f"skipping sample {sample_id}. Use --create-dirs to create it.",
                    file=sys.stderr,
                )
                return None
        output_path = sample_dir / f"{sample_id}_metadata.json"
    else:
        output_path = output_dir / f"{sample_id}_metadata.json"

    indent = 2 if pretty else None
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=indent)
        if pretty:
            f.write("\n")

    return output_path


def main() -> int:
    args = parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Create output directory if needed
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    # Read PED file
    print(f"Reading PED file: {args.input}")
    try:
        samples = read_ped_file(args.input)
    except Exception as e:
        print(f"Error reading PED file: {e}", file=sys.stderr)
        return 1

    print(f"Found {len(samples)} samples")

    # Write JSON files
    written_count = 0
    skipped_count = 0

    for sample in samples:
        result = write_sample_json(
            sample=sample,
            output_dir=args.output_dir,
            dir_per_sample=args.dir_per_sample,
            create_dirs=args.create_dirs,
            pretty=args.pretty,
        )
        if result is not None:
            written_count += 1
        else:
            skipped_count += 1

    print(f"Written: {written_count} JSON files")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} samples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
