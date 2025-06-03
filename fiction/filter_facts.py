import pathlib as pl
import argparse
from fiction.utils import load_facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=pl.Path,
        help="Starting dataset directory, with {train|valid|test}.txt and {entity2id|relation2id|ts2id}.json.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pl.Path,
        help="File where new facts will be dumped, one per line.",
    )
    parser.add_argument(
        "-miny",
        "--min-year",
        type=int,
        help="Minimum year for which to extract facts (included).",
    )
    parser.add_argument(
        "-maxy",
        "--max-year",
        type=int,
        help="Maximum year for which to generate facts (included).",
    )
    args = parser.parse_args()

    facts = (
        load_facts(args.dataset_dir / "train.txt")
        + load_facts(args.dataset_dir / "valid.txt")
        + load_facts(args.dataset_dir / "test.txt")
    )

    with open(args.output_file, "w") as f:
        for subj, rel, obj, ts in facts:
            year = int(ts.split("-")[0])
            if year >= args.min_year and year <= args.max_year:
                f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")
