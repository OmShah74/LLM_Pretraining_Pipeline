from __future__ import annotations

import argparse
from pathlib import Path

from src.data.pipeline import run_data_prep, validate_stream_sources
from src.utils.config import load_full_config
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data preparation pipeline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--validate-only", action="store_true", help="Validate streaming sources without extracting/cleaning the full datasets.")
    parser.add_argument("--roles", nargs="*", default=None, help="Optional subset of roles to validate or prepare.")
    args = parser.parse_args()

    config = load_full_config(args.config)
    if args.validate_only:
        output_path = Path(config["data"].processed_dir) / "data_source_validation.json"
        try:
            summary = validate_stream_sources(config["data"], args.roles)
        except Exception as exc:
            write_json(
                output_path,
                {
                    "validated_sources": [],
                    "errors": [{"message": f"{type(exc).__name__}: {exc}"}],
                },
            )
            raise
        write_json(output_path, summary)
        return
    results = run_data_prep(config["data"])
    summary = {
        "raw": [manifest.to_dict() for manifest in results["raw"]],
        "cleaned": [manifest.to_dict() for manifest in results["cleaned"]],
        "tokenizer": results["tokenizer"].to_dict(),
        "packed": [manifest.to_dict() for manifest in results["packed"]],
        "mixture": results["mixture"].to_dict(),
    }
    write_json(Path(config["data"].processed_dir) / "data_prep_summary.json", summary)


if __name__ == "__main__":
    main()
