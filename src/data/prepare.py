from __future__ import annotations

import argparse
from pathlib import Path

from src.data.pipeline import run_data_prep, validate_stream_sources
from src.utils.config import load_full_config
from src.utils.io import write_json
from src.utils.logging import PipelineLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data preparation pipeline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--validate-only", action="store_true", help="Validate Hugging Face sources without materializing/cleaning datasets.")
    parser.add_argument("--roles", nargs="*", default=None, help="Optional subset of roles to validate or prepare.")
    args = parser.parse_args()

    config = load_full_config(args.config)
    logger = PipelineLogger(config["data"].log_dir, "data_prepare")
    if args.validate_only:
        output_path = Path(config["data"].processed_dir) / "data_source_validation.json"
        try:
            logger.event("data.validate", "starting source validation", roles=args.roles or "all")
            summary = validate_stream_sources(config["data"], args.roles)
        except Exception as exc:
            write_json(
                output_path,
                {
                    "validated_sources": [],
                    "errors": [{"message": f"{type(exc).__name__}: {exc}"}],
                },
            )
            logger.event("data.validate", "failed", error=f"{type(exc).__name__}: {exc}", output=str(output_path))
            raise
        write_json(output_path, summary)
        logger.event("data.validate", "complete", sources=len(summary["validated_sources"]), output=str(output_path))
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
