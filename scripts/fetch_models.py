"""Downloads every model the project uses and records the revisions.

ACT checkpoints come at the revisions pinned in configs/base.yaml; the VLM models at their current
revision, which is written to the output file and later to each score meta. The world-model
encoder comes through torch.hub at its pinned repo commit (TORCH_HOME). Uses HF_HOME.

    python scripts/fetch_models.py --out ~/amon-3070/model_revisions.json [--skip-vlm]
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from amon.settings import load_settings


def main() -> None:
    """Downloads and writes the revision file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-vlm", action="store_true", help="skip the two VLM models")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[fetch] %(message)s", force=True)
    settings = load_settings()

    revisions: dict[str, str] = {}
    for task in settings.tasks.values():
        logging.info("%s @ %s", task.policy_repo, task.policy_revision[:8])
        path = snapshot_download(task.policy_repo, revision=task.policy_revision)
        revisions[task.policy_repo] = Path(path).name
    if not arguments.skip_vlm:
        for repo in (settings.vlm.model, settings.vlm.fallback_model):
            logging.info("%s (several GB)", repo)
            revisions[repo] = Path(snapshot_download(repo)).name
    logging.info("%s from %s", settings.wm.encoder, settings.wm.encoder_repo)
    torch.hub.load(settings.wm.encoder_repo, settings.wm.encoder, trust_repo=True)
    revisions[settings.wm.encoder] = settings.wm.encoder_repo

    arguments.out.write_text(json.dumps(revisions, indent=2) + "\n", encoding="utf-8")
    logging.info("wrote %s", arguments.out)


if __name__ == "__main__":
    main()
