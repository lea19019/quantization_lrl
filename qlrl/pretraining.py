'''
Entry point for the from-scratch model (docs/project/pretraining_model.md §5):

    python -m qlrl.pretraining configs/pretrain/tokenizer.yaml [--test]           the tokenizer
    python -m qlrl.pretraining configs/pretrain/tokenizer.yaml <index> [--test]   one corpus's
                                                                                  token files
    torchrun --nproc_per_node=<gpus> -m qlrl.pretraining configs/pretrain/llama_576m.yaml

A config with a model: section trains; the tokenizer config trains the tokenizer, or with an
index (the Slurm array task) encodes that corpus. --test: the first documents of each corpus
and a 2,048-token vocabulary, into <name>_test.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import sys
from pathlib import Path

import torch.distributed as dist
import yaml

from qlrl.pretrain import Pretraining, encode_corpus, train_tokenizer


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    test = "--test" in sys.argv
    index = [int(arg) for arg in sys.argv[2:] if arg.isdigit()]

    if "model" not in config:
        if index:
            encode_corpus(config, index[0], test)
        else:
            train_tokenizer(config, test)
        return

    if (Path(config["output"]) / config["name"] / "manifest.json").exists():
        print(f"{config['name']} is already trained", flush=True)
        return
    run = Pretraining(config)
    if run.run():  # False: the job's hours ran out; the next job resumes
        run.finish()
    if run.world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
