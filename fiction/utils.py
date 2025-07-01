from typing import Any, List, Optional, Tuple
import json
import pathlib as pl
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

Fact = Tuple[str, str, str, str]


def dump_json(value: Any, path: pl.Path, progress_msg: Optional[str] = None, **kwargs):
    if not progress_msg is None:
        print(progress_msg + "...", end="")
    with open(path, "w") as f:
        json.dump(value, f, **kwargs)
    if not progress_msg is None:
        print("done!")


def dump_facts(facts: List[Fact], path: pl.Path, progress_msg: Optional[str] = None):
    if not progress_msg is None:
        print(progress_msg + "...", end="")
    with open(path, "w") as f:
        for subj, rel, obj, ts in facts:
            f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")
    if not progress_msg is None:
        print("done!")


def load_facts(path: pl.Path, progress_msg: Optional[str] = None) -> List[Fact]:
    if not progress_msg is None:
        print(progress_msg + "...", end="")
    facts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            subj, rel, obj, ts = line.split("\t")
            facts.append((subj, rel, obj, ts))
    if not progress_msg is None:
        print("done!")
    return facts


@dataclass
class FactDataset:
    train_facts: list[Fact]
    valid_facts: list[Fact]
    test_facts: list[Fact]
    entity2id: dict[str, int]
    rel2id: dict[str, int]
    ts2id: dict[str, int]

    def all_facts(self) -> list[Fact]:
        return self.train_facts + self.valid_facts + self.test_facts

    def subj_entities(self) -> set[str]:
        return {f[0] for f in self.all_facts()}

    def obj_entities(self) -> set[str]:
        return {f[2] for f in self.all_facts()}

    def subj_facts(self) -> dict[str, list[Fact]]:
        subj_facts = defaultdict(list)
        for fact in self.all_facts():
            subj = fact[0]
            subj_facts[subj].append(fact)
        return subj_facts

    def map_to_idx(self) -> np.ndarray:
        facts = self.all_facts()
        subs = [self.entity2id[x[0]] for x in facts]
        rels = [self.rel2id[x[1]] for x in facts]
        objs = [self.entity2id[x[2]] for x in facts]
        tss = [self.ts2id[x[3]] for x in facts]
        return np.column_stack((subs, rels, objs, tss))


def load_fact_dataset(root: pl.Path) -> FactDataset:
    train_facts = load_facts(root / "train.txt")
    valid_facts = load_facts(root / "valid.txt")
    test_facts = load_facts(root / "test.txt")
    with open(root / "entity2id.json") as f:
        entity2id = json.load(f)
    with open(root / "relation2id.json") as f:
        rel2id = json.load(f)
    with open(root / "ts2id.json") as f:
        ts2id = json.load(f)
    return FactDataset(train_facts, valid_facts, test_facts, entity2id, rel2id, ts2id)
