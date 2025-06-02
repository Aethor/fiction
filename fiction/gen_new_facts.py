from typing import Literal, Optional, Tuple, List, Dict, TypeVar
from datetime import date, timedelta
import pathlib as pl
import json, random, re, argparse
import numpy as np
from joblib import Parallel, delayed
from fiction.tlogic.apply import apply_rules
from fiction.tlogic.grapher import Grapher
import fiction.tlogic.rule_application as ra
from fiction.tlogic.score_functions import score_12
from fiction.tlogic.temporal_walk import store_edges
from fiction.yagottl.TurtleUtils import YagoDBInfo
from fiction.yagottl.schema import is_rel_allowed, is_obj_allowed
from fiction.utils import load_facts

# (subj, rel, obj, ts)
Fact = Tuple[str, str, str, str]

# (subj, rel, ?, ts)
Query = Tuple[str, str, Literal["?"], str]

# [(obj, score), ...]
QueryOutput = List[Tuple[str, float]]


def load_rules(path: pl.Path, rule_lengths: List[int]) -> Dict[int, dict]:
    with open(path) as f:
        rules_dict = json.load(f)
    rules_dict = {int(k): v for k, v in rules_dict.items()}
    rules_dict = ra.filter_rules(
        rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
    )
    return rules_dict


def make_grapher(
    queries: List[Query],
    train_facts: List[Fact],
    entity2id: Dict[str, int],
    rel2id: Dict[str, int],
    ts2id: Dict[str, int],
) -> Grapher:
    grapher = Grapher.__new__(Grapher)
    grapher.dataset_dir = None
    grapher.entity2id = entity2id
    grapher.relation2id = rel2id.copy()
    counter = len(rel2id)
    for relation in rel2id:
        grapher.relation2id["_" + relation] = counter  # Inverse relation
        counter += 1
    grapher.ts2id = ts2id
    grapher.id2entity = dict([(v, k) for k, v in grapher.entity2id.items()])
    grapher.id2relation = dict([(v, k) for k, v in grapher.relation2id.items()])
    grapher.id2ts = dict([(v, k) for k, v in grapher.ts2id.items()])

    grapher.inv_relation_id = dict()
    num_relations = len(rel2id)
    for i in range(num_relations):
        grapher.inv_relation_id[i] = i + num_relations
    for i in range(num_relations, num_relations * 2):
        grapher.inv_relation_id[i] = i % num_relations

    grapher.train_idx = grapher.add_inverses(grapher.map_to_idx(train_facts))
    grapher.test_idx = grapher.add_inverses(grapher.map_to_idx(queries))
    grapher.all_idx = np.vstack((grapher.train_idx, grapher.test_idx))
    return grapher


def query(
    queries: List[Query],
    train_facts: List[Fact],
    rules: Dict[int, dict],
    entity2id: Dict[str, int],
    rel2id: Dict[str, int],
    ts2id: Dict[str, int],
    process_nb: int = 1,
) -> List[QueryOutput]:
    if len(queries) == 0:
        return []

    entity2id["?"] = -1

    max_ts_id = max(ts2id.values())
    updated_ts2id = ts2id.copy()
    i = 0
    for ts in set(q[3] for q in queries):
        if not ts in updated_ts2id:
            i += 1
            updated_ts2id[ts] = max_ts_id + i

    grapher = make_grapher(queries, train_facts, entity2id, rel2id, updated_ts2id)

    id2entity = {v: k for k, v in entity2id.items()}
    if process_nb == 1:
        window = 0
        scores, _ = apply_rules(
            grapher.test_idx,
            rules,
            grapher,
            # This argument is only used in ra.get_window_edges. In
            # that function, it is discarded unless window ==
            # -1. Therefore, we optimize by not computing it here if
            # not needed.
            store_edges(grapher.train_idx) if window == -1 else None,
            score_12,
            20,  # top_k
            0,
            len(grapher.test_idx),
            # (lambda, a) for score_12
            # a * confidence + (1 - a) temporal_distance(lambda)
            # where temporal_distance is e^{lambda * (max_walk_ts - query_ts)}
            [[0.1, 0.5]],
            window,
        )
        answers = [[] for _ in range(len(queries))]
        for answer_i, scores in scores[0].items():
            try:
                answers[answer_i] = [(id2entity[k], v) for k, v in scores.items()]
            except IndexError:
                continue
        return answers
    else:
        window = 0
        queries_nb = len(grapher.test_idx) // process_nb
        poutput = Parallel(n_jobs=process_nb)(
            delayed(apply_rules)(
                grapher.test_idx,
                rules,
                grapher,
                # This argument is only used in ra.get_window_edges. In
                # that function, it is discarded unless window ==
                # -1. Therefore, we optimize by not computing it here if
                # not needed.
                store_edges(grapher.train_idx) if window == -1 else None,
                score_12,
                20,  # top_k
                i,
                queries_nb,
                [[0.1, 0.5]],  # args for score_12
                window,  # window
            )
            for i in range(process_nb)
        )

        answers = [[] for _ in range(len(queries))]
        for i in range(process_nb):
            for answer_i, scores in poutput[i][0][0].items():
                try:
                    answers[answer_i] = [(id2entity[k], v) for k, v in scores.items()]
                except IndexError:
                    continue

    return answers


T = TypeVar("T")


def maybe_max(lst: List[T], **kwargs) -> Optional[T]:
    if len(lst) == 0:
        return None
    return max(lst, **kwargs)


def rel_is_active(rel: str, entity_facts: List[Fact]) -> bool:
    latest_start = maybe_max([f[3] for f in entity_facts if f[1] == rel])
    if latest_start is None:
        return False
    endRel = "end" + rel[5:]
    latest_end = maybe_max([f[3] for f in entity_facts if f[1] == endRel])
    if latest_end is None:
        return False
    return date.fromisoformat(latest_start) > date.fromisoformat(latest_end)


def unlinearize_rel(rel: str) -> str:
    """REL is originally of the form:

      prefix:name

    however, after linearization, it is of the form:

      prefix:(start|end)Name

    however, the loaded facts/schema/taxonomy is unaware of this, so
    we fix the issue.

    """
    if m := re.match(r"([a-zA-Z]+):(start|end)([a-zA-Z]+)", rel):
        prefix = m.group(1)
        name = m.group(3)
        rel = f"{prefix}:{name[0].lower() + name[1:]}"
        return rel
    return rel


def is_fact_valid(fact: Fact, db_info: YagoDBInfo) -> bool:
    subj, rel, obj, _ = fact
    rel = unlinearize_rel(rel)
    out = is_rel_allowed(subj, rel, db_info) and is_obj_allowed(obj, rel, db_info)
    return out


def sample_new_fact(
    subj: str,
    subj_facts: List[Fact],
    ts: str,
    rules: dict,
    train_facts: List[Fact],
    entity2id: Dict[str, int],
    rel2id: Dict[str, int],
    ts2id: Dict[str, int],
    db_info: YagoDBInfo,
    process_nb: int = 1,
) -> Optional[Fact]:
    rel_candidates: List[str] = []
    for rel in rel2id:
        # NOTE: we pre-validate the (subj, rel) pair to optimize the
        # number of queries
        if not is_rel_allowed(subj, unlinearize_rel(rel), db_info):
            continue
        if rel.startswith("start"):
            if not rel_is_active(rel, subj_facts):
                rel_candidates.append(rel)
        elif rel.startswith("end"):
            if rel_is_active(rel, subj_facts):
                rel_candidates.append(rel)
        else:
            rel_candidates.append(rel)
    # [[(object, score), ...], ... x len(future_rels_candidate)]
    answers = query(
        [(subj, rel, "?", ts) for rel in rel_candidates],
        train_facts,
        rules,
        entity2id,
        rel2id,
        ts2id,
        process_nb=process_nb,
    )
    # transform each (obj, score) couple into fact
    obj_candidates = [
        [(subj, rel, obj, ts) for obj, _ in candidates]
        for candidates, rel in zip(answers, rel_candidates)
    ]
    # filter and keep only valid facts
    obj_candidates = [
        [fact for fact in candidates if is_fact_valid(fact, db_info)]
        for candidates in obj_candidates
    ]
    obj_candidates = [c for c in obj_candidates if len(c) > 0]
    if len(obj_candidates) == 0 or all(len(ans) == 0 for ans in answers):
        return None
    return random.choice(obj_candidates)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rules", type=pl.Path)
    parser.add_argument("-l", "--rule-lengths", nargs="*", type=int)
    parser.add_argument("-d", "--dataset-dir", type=pl.Path)
    parser.add_argument("-y", "--yago-dir", type=pl.Path)
    parser.add_argument("-o", "--output-file", type=pl.Path)
    parser.add_argument("-e", "--year", type=int)
    parser.add_argument("-n", "--facts-per-day", type=int)
    parser.add_argument("-p", "--process-nb", type=int, default=1)
    args = parser.parse_args()

    rules = load_rules(args.rules, args.rule_lengths)
    train_facts = (
        load_facts(args.dataset_dir / "train.txt")
        + load_facts(args.dataset_dir / "valid.txt")
        + load_facts(args.dataset_dir / "test.txt")
    )
    with open(args.dataset_dir / "entity2id.json") as f:
        entity2id = json.load(f)
    with open(args.dataset_dir / "relation2id.json") as f:
        rel2id = json.load(f)
    with open(args.dataset_dir / "ts2id.json") as f:
        ts2id = json.load(f)

    db_info = YagoDBInfo.from_yago_dir(args.yago_dir)

    subj_entities = list(set([f[0] for f in train_facts]))
    new_facts = []
    d = date(args.year, 1, 1)

    while d.year < args.year + 1:
        ts = d.strftime("%Y-%m-%d")

        for i in range(args.facts_per_day):
            new_fact = None
            print(f"generating a fact for {ts}...", end="")
            tries = 0

            while new_fact is None and tries <= 10:
                entity = random.choice(subj_entities)
                entity_facts = [f for f in train_facts if f[0] == entity]
                new_fact = sample_new_fact(
                    entity,
                    entity_facts,
                    ts,
                    rules,
                    train_facts,
                    entity2id,
                    rel2id,
                    ts2id,
                    db_info,
                    process_nb=args.process_nb,
                )
                tries += 1
                print(f".", end="", flush=True)

            if new_fact is None:
                print(f"I give up.")
            else:
                train_facts.append(new_fact)
                new_facts.append(new_fact)
                if not ts in ts2id:
                    ts2id[ts] = max(ts2id.values()) + 1
                print(new_fact)

        d = d + timedelta(days=1)

    with open(args.output_file, "w") as f:
        for subj, rel, obj, ts in new_facts:
            f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")
