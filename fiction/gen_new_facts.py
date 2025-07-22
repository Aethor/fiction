from typing import Literal, Optional, TypeVar
from datetime import date, timedelta, datetime
from collections import Counter, defaultdict
import pathlib as pl
import json, random, re, argparse
import numpy as np
from more_itertools import flatten
from tqdm import tqdm
from joblib import Parallel, delayed
from fiction.tlogic.apply import apply_rules
from fiction.tlogic.grapher import Grapher
import fiction.tlogic.rule_application as ra
from fiction.tlogic.score_functions import score_12
from fiction.tlogic.temporal_walk import store_edges
from fiction.yagottl.TurtleUtils import YagoDBInfo
from fiction.yagottl.schema import is_rel_allowed, is_obj_allowed
from fiction.utils import FactDataset, load_fact_dataset, load_facts

# (subj, rel, obj, ts)
Fact = tuple[str, str, str, str]

# (subj, rel, ?, ts)
Query = tuple[str, str, Literal["?"], str]

# [(obj, score), ...]
QueryOutput = list[tuple[str, float]]


def load_rules(
    path: pl.Path,
    rule_lengths: list[int],
    min_conf: float = 0.0,
    min_body_supp: int = 1,
) -> dict[int, dict]:
    with open(path) as f:
        rules_dict = json.load(f)
    rules_dict = {int(k): v for k, v in rules_dict.items()}
    rules_dict = ra.filter_rules(
        rules_dict,
        min_conf=min_conf,
        min_body_supp=min_body_supp,
        rule_lengths=rule_lengths,
    )
    return rules_dict


def make_grapher(
    queries: list[Query],
    fact_dataset: FactDataset,
    updated_ts2id: dict[str, int],
    _train_idx: Optional[np.ndarray] = None,
) -> Grapher:
    grapher = Grapher.__new__(Grapher)
    grapher.dataset_dir = None
    grapher.entity2id = fact_dataset.entity2id.copy()
    grapher.entity2id["?"] = -1
    grapher.relation2id = fact_dataset.rel2id.copy()
    counter = len(fact_dataset.rel2id)
    for relation in fact_dataset.rel2id:
        grapher.relation2id["_" + relation] = counter  # Inverse relation
        counter += 1
    grapher.ts2id = updated_ts2id
    grapher.id2entity = dict([(v, k) for k, v in grapher.entity2id.items()])
    grapher.id2relation = dict([(v, k) for k, v in grapher.relation2id.items()])
    grapher.id2ts = dict([(v, k) for k, v in grapher.ts2id.items()])

    grapher.inv_relation_id = dict()
    num_relations = len(fact_dataset.rel2id)
    for i in range(num_relations):
        grapher.inv_relation_id[i] = i + num_relations
    for i in range(num_relations, num_relations * 2):
        grapher.inv_relation_id[i] = i % num_relations

    if _train_idx is None:
        grapher.train_idx = grapher.add_inverses(
            grapher.map_to_idx(fact_dataset.all_facts())
        )
    else:
        grapher.train_idx = grapher.add_inverses(_train_idx)

    grapher.test_idx = grapher.add_inverses(grapher.map_to_idx(queries))
    grapher.all_idx = np.vstack((grapher.train_idx, grapher.test_idx))
    return grapher


def query_tlogic(
    queries: list[Query],
    rules: dict[int, dict],
    fact_dataset: FactDataset,
    _train_idx: Optional[np.ndarray] = None,
) -> list[QueryOutput]:
    if len(queries) == 0:
        return []

    # Updated ts2id to incorporate query timestamps since they might
    # not be in fact_dataset
    max_ts_id = max(fact_dataset.ts2id.values())
    updated_ts2id = fact_dataset.ts2id.copy()
    i = 0
    for ts in set(q[3] for q in queries):
        if not ts in updated_ts2id:
            i += 1
            updated_ts2id[ts] = max_ts_id + i

    grapher = make_grapher(queries, fact_dataset, updated_ts2id, _train_idx)

    id2entity = {v: k for k, v in fact_dataset.entity2id.items()}
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


T = TypeVar("T")


def maybe_max(lst: list[T], **kwargs) -> Optional[T]:
    if len(lst) == 0:
        return None
    return max(lst, **kwargs)


def rel_is_active(rel: str, entity_facts: list[Fact]) -> bool:
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


def prepare_queries(
    rel: str,
    subj_facts: dict[str, list[Fact]],
    db_info: YagoDBInfo,
    max_queries: int = 4,
) -> list[Query]:
    subjects = list(subj_facts.keys())

    subject_candidates = []
    while len(subject_candidates) < max_queries and len(subjects) > 0:
        # we sample the index here to efficiently remove subj from
        # subjects later.
        subj_i = random.randrange(0, len(subjects))
        subj = subjects[subj_i]

        if is_rel_allowed(subj, unlinearize_rel(rel), db_info):
            if rel.startswith("start"):
                if not rel_is_active(rel, subj_facts[subj]):
                    subject_candidates.append(subj)
            elif rel.startswith("end"):
                if rel_is_active(rel, subj_facts[subj]):
                    subject_candidates.append(subj)
            else:
                subject_candidates.append(subj)

        subjects.pop(subj_i)

    return [(subj, rel, "?", ts) for subj in subject_candidates]


def filter_query_answers(
    answers: list[QueryOutput], queries: list[Query], db_info: YagoDBInfo
) -> Optional[Fact]:
    # transform each (obj, score) couple into (fact, score)
    obj_candidates = [
        [((query[0], query[1], obj, query[3]), score) for obj, score in candidates]
        for candidates, query in zip(answers, queries)
    ]
    # filter and keep only valid facts
    obj_candidates = [
        [(fact, score) for fact, score in candidates if is_fact_valid(fact, db_info)]
        for candidates in obj_candidates
    ]
    obj_candidates = [c for c in obj_candidates if len(c) > 0]
    if len(obj_candidates) == 0 or all(len(ans) == 0 for ans in answers):
        return None
    return max(flatten(obj_candidates), key=lambda fact_and_score: fact_and_score[1])[0]


def sample_new_facts(
    ts: str,
    facts_per_day: int,
    rel_probs: dict[str, float],
    rules: dict,
    fact_dataset: FactDataset,
    db_info: YagoDBInfo,
    max_tries_nb: int,
    parallel: Parallel,
) -> list[Fact]:
    assert max_tries_nb >= 1

    print(f"generating {facts_per_day} facts for {ts}...")

    new_facts = []
    tries_nb = 0
    progress = tqdm(total=facts_per_day, ascii=True)
    while len(new_facts) < facts_per_day or tries_nb > max_tries_nb:
        to_gen_nb = facts_per_day - len(new_facts)

        relations = random.choices(
            list(fact_dataset.rel2id.keys()),
            [rel_probs.get(rel, 0) for rel in fact_dataset.rel2id.keys()],
            k=to_gen_nb,
        )

        # We have to cut processing in 3 steps. Steps 1 and 3 cant be
        # parallelized with joblib since that would require copying
        # db_info, which is way too large.
        # 1. preparation
        subj_facts = fact_dataset.subj_facts()  # { subject => [fact, ...]}
        queries = []
        for rel in relations:
            rel_queries = prepare_queries(rel, subj_facts, db_info)
            queries.append(rel_queries)
            for rel_query in rel_queries:
                subj = rel_query[0]
                # make sure we don't generate two facts for the same
                # subject on the same day - this avoids generating
                # contradictory facts
                del subj_facts[subj]
        # 2. TLogic query
        # OPTIM: we precompute train_idx for make_grapher, see query_tlogic.
        _train_idx = fact_dataset.map_to_idx()
        query_answers = parallel(
            delayed(query_tlogic)(queries[i], rules, fact_dataset, _train_idx)
            for i in range(to_gen_nb)
        )
        for answers, rel_queries in zip(query_answers, queries):
            # 3. filtering: we keep only valid candidates according to
            # db_info
            new_fact = filter_query_answers(answers, rel_queries, db_info)
            if new_fact is None:
                continue

            new_facts.append(new_fact)
            progress.update()
            tqdm.write(str(new_fact))

            # we don't need to generate new facts anymore: cancel
            # remaining workers (will generate a warning the first
            # time)
            if len(new_facts) == facts_per_day:
                break

        del query_answers
        tries_nb += 1

    if tries_nb > max_tries_nb:
        print("I give up.")
    else:
        print("done!")

    return new_facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--rules",
        type=pl.Path,
        help="Rules learned by TLogic to generate new facts.",
    )
    parser.add_argument(
        "-l", "--rule-lengths", nargs="*", type=int, help="TLogic rule lengths."
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=pl.Path,
        help="Starting dataset directory, with {train|valid|test}.txt and {entity2id|relation2id|ts2id}.json.",
    )
    parser.add_argument(
        "-y",
        "--yago-dir",
        type=pl.Path,
        help="YAGO directory. Used to ensure that generated facts respect database scheme.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pl.Path,
        help="File where new facts will be dumped, one per line.",
    )
    parser.add_argument(
        "-s",
        "--restart-from-output",
        action="store_true",
        help="if specified, restart generation from --output-file. Useful to restart a crashed generation process.",
    )
    parser.add_argument(
        "-m",
        "--mimic-year",
        type=int,
        help="year of the past dataset to use as an inspiration when generating the dataset. Number of facts per day will be the same as that year (up to 128 days). Relations will be sampled according to the distribution of relations that year.",
    )
    parser.add_argument(
        "-e", "--year", type=int, help="Year for which to generate new facts."
    )
    parser.add_argument(
        "-p",
        "--process-nb",
        type=int,
        default=1,
        help="Number of process used for generation (note: currently this does not seem to increase performance).",
    )
    args = parser.parse_args()

    rules = load_rules(args.rules, args.rule_lengths, min_conf=0.01, min_body_supp=2)
    fact_dataset = load_fact_dataset(args.dataset_dir)
    db_info = YagoDBInfo.from_yago_dir(args.yago_dir)

    subj_entities = list(fact_dataset.subj_entities())

    if args.restart_from_output:
        new_facts = load_facts(args.output_file, f"loading {args.output_file}")
        last_ts = new_facts[-1][3]
        d = datetime.strptime(last_ts, "%Y-%m-%d").date()
        assert d.year == args.year
        print(f"restarting from {args.output_file} (last known date: {d})")
        d = d + timedelta(days=1)  # start from next day
        # update fact_dataset
        fact_dataset.test_facts += new_facts
        for _, _, _, ts in new_facts:
            if not ts in fact_dataset.ts2id:
                fact_dataset.ts2id[ts] = max(fact_dataset.ts2id.values()) + 1
    else:
        new_facts = []
        d = date(args.year, 1, 1)

    # facts for the year we are trying to mimic. We copy:
    # - the number of facts per day of that year
    # - the relationship distribution for that year
    mimic_year_facts = [
        fact
        for fact in fact_dataset.all_facts()
        if datetime.strptime(fact[3], "%Y-%m-%d").year == args.mimic_year
    ]
    mimic_year_dates = [
        datetime.strptime(ts, "%Y-%m-%d") for _, _, _, ts in mimic_year_facts
    ]
    rel_counter = Counter(rel for _, rel, _, _ in fact_dataset.all_facts())
    max_counter = max(rel_counter.values())
    rel_probs = {rel: counter / max_counter for rel, counter in rel_counter.items()}

    with Parallel(n_jobs=args.process_nb, return_as="generator_unordered") as parallel:
        while d.year < args.year + 1:
            ts = d.strftime("%Y-%m-%d")

            facts_per_day = sum(
                mimic_year_d.day == d.day and mimic_year_d.month == d.month
                for mimic_year_d in mimic_year_dates
            )
            facts_per_day = min(128, facts_per_day)

            local_new_facts = sample_new_facts(
                ts,
                facts_per_day,
                rel_probs,
                rules,
                fact_dataset,
                db_info,
                10,
                parallel,
            )

            new_facts += local_new_facts

            # save on disk to ensure that, even if the program
            # crashes, we have a partial dataset
            with open(args.output_file, "w") as f:
                for subj, rel, obj, ts in new_facts:
                    f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")

            # extend fact_dataset with the new generated facts so that
            # they can be used in new preditions
            fact_dataset.test_facts += local_new_facts
            if not ts in fact_dataset.ts2id:
                fact_dataset.ts2id[ts] = max(fact_dataset.ts2id.values()) + 1

            d = d + timedelta(days=1)
