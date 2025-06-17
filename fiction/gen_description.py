from typing import Tuple, List, Optional
import argparse, re
import pathlib as pl
import torch
from transformers import pipeline  # type: ignore
from transformers.pipelines.base import Pipeline
from tqdm import tqdm
import numpy as np
from more_itertools import flatten
from sklearn.cluster import AgglomerativeClustering
from fiction.yagottl.TurtleUtils import YagoDBInfo
from fiction.yagottl.schema import facts_dist
from fiction.utils import dump_json, load_facts

# (subj, rel, obj, ts)
Fact = Tuple[str, str, str, str]


def string_lstrip(s: str, to_strip: str) -> str:
    try:
        s = s[s.index(to_strip) + len(to_strip) :]
    except ValueError:
        pass
    return s


def clean_prefix(elt: str) -> str:
    elt = string_lstrip(elt, "yago:")
    elt = string_lstrip(elt, "schema:")
    return elt


def clean_fact_prefix(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_prefix(subj), clean_prefix(rel), clean_prefix(obj), clean_prefix(ts))


def parse_hex_unicode(hex_unicode: str) -> str:
    assert hex_unicode.startswith("u")
    return chr(int(hex_unicode[1:], base=16))


def clean_unicode(elt: str) -> str:
    return re.sub(r"_u[0-9A-E]{4}", lambda m: parse_hex_unicode(m.group()[1:]), elt)


def clean_fact_unicode(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_unicode(subj), rel, clean_unicode(obj), ts)


def clean_underscore(elt: str) -> str:
    elt = re.sub(r"_$", "", elt)
    elt = re.sub(r"_+", " ", elt)
    return elt


def clean_fact_underscore(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_underscore(subj), rel, clean_underscore(obj), ts)


def clean_wiki_id(elt: str) -> str:
    return re.sub(r"Q[0-9]+", "", elt)


def clean_fact_wiki_id(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_wiki_id(subj), rel, clean_wiki_id(obj), ts)


def format_fact(fact: Fact) -> Fact:
    fact = clean_fact_prefix(fact)
    fact = clean_fact_unicode(fact)
    fact = clean_fact_wiki_id(fact)
    fact = clean_fact_underscore(fact)
    return fact


def group_related_facts(
    facts: List[Fact],
    min_size: int,
    max_size: int,
    db_info: YagoDBInfo,
    alpha: float = 0.9,
    k: float = 0.03,
) -> List[List[Fact]]:
    """Group related facts, returning a list of groups of such facts"""
    dists = np.zeros((len(facts), len(facts)))
    for i in tqdm(range(len(facts)), desc="dist"):
        for j in range(i):
            dist = facts_dist(facts[i], facts[j], alpha, k, db_info)
            dists[i][j] = dist
            dists[j][i] = dist

    clustering = AgglomerativeClustering(
        metric="precomputed", linkage="average", distance_threshold=0.5, n_clusters=None
    ).fit(dists)
    clusters_nb = len(set(clustering.labels_))
    clusters = [[[]] for _ in range(clusters_nb)]
    for fact, label in zip(facts, clustering.labels_):
        if len(clusters[label][-1]) < max_size:
            clusters[label][-1].append(fact)
        else:
            # max size of this cluster has been reached: create a new
            # one
            clusters[label].append([])
    # flatten nested clusters, filter for min_size
    return [c for c in flatten(clusters) if len(c) >= min_size]


YAGO_REL_DESC: Optional[dict[str, str]] = None


def get_relation_desc(rel: str) -> Optional[str]:
    global YAGO_REL_DESC

    if YAGO_REL_DESC is None:
        YAGO_REL_DESC = {}
        with open("./yago_rel_desc.csv") as f:
            for line in f:
                try:
                    rel, desc = line.rstrip("\n").split(",")
                    YAGO_REL_DESC[rel] = desc
                except ValueError:
                    continue

    return YAGO_REL_DESC.get(rel)


def gen_multifacts_description(
    fact_groups: List[List[Fact]], pipe: Pipeline, batch_size: int = 4
) -> List[str]:
    prompt = """Given the following events represented as quadruplets of the form (subject, relation, object, timestamp):
    {}
    and the following definitions for the relations:
    {}
    Generate a short paragraph describing these events, in the style of a newspaper.
    You can add additional details, but the entirety of the information in the given quadruplets must be preserved. 
    Do NOT add any additional information or text: you must only generate the description.
    """

    messages = [
        [
            {
                "role": "system",
                "content": "You are a generation model that is expert at outputting description of events.",
            },
            {
                "role": "user",
                "content": prompt.format(
                    "\n".join(str(format_fact(fact)) for fact in fact_group),
                    "\n".join(
                        f"{fact[1]}: {get_relation_desc(fact[1])}"
                        for fact in fact_group
                    ),
                ),
            },
        ]
        for fact_group in fact_groups
    ]

    descriptions = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i : i + batch_size]
        outputs = pipe(
            batch,
            max_new_tokens=256,
            pad_token_id=pipe.tokenizer.eos_token_id,
            batch_size=batch_size,
        )
        descriptions += [out[0]["generated_text"][-1]["content"] for out in outputs]

    return descriptions


def gen_multifact_description(fact_group: List[Fact], pipe: Pipeline) -> str:
    return gen_multifacts_description([fact_group], pipe)[0]


def gen_facts_description(
    facts: List[Fact], pipe: Pipeline, batch_size: int = 4
) -> List[str]:
    """Given list of quadruples FACTS, generate a description using
    PIPE.

    :param facts: quadruples for which to generate a description
    :param pipe: huggingface text-generation pipeline
    """
    prompt = """Given the following event represented as a quadruplet of the form (subject, relation, object, timestamp):
    {}
    and the following definition for the relation:
    {}
    Generate a one to three sentences description text for this event, in the style of a newspaper.
    You can add additional details, but the entirety of the information in the given quadruplet must be preserved. 
    Do NOT add any additional information or text: you must only generate the description.
    """

    messages = [
        [
            {
                "role": "system",
                "content": "You are a generation model that is expert at outputting description of events.",
            },
            {
                "role": "user",
                "content": prompt.format(format_fact(fact), get_relation_desc(fact[1])),
            },
        ]
        for fact in facts
    ]

    descriptions = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i : i + batch_size]
        outputs = pipe(
            batch,
            max_new_tokens=256,
            pad_token_id=pipe.tokenizer.eos_token_id,
            batch_size=batch_size,
        )
        descriptions += [out[0]["generated_text"][-1]["content"] for out in outputs]

    return descriptions


def gen_fact_description(fact: Fact, pipe: Pipeline) -> str:
    """Given the quadruples FACT, generate a description using LM.

    :param fact: quadruple for which to generate a description
    :param pipe: huggingface text-generation pipeline
    """
    return gen_facts_description([fact], pipe)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="If a --multi-* argument is specified, all --multi-* arguments must be specified."
    )
    parser.add_argument(
        "-f",
        "--facts-file",
        type=pl.Path,
        help="file containing facts, one fact per line.",
    )
    parser.add_argument(
        "-mn",
        "--multi-min-size",
        type=int,
        default=None,
        help="Min size for multi-facts generation.",
    )
    parser.add_argument(
        "-mx",
        "--multi-max-size",
        type=int,
        default=None,
        help="Max size for multi-facts generation.",
    )
    parser.add_argument(
        "-my",
        "--multi-yago-dir",
        type=pl.Path,
        default=None,
        help="Yago directory for multi-facts generation.",
    )
    parser.add_argument(
        "-ma",
        "--multi-alpha",
        type=float,
        default=None,
        help="alpha in fact similarity computation for multi-facts generation.",
    )
    parser.add_argument(
        "-mk",
        "--multi-k",
        type=float,
        default=None,
        help="k for similarity computation in multi-facts generation.",
    )
    parser.add_argument("-o", "--output-file", type=pl.Path, help="output JSON file.")
    parser.add_argument(
        "-l",
        "--language-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    args = parser.parse_args()

    facts = load_facts(args.facts_file, "loading facts")

    pipe = pipeline(
        "text-generation",
        model=args.language_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    assert not pipe.tokenizer is None
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = "left"

    dataset = []
    if args.multi_min_size:  # all --multi arguments should be specified
        db_info = YagoDBInfo.from_yago_dir(args.multi_yago_dir)
        fact_groups = group_related_facts(
            facts,
            args.multi_min_size,
            args.multi_max_size,
            db_info,
            alpha=args.multi_alpha,
            k=args.multi_k,
        )
        descs = gen_multifacts_description(fact_groups, pipe)
        for fact_group, desc in zip(fact_groups, descs):
            dataset.append(
                {
                    "facts": [
                        {
                            "subject": fact[0],
                            "relation": fact[1],
                            "object": fact[2],
                            "timestamp": fact[3],
                        }
                        for fact in fact_groups
                    ],
                    "description": desc,
                }
            )
    else:
        descs = gen_facts_description(facts, pipe)
        for fact, desc in zip(facts, descs):
            dataset.append(
                {
                    "subject": fact[0],
                    "relation": fact[1],
                    "object": fact[2],
                    "timestamp": fact[3],
                    "description": desc,
                }
            )
    dump_json(dataset, args.output_file, "dumping dataset")
