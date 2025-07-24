from __future__ import annotations
import argparse, re, random, json, subprocess
import pathlib as pl
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import torch
import requests
from transformers import pipeline  # type: ignore
from transformers.pipelines.base import Pipeline
from tqdm import tqdm
import numpy as np
from more_itertools import flatten
from sklearn.cluster import AgglomerativeClustering
from fiction.yagottl.TurtleUtils import YagoDBInfo
from fiction.yagottl.schema import facts_dist
from fiction.utils import dump_json, load_facts
from fiction.yago_rel_desc import YAGO_REL_DESC

# (subj, rel, obj, ts)
Fact = tuple[str, str, str, str]


@dataclass
class GCloudConfig:
    project: str
    location: str
    api_endpoint: str

    @staticmethod
    def from_json(json_str: str) -> GCloudConfig:
        return GCloudConfig(**json.loads(json_str))


def hf_get_pipeline(huggingface_id: str) -> Pipeline:
    pipe = pipeline(
        "text-generation",
        model=huggingface_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    assert not pipe.tokenizer is None
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = "left"
    return pipe


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
    assert hex_unicode.lower().startswith("u")
    return chr(int(hex_unicode[1:], base=16))


def clean_unicode(elt: str) -> str:
    return re.sub(r"_[uU][0-9A-E]{4}", lambda m: parse_hex_unicode(m.group()[1:]), elt)


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


def clean_generic_instance(elt: str) -> str:
    return re.sub(r" ?generic instance", "", elt, flags=re.IGNORECASE)


def clean_fact_generic_instance(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_generic_instance(subj), rel, clean_generic_instance(obj), ts)


def format_fact(fact: Fact) -> Fact:
    fact = clean_fact_prefix(fact)
    fact = clean_fact_unicode(fact)
    fact = clean_fact_wiki_id(fact)
    fact = clean_fact_underscore(fact)
    fact = clean_fact_generic_instance(fact)
    return fact


def group_related_facts(
    facts: list[Fact],
    min_size: int,
    max_size: int,
    db_info: YagoDBInfo,
    alpha: float = 0.9,
    k: float = 0.03,
) -> list[list[Fact]]:
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


def _get_multifact_prompt(fact_group: list[Fact]) -> str:
    prompt_template = """Given the following events represented as quadruplets of the form (subject, relation, object, timestamp):
    {}
    and the following definitions for the relations:
    {}
    Generate a short paragraph describing these events, in the style of a newspaper.
    You can add additional details, but the entirety of the information in the given quadruplets must be preserved.
    Do NOT add any additional information or text: you must only generate the description.
    """
    formatted_facts = [format_fact(fact) for fact in fact_group]
    formatted_facts = [randomize_fact_ts_style(fact) for fact in formatted_facts]

    relations = {rel for _, rel, _, _ in formatted_facts}

    current_date = None
    if random.random() < 0.25:
        dates = sorted(
            [datetime.strptime(ts, "%Y-%m-%d") for _, _, _, ts in fact_group]
        )
        min_date = dates[0] - timedelta(days=random.randint(0, 7))
        max_date = dates[0] + timedelta(days=random.randint(0, 7))
        delta = max_date - min_date
        current_date = min_date + timedelta(random.randint(0, delta.days))
        current_date = randomize_ts_style(current_date)

    if not current_date is None:
        prompt_template += f"The current date is {current_date}. In addition to the date of the event, indicate the current date at the top of your text as part of a news headline."

    return prompt_template.format(
        "\n".join(str(fact) for fact in formatted_facts),
        "\n".join(f"{rel}: {YAGO_REL_DESC.get(rel)}" for rel in relations),
    )


def hf_gen_multifacts_description(
    fact_groups: list[list[Fact]], pipe: Pipeline, batch_size: int = 8
) -> list[str]:
    messages = [
        [
            {
                "role": "system",
                "content": "You are a generation model that is expert at outputting description of events.",
            },
            {
                "role": "user",
                "content": _get_multifact_prompt(fact_group),
            },
        ]
        for fact_group in fact_groups
    ]

    descriptions = []
    for batch_start in tqdm(range(0, len(messages), batch_size)):
        batch_end = batch_start + batch_size
        batch_messages = messages[batch_start:batch_end]
        batch_descriptions = ["" for _ in batch_messages]
        for facts in fact_groups[batch_start:batch_end]:
            assert (
                len({datetime.strptime(fact[3], "%Y-%m-%d").year for fact in facts})
                == 1
            )
        batch_years = [
            str(datetime.strptime(facts[0][3], "%Y-%m-%d").year)
            for facts in fact_groups[batch_start:batch_end]
        ]
        has_years = False
        while not has_years:
            # only perform generation for description that don't have
            # fact year yet
            batch_indices = [
                i
                for i in range(len(batch_messages))
                if not batch_years[i] in batch_descriptions[i]
            ]
            outputs = pipe(
                [batch_messages[i] for i in batch_indices],
                max_new_tokens=256,
                pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore
                batch_size=len(batch_indices),
            )
            for i, output in enumerate(outputs):  # type: ignore
                desc = output[0]["generated_text"][-1]["content"]  # type: ignore
                batch_descriptions[batch_indices[i]] = desc  # type: ignore
            has_years = all(
                year in desc for year, desc in zip(batch_years, batch_descriptions)
            )
        descriptions += batch_descriptions

    return descriptions


def hf_gen_multifact_description(fact_group: list[Fact], pipe: Pipeline) -> str:
    return hf_gen_multifacts_description([fact_group], pipe)[0]


def vertexai_gen_multifacts_description(
    fact_groups: list[list[Fact]], model_id: str, config: GCloudConfig
) -> list[Optional[str]]:
    url = f"https://{config.api_endpoint}/v1/projects/{config.project}/locations/{config.location}/endpoints/openapi/chat/completions"

    descriptions = []
    usage_stats = []

    for fact_group in tqdm(fact_groups):
        access_token = (
            subprocess.check_output(["gcloud", "auth", "print-access-token"])
            .decode("utf8")
            .strip("\n")
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model_id,
            "stream": False,
            "messages": [
                {"role": "user", "content": _get_multifact_prompt(fact_group)}
            ],
        }

        try:
            response = requests.post(url, headers=headers, json=data)
        except Exception as e:
            tqdm.write(
                f"warning: could not generate a description for {fact_group}. (reason: {e})"
            )
            descriptions.append(None)
            continue
        if response.status_code != 200:
            tqdm.write(
                f"warning: could not generate a description for {fact_group}. (reason: {response.status_code} {response.json()})"
            )
            descriptions.append(None)
            continue
        response_json = response.json()
        desc = response_json["choices"][0]["message"]["content"]
        descriptions.append(desc)
        usage_stats.append(response_json["usage"])

    print("usage summary:")
    print(
        "completions_tokens: {}".format(
            sum(s["completion_tokens"] for s in usage_stats)
        )
    )
    print("prompt_tokens: {}".format(sum(s["prompt_tokens"] for s in usage_stats)))

    return descriptions


def vertexai_gen_multifact_description(
    fact_group: list[Fact], model_id: str, config: GCloudConfig
) -> Optional[str]:
    return vertexai_gen_multifacts_description([fact_group], model_id, config)[0]


def ts_day_ordinal(day: int) -> str:
    ord_suffix = (
        "th" if 10 <= day <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return str(day) + ord_suffix


def randomize_ts_style(d: datetime) -> str:
    day_style = random.choice(["%B ~d, %Y", "%Y-%m-%d"])
    weekday_style = random.choice(["%A, ", "%a, ", ""])
    style = weekday_style + day_style
    return d.strftime(style).replace("~d", ts_day_ordinal(d.day))


def randomize_fact_ts_style(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    d = datetime.strptime(ts, "%Y-%m-%d")
    new_ts = randomize_ts_style(d)
    return (subj, rel, obj, new_ts)


def _get_fact_prompt(fact: Fact) -> str:
    formatted_fact = format_fact(fact)

    formatted_fact = randomize_fact_ts_style(formatted_fact)
    current_date = None
    if random.random() < 0.25:
        d = datetime.strptime(fact[3], "%Y-%m-%d")
        current_date = d + timedelta(days=random.randint(-7, 7))
        current_date = randomize_ts_style(current_date)

    formatted_relation = formatted_fact[1]

    prompt = f"""Given the following event represented as a quadruplet of the form (subject, relation, object, timestamp):
    {formatted_fact},
    The following definition for the {formatted_relation} relation:
    {YAGO_REL_DESC.get(formatted_relation)},
    Generate a one to three sentences description text for this event, in the style of a newspaper.
    You can add additional details, but the entirety of the information in the given quadruplet must be preserved. 
    Do NOT add any additional information or text: you must only generate the description.
    """
    if not current_date is None:
        prompt += f"The current date is {current_date}. In addition to the date of the event, indicate the current date at the top of your text as part of a news headline."
    return prompt


def hf_gen_facts_description(
    facts: list[Fact], pipe: Pipeline, batch_size: int = 8
) -> list[str]:
    """Given list of quadruples FACTS, generate a description using
    PIPE.

    :param facts: quadruples for which to generate a description
    :param pipe: huggingface text-generation pipeline
    """
    messages = [
        [
            {
                "role": "system",
                "content": "You are a generation model that is expert at outputting description of events.",
            },
            {"role": "user", "content": _get_fact_prompt(fact)},
        ]
        for fact in facts
    ]

    descriptions = []
    for batch_start in tqdm(range(0, len(messages), batch_size)):
        batch_end = batch_start + batch_size
        batch_messages = messages[batch_start:batch_end]
        batch_descriptions = ["" for _ in batch_messages]
        batch_years = [
            str(datetime.strptime(ts, "%Y-%m-%d").year)
            for _, _, _, ts in facts[batch_start:batch_end]
        ]
        has_years = False
        while not has_years:
            # only perform generation for description that don't have
            # fact year yet
            batch_indices = [
                i
                for i in range(len(batch_messages))
                if not batch_years[i] in batch_descriptions[i]
            ]
            outputs = pipe(
                [batch_messages[i] for i in batch_indices],
                max_new_tokens=256,
                pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore
                batch_size=len(batch_indices),
            )
            for i, output in enumerate(outputs):  # type: ignore
                desc = output[0]["generated_text"][-1]["content"]  # type: ignore
                batch_descriptions[batch_indices[i]] = desc  # type: ignore
            has_years = all(
                year in desc for year, desc in zip(batch_years, batch_descriptions)
            )
        descriptions += batch_descriptions

    return descriptions


def hf_gen_fact_description(fact: Fact, pipe: Pipeline) -> str:
    """Given the quadruples FACT, generate a description using LM.

    :param fact: quadruple for which to generate a description
    :param pipe: huggingface text-generation pipeline
    """
    return hf_gen_facts_description([fact], pipe)[0]


def vertexai_gen_facts_description(
    facts: list[Fact], model_id: str, config: GCloudConfig
) -> list[Optional[str]]:
    url = f"https://{config.api_endpoint}/v1/projects/{config.project}/locations/{config.location}/endpoints/openapi/chat/completions"

    descriptions = []
    usage_stats = []

    for fact in tqdm(facts):
        access_token = (
            subprocess.check_output(["gcloud", "auth", "print-access-token"])
            .decode("utf8")
            .strip("\n")
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model_id,
            "stream": False,
            "messages": [{"role": "user", "content": _get_fact_prompt(fact)}],
        }

        try:
            response = requests.post(url, headers=headers, json=data)
        except Exception as e:
            tqdm.write(
                f"warning: could not generate a description for {fact}. (reason: {e})"
            )
            descriptions.append(None)
            continue
        if response.status_code != 200:
            tqdm.write(
                f"warning: could not generate a description for {fact}. (reason: {response.status_code} {response.json()})"
            )
            descriptions.append(None)
            continue
        response_json = response.json()
        desc = response_json["choices"][0]["message"]["content"]
        descriptions.append(desc)
        usage_stats.append(response_json["usage"])

    print("usage summary:")
    print(
        "completions_tokens: {}".format(
            sum(s["completion_tokens"] for s in usage_stats)
        )
    )
    print("prompt_tokens: {}".format(sum(s["prompt_tokens"] for s in usage_stats)))

    return descriptions


def vertexai_gen_fact_description(
    fact: Fact, model_id: str, config: GCloudConfig
) -> Optional[str]:
    return vertexai_gen_facts_description([fact], model_id, config)[0]


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
        default="hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace ID of the language model used to generate descriptions, prefixed by 'hf:' (example: 'hf:meta-llama/Meta-Llama-3.1-8B-Instruct'). Alternatively, the ID of a Google Vertex AI model, prefixed by 'vertexai:' (example: 'vertexai:meta/llama-3.3-70b-instruct-maas'). In that case, you must also set --gcloud-config.",
    )
    parser.add_argument(
        "-g",
        "--gcloud-config",
        type=str,
        default="{}",
        help='google cloud config, as a json dictionary. The following keys must be present: "model_id", "project", "location", "api_endpoint". Example: {"project": "your_project_id", "location": "us-central1", "api_endpoint": "us-central1-aiplatform.googleapis.com"}. Note that the access token will be dynamically obtained with $(gcloud auth print-access-token), so make sure you configured your gcloud CLI accordingly.',
    )
    args = parser.parse_args()

    facts = load_facts(args.facts_file, "loading facts")

    lm_provider, lm = args.language_model.split(":")

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
        if lm_provider == "hf":
            pipe = hf_get_pipeline(lm)
            descs = hf_gen_multifacts_description(fact_groups, pipe)
        elif lm_provider == "vertexai":
            gconfig = GCloudConfig.from_json(args.gcloud_config)
            descs = vertexai_gen_multifacts_description(fact_groups, lm, gconfig)
        else:
            raise ValueError(f"Unknown LLM provider: {lm_provider}.")
        for fact_group, desc in zip(fact_groups, descs):
            if desc is None:
                desc = ["None", "None", "None", "None"]
            dataset.append(
                {
                    "facts": [
                        {
                            "subject": fact[0],
                            "relation": fact[1],
                            "object": fact[2],
                            "timestamp": fact[3],
                        }
                        for fact in fact_group
                    ],
                    "description": desc,
                }
            )
    else:
        if lm_provider == "hf":
            pipe = hf_get_pipeline(lm)
            descs = hf_gen_facts_description(facts, pipe)
        elif lm_provider == "vertexai":
            gconfig = GCloudConfig.from_json(args.gcloud_config)
            descs = vertexai_gen_facts_description(facts, lm, gconfig)
        else:
            raise ValueError(f"Unknown LLM provider: {lm_provider}.")
        for fact, desc in zip(facts, descs):
            if desc is None:
                desc = ["None", "None", "None", "None"]
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
