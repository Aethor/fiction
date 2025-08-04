import os, argparse, shutil
import pathlib as pl
from fiction.utils import load_facts, dump_facts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=pl.Path)
    parser.add_argument("-o", "--output-dir", type=pl.Path)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    to_remove = "schema:mainEntityOfPage|rdfs:comment|rdfs:label|yago:demonym|schema:alternateName|schema:numberOfEmployees|yago:elevation|schema:geo|schema:url|schema:sameAs|schema:image|schema:parentTaxon|yago:populationNumber"
    # get a smaller subset of yago-facts
    os.system(
        f'grep -P -v "{to_remove}" {args.input_dir / "yago-facts.ttl"} > {args.output_dir / "yago-facts.ttl"}'
    )
    # get rdf:type for entities from yago and yago-beyond-wikipedia
    os.system(
        f'grep "rdf:type" {args.input_dir / "yago-facts.ttl"} > {args.output_dir / "yago-facts-types.ttl"}'
    )
    os.system(
        f'grep -P -v "{to_remove}" {args.input_dir / "yago-beyond-wikipedia.ttl"} | grep "rdf:type" >> {args.output_dir / "yago-facts-types.ttl"}'
    )

    # manually only keep rdf:type about entities in yago-meta-facts in yago-facts-types
    # HACK: load triplets with a quadruplet loading function. This is OK since there is a
    # fourth element (a dot) that will get mistaken for a timestamp
    facts = load_facts(args.output_dir / "yago-facts-types.ttl")
    entities_to_keep = set()
    with open(args.input_dir / "yago-meta-facts.ntx") as f:
        print("loading meta facts...", end="")
        i = 0
        for line in f:
            if i % 1000000 == 0:
                print(".", end="", flush=True)
            i += 1
            try:
                _, subj, _, obj, _, _, _ = line.split("\t")
            except ValueError:
                continue
            entities_to_keep.add(subj)
            entities_to_keep.add(obj)
    print("done!")
    facts = [f for f in facts if f[0] in entities_to_keep]
    dump_facts(facts, args.output_dir / "yago-facts-types.ttl", "dumping facts-types")
    shutil.copy(
        args.input_dir / "yago-meta-facts.ntx", args.output_dir / "yago-meta-facts.ntx"
    )
    shutil.copy(args.input_dir / "yago-schema.ttl", args.output_dir / "yago-schema.ttl")
    shutil.copy(
        args.input_dir / "yago-taxonomy.ttl", args.output_dir / "yago-taxonomy.ttl"
    )
