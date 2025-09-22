from typing import Set, Tuple
import math
from datetime import datetime
from fiction.yagottl.TurtleUtils import YagoDBInfo

# (subj, rel, obj, ts)
Fact = Tuple[str, str, str, str]


def class_parents(cls: str, db_info: YagoDBInfo) -> Set[str]:
    """Get the direct parents of CLS"""
    return db_info.taxonomy.index.get(cls, {}).get("rdfs:subClassOf", set())


def class_superclasses(cls: str, db_info: YagoDBInfo) -> Set[str]:
    """Get all the superclasses of CLS"""
    parents = class_parents(cls, db_info)
    superclasses = set(parents)
    for parent in parents:
        superclasses |= class_superclasses(parent, db_info)
    return superclasses


def entity_types(entity: str, db_info: YagoDBInfo) -> Set[str]:
    """Return the direct types of ENTITY

    .. note::

        NOTE: this is a safe way of getting the type, that does not crash
        if the type is unknown because a fact is missing (in that case, it
        will return the empty set)
    """
    return db_info.types.index.get(entity, {}).get("rdf:type", set())


def allowed_rels(subj: str, db_info: YagoDBInfo) -> Set[str]:
    """Return the set of allowed relations for SUBJ"""
    allowed_relations = set()
    types = set()
    for typ in entity_types(subj, db_info):
        types |= class_superclasses(typ, db_info)
    for typ in types:
        for prop in db_info.schema.index.get(typ, {}).get("sh:property", set()):
            for prop_path in db_info.schema.index[prop]["sh:path"]:
                allowed_relations.add(prop_path)
    return allowed_relations


def is_rel_allowed(subj: str, rel: str, db_info: YagoDBInfo) -> bool:
    """Check whether SUBJ is allowed to have relation REL"""
    return rel in allowed_rels(subj, db_info)


def allowed_obj_types(rel: str, db_info: YagoDBInfo) -> Set[str]:
    """Return the set of object types allowed for relation REL"""
    # get all properties corresponding to rel
    assert not db_info.schema.inverseGraph is None
    properties = db_info.schema.inverseGraph.index[rel].get("sh:path", set())
    # for each property, get the allowed class
    allowed_classes = set()
    for prop in properties:
        allowed_classes |= db_info.schema.index[prop].get("sh:class", set())
    return allowed_classes


def is_subclass(cls: str, superclass: str, db_info: YagoDBInfo) -> bool:
    """Check whether CLS is a subclass of SUPERCLASS"""
    return superclass in class_superclasses(cls, db_info)


def is_obj_allowed(obj: str, rel: str, db_info: YagoDBInfo) -> bool:
    """Check whether OBJ is allowed for relation REL"""
    allowed_types = allowed_obj_types(rel, db_info)
    for obj_type in entity_types(obj, db_info):
        if any(
            is_subclass(obj_type, allowed_type, db_info)
            for allowed_type in allowed_types
        ):
            return True
    return False


def _ent_dist(ent1: str, ent2: str, db_info: YagoDBInfo) -> float:
    # special case: this should not be needed, but its possible that
    # the type of entities are not present in db_info
    if ent1 == ent2:
        return 0

    level = 0
    # NOTE: we get a fresh copy to avoid modifying db_info
    ent1_types = set(entity_types(ent1, db_info))
    ent2_types = set(entity_types(ent2, db_info))

    while True:
        # there is a common (super?)type between entities
        if len(ent1_types.intersection(ent2_types)) > 0:
            return 1 - 1 / (level + 1)

        prev_ent1_types = set(ent1_types)
        for typ in prev_ent1_types:
            ent1_types |= class_parents(typ, db_info)
        prev_ent2_types = set(ent2_types)
        for typ in prev_ent2_types:
            ent2_types |= class_parents(typ, db_info)

        # no more parent types, we havent found any common types
        # between entities: the distance is maximal
        if len(ent1_types) == len(prev_ent1_types) and len(ent2_types) == len(
            prev_ent2_types
        ):
            return 1

        level += 1


def _ts_dist(ts1: str, ts2: str, k: float) -> float:
    """1 / (1 + e^{-k (|t_1 - t_0| - 365/2)})"""
    ts1_datetime = datetime.fromisoformat(ts1)
    ts2_datetime = datetime.fromisoformat(ts2)
    day_diff = abs((ts2_datetime - ts1_datetime).days)
    return 1 / (1 + math.exp(-k * (day_diff - 365 / 2)))


def facts_dist(
    fact1: Fact, fact2: Fact, alpha: float, k: float, db_info: YagoDBInfo
) -> float:
    """
    α (d_ent(s_1, s_2) / 2 + d_ent(o_1, o_2) / 2) + (1 - α) (d_time(t_1, t_2))
    """
    subj1, _, obj1, ts1 = fact1
    subj2, _, obj2, ts2 = fact2

    subj_dist = _ent_dist(subj1, subj2, db_info)
    obj_dist = _ent_dist(obj1, obj2, db_info)
    ts_dist = _ts_dist(ts1, ts2, k)
    return alpha * (subj_dist / 2 + obj_dist / 2) + (1 - alpha) * ts_dist
