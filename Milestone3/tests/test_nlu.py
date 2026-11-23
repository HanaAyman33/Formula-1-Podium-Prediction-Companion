"""Checks for Member 1's part, one per behaviour the milestone asks for.

    python tests/test_nlu.py          (or: pytest tests/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlu.entities import extract, missing_question, missing_slots, to_params   # noqa: E402
from nlu.rules import classify_rules                                            # noqa: E402


def names(ex):
    return [d["name"] for d in ex.entities["drivers"]]


def test_full_name_and_surname():
    assert names(extract("How many podiums did Charles Leclerc get in 2022?")) == ["Charles Leclerc"]
    assert names(extract("How many podiums did Leclerc get in 2022?")) == ["Charles Leclerc"]


def test_ambiguous_surname_asks():
    ex = extract("How many points did Schumacher score?", "driver_stats")
    assert ex.ambiguous and ex.ambiguous[0]["slot"] == "driver"
    assert "Michael Schumacher" in ex.question() and "Mick Schumacher" in ex.question()


def test_season_narrows_an_ambiguous_surname():
    assert names(extract("Schumacher points in 2021", "driver_stats")) == ["Mick Schumacher"]


def test_constructor_aliases():
    for text, team in [("redbull points in 2022", "Red Bull"), ("mercedes amg wins in 2019", "Mercedes"),
                       ("How reliable was Williams between 2014 and 2021?", "Williams")]:
        ex = extract(text, "constructor_stats")
        assert [c["name"] for c in ex.entities["constructors"]] == [team], text


def test_team_name_that_is_also_a_surname():
    ex = extract("Who finished ahead more often, Norris or Ricciardo at McLaren?", "head_to_head")
    assert names(ex) == ["Lando Norris", "Daniel Ricciardo"]
    assert [c["name"] for c in ex.entities["constructors"]] == ["McLaren"]


def test_circuit_by_name_or_city():
    assert extract("Where is Suzuka?").entities["circuit"]["circuit_id"] == 22
    assert extract("races held in Montreal").entities["circuit"]["circuit_id"] == 7
    assert extract("Hamilton at Monaco").entities["circuit"]["circuit_id"] == 6


def test_race_by_name_and_year():
    p = to_params(extract("Who won the 2019 German Grand Prix?", "race_result").entities)
    assert p["race_id"] == 1020 and p["season"] == 2019


def test_race_without_year_asks():
    ex = extract("Who won the British GP?", "race_result")
    assert missing_question("race_result", ex.entities) == "Which year of that Grand Prix?"


def test_race_from_circuit_and_year():
    assert to_params(extract("Who won at Monaco in 2018?", "race_result").entities)["race_id"] == 994


def test_year_ranges():
    e = extract("How reliable was Williams between 2014 and 2021?").entities
    assert (e["year_from"], e["year_to"]) == (2014, 2021)
    e = extract("wins at Silverstone before 2020").entities
    assert e["year_to"] == 2019                       # races dated before 2020-01-01
    e = extract("Perez points 2014-21").entities
    assert (e["year_from"], e["year_to"]) == (2014, 2021)


def test_missing_required_entity_asks():
    e = extract("How many points did he score in 2019?", "driver_stats")
    assert "driver" in e.references
    assert missing_slots("driver_stats", e.entities) == ["driver"]
    e = extract("Compare Norris with his teammate", "head_to_head").entities
    assert missing_question("head_to_head", e) == "Which driver should I compare them with?"


def test_typo_and_no_false_match():
    assert names(extract("How many points did Verstapen score in 2021?")) == ["Max Verstappen"]
    assert names(extract("Recommend a good book about space")) == []          # not the driver "Pace"
    assert names(extract("Which driver gained the most places at Monza in 2021?")) == []  # not "Paddy Driver"


def test_follow_ups_keep_the_cached_intent():
    for text in ["And in 2020?", "What about Norris?", "2021?"]:
        assert classify_rules(text).label == "follow_up", text


def test_out_of_scope():
    for text in ["Who won the 2022 Champions League?", "What is Max Verstappen's salary?",
                 "Who won the sprint at Interlagos in 2021?", "What is the capital of France?"]:
        assert classify_rules(text).label == "out_of_scope", text
    # a pronoun question with an F1 pattern is not out of scope
    assert classify_rules("How many points did he score in 2019?").label == "driver_stats"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__} {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
