"""Entity extraction (Milestone 3, section 1.b).

    ex = extract("How many wins did Hamilton have at Silverstone before 2020?", intent="driver_stats")
    ex.entities      -> {"drivers": [...], "circuit": {...}, "year_to": 2019, ...}
    ex.ambiguous     -> [] or the options to ask about ("Schumacher" -> Michael / Ralf / Mick)
    ex.question()    -> the clarifying question to show, or None

The values in ex.entities are ids from the CSVs (the same ids the graph uses), so they go straight
into the Cypher parameters through to_params(). Nothing is guessed: a name that fits several
drivers, teams or circuits, or a required slot that is missing, becomes a clarifying question.
The session cache (Member 4) merges these entities with the previous turn and then calls
missing_slots() / complete() on the merged result.
"""
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from .lexicon import load, race_stem
from .text import ngrams, normalize, tokens

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10}

# Words of the question itself: never fuzzy-matched to a name ("points" must not become "Pons").
QUESTION_WORDS = set("""
how many much what which who whom whose when where why did does do was were is are has have had the
a an of in at on for to from with and or vs versus against between before after since until than
win wins won winner winners victory victories podium podiums points point score scored finish finished
finishes finishing average career season seasons year years race races raced racing grand prix gp
pole poles position positions qualifying qualified qualify grid start started starts gained gain
lost places place fastest slowest lap laps stop stops pit pits strategy team teams teammate teammates
driver drivers constructor constructors championship champion leader led standings reliable reliability
retire retired retirements finishes circuit circuits track tracks held hosted located location country
city predict prediction chance chances probability likely podium compare comparison ahead more often
better than total number list show tell give me about please would could will should their his her
they them he she it its this that these those there here also again only every each most least best
worst first second third last next previous change changed changes switched moved drove drive driving
times time between during over under across result results classification order top fastest
""".split())

DRIVER_INTENTS = {"driver_stats", "head_to_head", "predict_podium", "qualifying", "pit_stops"}
TEAM_CUES = {"at", "for", "with", "drove", "team", "car", "cars", "teams"}


@dataclass
class Extraction:
    entities: dict
    ambiguous: list = field(default_factory=list)     # [{"slot", "text", "options": [{"id","name","detail"}]}]
    references: list = field(default_factory=list)    # "driver" for he/his, "constructor" for they/the team
    fuzzy: list = field(default_factory=list)         # names matched despite a typo, e.g. "verstapen"
    tokens: list = field(default_factory=list)        # the normalised words of the message
    spans: list = field(default_factory=list)         # (start, end, kind) of every name found

    def masked(self):
        """The message with names and years replaced by their type: 'how many points did driver
        score in year'. The intent is in the wording, not in which driver or year is named."""
        tag = {"race": "grand prix", "driver_full": "driver", "driver_code": "driver",
               "driver_surname": "driver", "constructor": "team", "circuit": "circuit"}
        out, k = [], 0
        starts = {i: (j, kind) for i, j, kind in self.spans}
        while k < len(self.tokens):
            if k in starts:
                j, kind = starts[k]
                out.append(tag[kind])
                k = j
                continue
            tk = self.tokens[k]
            out.append("year" if re.fullmatch(YEAR, tk) else tk)
            k += 1
        return " ".join(out)

    def question(self):
        """The clarifying question for the first ambiguity, or None."""
        if not self.ambiguous:
            return None
        a = self.ambiguous[0]
        opts = [f'{o["name"]} ({o["detail"]})' if o.get("detail") else o["name"] for o in a["options"]]
        what = {"driver": "driver", "constructor": "team", "circuit": "circuit"}.get(a["slot"], a["slot"])
        listing = ", ".join(opts[:-1]) + f" or {opts[-1]}" if len(opts) > 1 else opts[0]
        return f'"{a["text"]}" matches more than one {what}. Did you mean {listing}?'


def empty_entities():
    return {"drivers": [], "constructors": [], "circuit": None, "race": None,
            "season": None, "year_from": None, "year_to": None, "years": [], "top_n": None}


# ---------------------------------------------------------------------------------------------
# Years
# ---------------------------------------------------------------------------------------------
YEAR = r"(19[5-9]\d|20[0-4]\d)"


def _years(text, lx, ents):
    t = normalize(text)
    found = [int(y) for y in re.findall(rf"\b{YEAR}\b", t)]
    ents["years"] = found
    # two-digit end of a range: "2014-21", "2014 to 21"
    m = re.search(rf"\b(?:between|from)?\s*{YEAR}\s*(?:-|to|and|until|through|till)\s*(\d{{2}}|{YEAR[1:-1]})\b", t)
    if m:
        a, b = int(m.group(1)), m.group(2)
        b = int(b) if len(b) == 4 else (a // 100) * 100 + int(b)
        if b > a:
            ents["year_from"], ents["year_to"] = a, b
            return
    m = re.search(rf"\b(?:since|after|from)\s+{YEAR}\b", t)
    if m:
        y = int(m.group(1))
        ents["year_from"] = y + 1 if t[m.start():m.start() + 5] == "after" else y
        ents["year_to"] = lx.max_year
    m = re.search(rf"\b(?:before|prior to|until|up to)\s+{YEAR}\b", t)
    if m:
        y = int(m.group(1))
        # "before 2020" = races dated before 2020-01-01, i.e. seasons up to 2019
        ents["year_to"] = y - 1 if m.group(0).startswith(("before", "prior")) else y
        ents["year_from"] = ents["year_from"] or lx.min_year
    if ents["year_from"] is None and ents["year_to"] is None and found:
        ents["season"] = found[0]


def _top_n(text, ents):
    t = normalize(text)
    m = re.search(r"\btop\s+(\d+|" + "|".join(NUM_WORDS) + r")\b", t)
    if m:
        v = m.group(1)
        ents["top_n"] = int(v) if v.isdigit() else NUM_WORDS[v]


# ---------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------
def _driver_opts(lx, ids):
    return sorted(({"id": i, "name": lx.drivers[i]["name"],
                    "detail": f'{lx.drivers[i]["first"]}–{lx.drivers[i]["last"]}'} for i in ids),
                  key=lambda o: -lx.drivers[o["id"]]["last"])


def _active(lx, did, ents):
    """Is driver did active in the season / range / race year the message names?"""
    d = lx.drivers[did]
    year = ents["season"] or (ents["race"] or {}).get("year")
    if year:
        return d["first"] <= year <= d["last"]
    if ents["year_from"] or ents["year_to"]:
        lo, hi = ents["year_from"] or lx.min_year, ents["year_to"] or lx.max_year
        return d["first"] <= hi and d["last"] >= lo
    return True


def _span_candidates(lx, toks, raw_text):
    """Every (start, end, kind, ids) a phrase of the message can mean, longest phrases first."""
    cands = []
    for i, j, ph in ngrams(toks, 5):
        nxt = toks[j:j + 2]
        if ph in lx.race_stems and (nxt[:1] == ["gp"] or nxt == ["grand", "prix"] or nxt[:1] == ["race"]):
            end = j + (2 if nxt == ["grand", "prix"] else 1)
            cands.append((i, end, "race", {ph}))
        if ph in lx.driver_full_index:
            cands.append((i, j, "driver_full", lx.driver_full_index[ph]))
        if ph in lx.constructor_index:
            cands.append((i, j, "constructor", lx.constructor_index[ph]))
        if ph in lx.circuit_index:
            cands.append((i, j, "circuit", lx.circuit_index[ph]))
        if ph in lx.driver_index:
            cands.append((i, j, "driver_surname", lx.driver_index[ph]))
    # three-letter driver codes, only when written in capitals ("HAM", "VER")
    for code in re.findall(r"\b[A-Z]{3}\b", raw_text):
        if code in lx.driver_codes:
            low = code.lower()
            for k, tk in enumerate(toks):
                if tk == low:
                    cands.append((k, k + 1, "driver_code", lx.driver_codes[code]))
    return cands


def _fuzzy(lx, toks, used):
    """Typo-tolerant match for single unmatched words of 5+ letters (surnames, teams, circuits)."""
    hits = []
    choices = {**{k: "driver_surname" for k in lx.driver_index},
               **{k: "constructor" for k in lx.constructor_index if " " not in k},
               **{k: "circuit" for k in lx.circuit_index if " " not in k}}
    for k, tk in enumerate(toks):
        if k in used or len(tk) < 5 or tk in QUESTION_WORDS or tk.isdigit():
            continue
        best = process.extractOne(tk, choices.keys(), scorer=fuzz.ratio, score_cutoff=85)
        # a typo keeps the first letter and roughly the length; short names are never fuzzy-matched
        # (otherwise "space" would become the driver Pace)
        if best and len(best[0]) >= 5 and best[0][0] == tk[0] and abs(len(best[0]) - len(tk)) <= 2:
            key = best[0]
            kind = choices[key]
            idx = {"driver_surname": lx.driver_index, "constructor": lx.constructor_index,
                   "circuit": lx.circuit_index}[kind]
            hits.append((k, k + 1, kind, idx[key], tk, key))
    return hits


def extract(text: str, intent: str | None = None) -> Extraction:
    lx = load()
    ents = empty_entities()
    ex = Extraction(entities=ents)
    toks = tokens(text)

    _years(text, lx, ents)
    _top_n(text, ents)

    low = f" {normalize(text)} "
    if re.search(r"\b(he|his|him|she|her)\b", low):
        ex.references.append("driver")
    if re.search(r"\b(they|their|the team|that team|this team)\b", low):
        ex.references.append("constructor")

    # pick non-overlapping phrases, longest first; on a tie prefer race > full name > team > circuit > surname
    prio = {"race": 0, "driver_full": 1, "driver_code": 2, "constructor": 3, "circuit": 4, "driver_surname": 5}
    cands = sorted(_span_candidates(lx, toks, text), key=lambda c: (-(c[1] - c[0]), prio[c[2]], c[0]))
    used, chosen = set(), []
    for c in cands:
        span = set(range(c[0], c[1]))
        same = [x for x in chosen if (x[0], x[1]) == (c[0], c[1])]
        if span & used and not same:
            continue
        chosen.append(c)
        used |= span
    for k0, k1, kind, ids, typo, key in _fuzzy(lx, toks, used):
        chosen.append((k0, k1, kind, ids))
        ex.fuzzy.append({"typed": typo, "matched": key})
        used.add(k0)

    # group the readings of the same phrase ("williams" = team or driver surname)
    spans = {}
    for i, j, kind, ids in sorted(chosen, key=lambda c: c[0]):
        spans.setdefault((i, j), []).append((kind, ids))
    ex.tokens = toks
    ex.spans = [(i, j, readings[0][0]) for (i, j), readings in spans.items()]

    races = []
    driver_mentions = []        # (text, ids)
    for (i, j), readings in spans.items():
        phrase = " ".join(toks[i:j])
        kinds = {k for k, _ in readings}
        if "race" in kinds:
            races.append(next(ids for k, ids in readings if k == "race"))
            continue
        if kinds & {"driver_full", "driver_code"}:
            driver_mentions.append((phrase, next(ids for k, ids in readings if k in ("driver_full", "driver_code"))))
            continue
        team = next((ids for k, ids in readings if k == "constructor"), None)
        circ = next((ids for k, ids in readings if k == "circuit"), None)
        sur = next((ids for k, ids in readings if k == "driver_surname"), None)
        if sur and (team or circ):
            # a surname that is also a team/circuit name: decide from the words around it
            before = toks[i - 1] if i > 0 else ""
            after = toks[j] if j < len(toks) else ""
            needed = 2 if intent == "head_to_head" else 1
            others = sum(1 for (a, b), r in spans.items() if (a, b) != (i, j)
                         and any(k.startswith("driver") for k, _ in r))
            as_team = (before in TEAM_CUES or after in {"team", "car"} or intent == "constructor_stats"
                       or not (intent in DRIVER_INTENTS and others < needed))
            if as_team:
                sur = None
            else:
                team = circ = None
        if sur:
            driver_mentions.append((phrase, sur))
        if team:
            ents["_teams"] = ents.get("_teams", []) + [(phrase, team)]
        if circ:
            ents["_circuits"] = ents.get("_circuits", []) + [(phrase, circ)]

    # races named "<stem> grand prix" need a year
    if races:
        stem = next(iter(races[0]))
        year = ents["season"] or (ents["years"][0] if ents["years"] else None)
        row = lx.races.get((stem, year)) if year else None
        if row:
            ents["race"] = dict(row)
        else:
            ents["race"] = {"stem": stem, "name": None, "year": year, "race_id": None}

    # drivers: one entry per mention; ambiguous surnames narrowed by the season, else asked about
    for phrase, ids in driver_mentions:
        ids = set(ids)
        if len(ids) > 1:
            active = {d for d in ids if _active(lx, d, ents)}
            ids = active or ids
        if len(ids) == 1:
            did = next(iter(ids))
            if did not in [d["driver_id"] for d in ents["drivers"]]:
                ents["drivers"].append({"driver_id": did, "name": lx.drivers[did]["name"]})
        else:
            ex.ambiguous.append({"slot": "driver", "text": phrase, "options": _driver_opts(lx, ids)})

    for phrase, ids in ents.pop("_teams", []):
        if len(ids) == 1:
            cid = next(iter(ids))
            if cid not in [c["constructor_id"] for c in ents["constructors"]]:
                ents["constructors"].append({"constructor_id": cid, "name": lx.constructors[cid]["name"]})
        else:
            opts = sorted(({"id": c, "name": lx.constructors[c]["name"],
                            "detail": f'{lx.constructors[c]["first"]}–{lx.constructors[c]["last"]}'}
                           for c in ids), key=lambda o: -lx.constructors[o["id"]]["last"])
            ex.ambiguous.append({"slot": "constructor", "text": phrase, "options": opts})

    for phrase, ids in ents.pop("_circuits", []):
        if ents["circuit"]:
            continue
        if len(ids) == 1:
            cid = next(iter(ids))
            ents["circuit"] = {"circuit_id": cid, "name": lx.circuits[cid]["name"]}
        else:
            ex.ambiguous.append({"slot": "circuit", "text": phrase,
                                 "options": [{"id": c, "name": lx.circuits[c]["name"],
                                              "detail": lx.circuits[c]["location"]} for c in sorted(ids)]})

    complete(ents)
    return ex


# ---------------------------------------------------------------------------------------------
# Completing and checking the slots (also used by the session cache after a merge)
# ---------------------------------------------------------------------------------------------
def complete(ents: dict) -> dict:
    """Fill what can be derived without guessing: a race from its name + year, or from circuit + year."""
    lx = load()
    race = ents.get("race")
    year = ents.get("season")
    if race and race.get("race_id") is None:
        y = race.get("year") or year
        row = lx.races.get((race.get("stem"), y)) if y else None
        if row:
            ents["race"] = dict(row)
        elif y:
            race["year"] = y
    if not ents.get("race") and ents.get("circuit") and year:
        rows = lx.race_by_circuit_year.get((ents["circuit"]["circuit_id"], year), [])
        if len(rows) == 1:
            ents["race"] = dict(rows[0])
    return ents


REQUIRED = {                       # each intent: alternatives, satisfied if one is fully present
    "driver_stats": [["driver"]],
    "constructor_stats": [["constructor"]],
    "head_to_head": [["driver", "driver2"]],
    "race_result": [["race"]],
    "circuit_info": [["circuit"]],
    "qualifying": [["driver"], ["race"]],
    "pit_stops": [["race"], ["driver", "season"]],
    "season_history": [["season"], ["year_range"]],
    "predict_podium": [["driver", "race"]],
    "out_of_scope": [[]],
}

ASK = {
    "driver": "Which driver do you mean?",
    "driver2": "Which driver should I compare them with?",
    "constructor": "Which team do you mean?",
    "circuit": "Which circuit do you mean?",
    "race": "Which race do you mean? Please give the Grand Prix and the year (e.g. the 2021 Italian Grand Prix).",
    "race_year": "Which year of that Grand Prix?",
    "season": "Which season?",
    "year_range": "Which season or range of years?",
}


def _has(ents, slot):
    if slot == "driver":
        return len(ents.get("drivers") or []) >= 1
    if slot == "driver2":
        return len(ents.get("drivers") or []) >= 2
    if slot == "constructor":
        return bool(ents.get("constructors"))
    if slot == "circuit":
        return bool(ents.get("circuit"))
    if slot == "race":
        return bool((ents.get("race") or {}).get("race_id"))
    if slot == "season":
        return ents.get("season") is not None
    if slot == "year_range":
        return ents.get("year_from") is not None or ents.get("year_to") is not None
    return False


def missing_slots(intent: str, ents: dict) -> list:
    """The slots still needed for this intent (the shortest missing list among the alternatives)."""
    best = None
    for alt in REQUIRED.get(intent, [[]]):
        miss = [s for s in alt if not _has(ents, s)]
        if best is None or len(miss) < len(best):
            best = miss
    return best or []


def missing_question(intent: str, ents: dict):
    miss = missing_slots(intent, ents)
    if not miss:
        return None
    slot = miss[0]
    if slot == "race" and (ents.get("race") or {}).get("stem") and not (ents.get("race") or {}).get("year"):
        return ASK["race_year"]
    if slot == "race" and (ents.get("race") or {}).get("stem"):
        return f'There is no {ents["race"]["stem"].title()} Grand Prix in {ents["race"]["year"]} in the data. ' \
               "Which race do you mean?"
    return ASK[slot]


def to_params(ents: dict) -> dict:
    """Flat Cypher parameters for the retrieval layer (Member 2)."""
    d = ents.get("drivers") or []
    c = ents.get("constructors") or []
    race = ents.get("race") or {}
    return {
        "driver_id": d[0]["driver_id"] if d else None,
        "driver2_id": d[1]["driver_id"] if len(d) > 1 else None,
        "constructor_id": c[0]["constructor_id"] if c else None,
        "circuit_id": (ents.get("circuit") or {}).get("circuit_id") or race.get("circuit_id"),
        "race_id": race.get("race_id"),
        "race_date": race.get("date"),
        "season": ents.get("season"),
        "year_from": ents.get("year_from"),
        "year_to": ents.get("year_to"),
        "top_n": ents.get("top_n"),
    }
