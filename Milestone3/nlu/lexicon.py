"""Name lookup tables built once from the dataset CSVs (the same data as the knowledge graph).

Every table maps a folded phrase (see text.fold) to the list of ids it can mean. A list with
more than one id is an ambiguity that the extractor turns into a clarifying question.
"""
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from .text import normalize

DATA_DIR = os.environ.get(
    "F1_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))


def _rows(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# Constructor names that are also ordinary English words: indexing them would turn
# "in March" or "the era" into a team. These old teams can still be named in full.
CONSTRUCTOR_STOPWORDS = {"march", "life", "spirit", "shadow", "eagle", "token", "wolf", "era", "ram",
                         "rae", "fry", "hall", "re", "bar", "lec", "turner", "adams", "moore", "elder",
                         "stevens", "sutton", "watson", "martini", "hill", "amon", "boro", "ensign"}

# Driver surnames that are ordinary words in a question ("which driver ...", "in May"): these
# drivers are matched only by their full name.
SURNAME_STOPWORDS = {"driver", "may", "miles", "cross", "green", "love", "rose", "sharp", "ward",
                     "hall", "bell", "will", "long", "start", "power"}

# Short names and nicknames people use for teams -> the constructor name in constructors.csv.
CONSTRUCTOR_ALIASES = {
    "red bull racing": "Red Bull", "redbull": "Red Bull", "rbr": "Red Bull",
    "mercedes amg": "Mercedes", "mercedes-amg": "Mercedes", "merc": "Mercedes", "amg": "Mercedes",
    "scuderia ferrari": "Ferrari",
    "aston": "Aston Martin", "aston martin aramco": "Aston Martin",
    "alpine": "Alpine F1 Team", "alpine f1": "Alpine F1 Team",
    "haas": "Haas F1 Team", "haas f1": "Haas F1 Team",
    "alfa": "Alfa Romeo", "alfa romeo racing": "Alfa Romeo",
    "alpha tauri": "AlphaTauri", "scuderia alphatauri": "AlphaTauri",
    "toro rosso": "Toro Rosso", "str": "Toro Rosso", "scuderia toro rosso": "Toro Rosso",
    "racing bulls": "RB F1 Team", "vcarb": "RB F1 Team", "visa cash app rb": "RB F1 Team", "rb": "RB F1 Team",
    "kick sauber": "Sauber", "stake": "Sauber", "stake f1": "Sauber",
    "brawn gp": "Brawn", "british american racing": "BAR",
    "mclaren mercedes": "McLaren", "williams racing": "Williams",
    "renault f1": "Renault", "manor": "Manor Marussia",
}

# Circuit names people use that are not in circuits.csv, or that the data makes ambiguous
# although everyone means one track (Barcelona -> Catalunya, not the 1950s street circuits).
CIRCUIT_ALIASES = {
    "monaco": 6, "monte carlo": 6, "barcelona": 4, "catalunya": 4, "paul ricard": 34, "le castellet": 34,
    "hockenheim": 10, "nurburgring": 20, "portimao": 75, "algarve": 75, "mugello": 76, "istanbul": 5,
    "sepang": 2, "kuala lumpur": 2, "yas marina": 24, "abu dhabi": 24, "marina bay": 15, "losail": 78,
    "lusail": 78, "spa": 13, "francorchamps": 13, "spa-francorchamps": 13, "interlagos": 18,
    "albert park": 1, "melbourne": 1, "red bull ring": 70, "spielberg": 70, "magny cours": 8,
    "magny-cours": 8, "kyalami": 30, "estoril": 27, "jerez": 26, "adelaide": 29, "watkins glen": 46,
    "brands hatch": 38, "donington": 31, "cota": 69, "circuit of the americas": 69, "austin": 69,
    "gilles villeneuve": 7, "montreal": 7, "suzuka": 22, "imola": 21, "zandvoort": 39, "baku": 73,
    "jeddah": 77, "miami": 79, "sakhir": 3, "shanghai": 17, "silverstone": 9, "monza": 14,
    "hungaroring": 11, "budapest": 11, "sochi": 71, "singapore": 15, "mexico city": 32,
    "hermanos rodriguez": 32, "sao paulo": 18,
}


@dataclass
class Lexicon:
    drivers: dict = field(default_factory=dict)          # id -> {"name", "first", "last"}
    constructors: dict = field(default_factory=dict)     # id -> {"name", "first", "last"}
    circuits: dict = field(default_factory=dict)         # id -> {"name", "location", "country"}
    races: dict = field(default_factory=dict)            # (stem, year) -> race row
    race_by_circuit_year: dict = field(default_factory=dict)
    race_stems: set = field(default_factory=set)
    driver_index: dict = field(default_factory=lambda: defaultdict(set))
    driver_full_index: dict = field(default_factory=lambda: defaultdict(set))
    driver_codes: dict = field(default_factory=dict)     # "HAM" -> {id}
    constructor_index: dict = field(default_factory=lambda: defaultdict(set))
    circuit_index: dict = field(default_factory=lambda: defaultdict(set))
    driver_surnames: set = field(default_factory=set)
    min_year: int = 1950
    max_year: int = 2024


def race_stem(name: str) -> str:
    """'British Grand Prix' -> 'british'; 'Indianapolis 500' stays as it is."""
    n = normalize(name)
    return n[:-len(" grand prix")] if n.endswith(" grand prix") else n


@lru_cache(maxsize=1)
def load() -> Lexicon:
    lx = Lexicon()

    races = _rows("races.csv")
    results = _rows("results.csv")
    race_year = {r["raceId"]: int(r["year"]) for r in races}
    lx.min_year = min(race_year.values())
    lx.max_year = max(race_year.values())

    # first and last season of every driver / constructor, to describe options in a clarifying question
    d_years, c_years = defaultdict(list), defaultdict(list)
    for r in results:
        y = race_year[r["raceId"]]
        d_years[int(r["driverId"])].append(y)
        c_years[int(r["constructorId"])].append(y)

    for r in _rows("drivers.csv"):
        did = int(r["driverId"])
        ys = d_years.get(did, [0])
        name = f'{r["forename"]} {r["surname"]}'
        lx.drivers[did] = {"name": name, "first": min(ys), "last": max(ys)}
        full, sur = normalize(name), normalize(r["surname"])
        lx.driver_full_index[full].add(did)
        if sur not in SURNAME_STOPWORDS:
            lx.driver_index[sur].add(did)
        lx.driver_surnames.add(sur)
        ref = normalize(r["driverRef"].replace("_", " "))
        if ref not in (full, sur):
            lx.driver_full_index[ref].add(did)          # e.g. "max verstappen" from driverRef
        if r["code"] and r["code"] != r"\N":
            lx.driver_codes.setdefault(r["code"].upper(), set()).add(did)

    for r in _rows("constructors.csv"):
        cid = int(r["constructorId"])
        ys = c_years.get(cid, [0])
        name = r["name"].replace("&amp;", "&")
        lx.constructors[cid] = {"name": name, "first": min(ys), "last": max(ys)}
        key = normalize(name)
        if key not in CONSTRUCTOR_STOPWORDS:
            lx.constructor_index[key].add(cid)
        ref = normalize(r["constructorRef"].replace("_", " "))
        if ref != key and ref not in CONSTRUCTOR_STOPWORDS and ref not in lx.driver_surnames:
            lx.constructor_index[ref].add(cid)
    by_name = {v["name"]: k for k, v in lx.constructors.items()}
    for alias, name in CONSTRUCTOR_ALIASES.items():
        lx.constructor_index[normalize(alias)] = {by_name[name]}
    # "Lotus" alone means three different teams: keep it ambiguous on purpose
    lx.constructor_index["lotus"] = {k for k, v in lx.constructors.items()
                                     if v["name"] in ("Team Lotus", "Lotus", "Lotus F1")}

    circuits = _rows("circuits.csv")
    by_country = defaultdict(set)
    for r in circuits:
        cid = int(r["circuitId"])
        lx.circuits[cid] = {"name": r["name"], "location": r["location"], "country": r["country"]}
        lx.circuit_index[normalize(r["name"])].add(cid)
        lx.circuit_index[normalize(r["location"])].add(cid)
        ref = normalize(r["circuitRef"].replace("_", " "))
        if ref not in lx.driver_surnames:
            lx.circuit_index[ref].add(cid)
        by_country[normalize(r["country"])].add(cid)
    for country, ids in by_country.items():
        if len(ids) == 1:                                 # a country with one circuit names that circuit
            lx.circuit_index[country] |= ids
    for alias, cid in CIRCUIT_ALIASES.items():
        lx.circuit_index[normalize(alias)] = {cid}

    for r in races:
        stem = race_stem(r["name"])
        row = {"race_id": int(r["raceId"]), "name": r["name"], "year": int(r["year"]),
               "round": int(r["round"]), "date": r["date"], "circuit_id": int(r["circuitId"])}
        lx.races[(stem, row["year"])] = row
        lx.race_by_circuit_year.setdefault((row["circuit_id"], row["year"]), []).append(row)
        lx.race_stems.add(stem)
    return lx
