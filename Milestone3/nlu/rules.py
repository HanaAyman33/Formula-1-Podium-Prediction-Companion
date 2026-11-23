"""Rule-based intent classifier (Milestone 3, section 1.a), plus the checks both classifiers share.

Each intent has keyword / paraphrase patterns with weights. The entities found in the message add
evidence too: "points" with a team named is constructor_stats, with a driver named it is
driver_stats, with only a season it is season_history.
"""
import re
from dataclasses import dataclass, field

from .entities import extract
from .text import normalize

# --------------------------------------------------------------------------------------------
# Shared checks
# --------------------------------------------------------------------------------------------
F1_TERMS = re.compile(
    r"\b(f1|formula ?(1|one)|grand prix|grands prix|gp|races?|racing|raced|drivers?|teams?|teammates?|"
    r"constructors?|poles?|podiums?|pits?|pitted|qualif\w*|grid|laps?|circuits?|tracks?|championships?|"
    r"champions?|q[123]|dnfs?|retire\w*|reliab\w*)\b")

# Things the race database does not hold: answering them would need knowledge the graph lacks.
UNANSWERABLE = re.compile(
    r"\b(salary|salaries|net worth|how much (money )?(does|did|do) \w+ (earn|make|get paid)|contracts?|"
    r"sponsor\w*|budget|tyres?|tires?|compounds?|weather|rain\w*|temperature|sprints?|lap times|"
    r"lap time on|on lap \d+|lap by lap|lap-by-lap|dating|girlfriend|boyfriend|wife|husband|married|"
    r"handsome|favou?rite|best ever|greatest|tickets?|news|rumou?rs?|horsepower|top speed|invest\w*|"
    r"stocks?|poem|story|joke|recipe|song)\b")

# Other sports and topics that reuse F1 words ("champions", "season", "points"): with no F1 name
# in the message these are out of scope.
NON_F1 = re.compile(
    r"\b(champions league|premier league|la liga|serie a|bundesliga|world cup|fifa|uefa|nba|nfl|nhl|"
    r"super bowl|football|soccer|basketball|tennis|wimbledon|cricket|golf|olympics?|motogp|nascar|"
    r"indycar|formula e|wrc|le mans 24|tour de france|boxing|ufc)\b")

FOLLOW_UP = re.compile(r"^(and|what about|how about|same for|and for|and in|now for|what of|then)\b")


def domain_evidence(text: str, ex=None) -> bool:
    """True when the message is clearly about F1: an F1 term, or a driver / team / circuit / race."""
    ex = ex or extract(text)
    e = ex.entities
    t = normalize(text)
    names = bool(e["drivers"] or e["constructors"] or e["circuit"] or e["race"] or ex.ambiguous)
    if NON_F1.search(t) and not names:
        return False
    if F1_TERMS.search(t) or e["race"] or ex.ambiguous:
        return True
    # "How many points did he score in 2019?": a pronoun plus an F1 question pattern
    if ex.references and _any_intent_pattern(t):
        return True
    # a name counts only if it was typed (near) exactly: a fuzzy match alone is too weak a signal
    exact = len(e["drivers"]) + len(e["constructors"]) + bool(e["circuit"]) - len(ex.fuzzy)
    return exact > 0


def unanswerable(text: str) -> bool:
    return bool(UNANSWERABLE.search(normalize(text)))


def is_follow_up(text: str, ex=None) -> bool:
    """'And in 2020?', 'What about Norris?', '2021?': no intent of its own, so the cached intent is kept."""
    t = normalize(text)
    if FOLLOW_UP.match(t):
        return True
    ex = ex or extract(text)
    e = ex.entities
    words = [w for w in t.split() if w not in {"in", "for", "the", "at", "of", "and", "then", "season"}]
    named = len(e["drivers"]) + len(e["constructors"]) + bool(e["circuit"]) + bool(e["race"]) \
        + len(e["years"]) + len(ex.ambiguous)
    # the whole message is names / years ("Norris?", "2021?", "Mercedes 2019")
    return 0 < len(words) <= 3 and named > 0 and not _any_intent_pattern(t)


# --------------------------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------------------------
STAT = r"\b(points?|wins?|won|victor\w*|podiums?|top (three|3) finish\w*|scored?)\b"

PATTERNS = {
    "predict_podium": [
        (r"\bpredict\w*|\bforecast\w*|\bmodel\b", 3),
        (r"\b(chances?|probab\w*|odds|likel\w*|expected to|going to)\b", 2),
        (r"\b(will|would|can|could)\b.*\b(podium|top (three|3))\b", 3),
    ],
    "pit_stops": [
        (r"\bpit ?(stops?|lane|crew)?\b|\bpits\b|\bpit-stops?\b|\bpitted\b|\bpitting\b|\bboxed\b", 3),
        (r"\bstops?\b|\bstationary\b", 1),
    ],
    "qualifying": [
        (r"\bpoles?\b|\bpole position\w*|\bpole sitter\b|\bqualif\w*|\bq3\b|\bq2\b|\bq1\b", 3),
        (r"\bgrid\b|\bstart(ed|ing)? (from|in|on|position)\b|\bfront row\b", 2),
        (r"\b(places?|positions?) (gained|lost|made up)\b|\b(gained|made up|climbed)\b.*\b(places?|positions?)\b|\bclimb\b", 3),
    ],
    "head_to_head": [
        (r"\bvs\b|\bversus\b|\bagainst\b|\bhead to head\b|\bhead-to-head\b|\bbeat\w*\b|\boutqualif\w*", 3),
        (r"\b(finished|came|was) (ahead|in front)\b|\bahead of\b|\bmore often\b|\bupper hand\b|\bcompar\w*", 2),
        (r"\bteammates?\b|\bbattle\b|\bwho was (better|faster|stronger)\b", 1),
    ],
    "race_result": [
        (r"\bwho won\b|\bwinner\b|\bwon the\b|\btook (the )?(victory|win)\b|\bwho (came|finished) (first|second|third|1st|2nd|3rd)\b", 2),
        (r"\b(podium|top \d+|top (three|five|ten)|results?|classification|finishing order)\b.*\b(gp|grand prix|race)\b", 2),
        (r"\bfastest lap\b", 2),
        (r"\b(came|finished) (first|second|third|1st|2nd|3rd|last)\b|\bon the podium\b", 1),
    ],
    "circuit_info": [
        (r"\bwhere is\b|\blocated\b|\blocation\b|\bwhich (country|city)\b|\bwhat (country|city)\b", 3),
        (r"\b(held|hosted|host)\b|\bhow many (races|grands prix|times)\b.*\b(held|hosted|run|raced)\b", 2),
        (r"\b(first|last) (race|grand prix|time)\b|\bmost (wins|successful|victories)\b|\bwho has won the most\b", 2),
        (r"\bcircuit\b|\btrack\b|\bvenue\b", 1),
    ],
    "constructor_stats": [
        (r"\breliab\w*|\bdnfs?\b|\bretire\w*|\bretirements?\b|\bmechanical\b|\bclassified\b|\breach(ed)? the (finish|flag)\b|\bfinish rate\b", 3),
        (r"\bconstructor\b|\bteam\b", 1),
    ],
    "driver_stats": [
        (r"\baverage (finish\w*|position)\b|\bcareer\b|\brecord at\b|\bhow (well|good) did\b", 2),
    ],
    "season_history": [
        (r"\bchampion(ship)?s?\b|\btitle\b|\bstandings\b|\bpoints (leader|table)\b|\bled the\b|\bleader\b", 3),
        (r"\bhow many (different )?drivers\b|\bdrivers (in|per|raced|took part)\b|\btook part\b", 3),
        (r"\b(changed|switched|moved|swapped) (to a different |to another )?teams?\b|\bteam changes?\b|\bnew teams?\b|\bmoved to\b", 3),
        (r"\b(rank|order|ranking)\b.*\b(teams?|constructors?)\b|\bmost points\b|\bscored the most\b", 2),
        (r"\bseason\b", 1),
    ],
}
_COMPILED = {k: [(re.compile(p), w) for p, w in v] for k, v in PATTERNS.items()}
_STAT = re.compile(STAT)


def _any_intent_pattern(t: str) -> bool:
    return bool(_STAT.search(t)) or any(rx.search(t) for v in _COMPILED.values() for rx, _ in v)


@dataclass
class IntentResult:
    label: str                       # one of the 10 intents, "follow_up" (keep cached intent) or "unclear"
    method: str
    scores: dict = field(default_factory=dict)
    reason: str = ""


def classify_rules(text: str) -> IntentResult:
    t = normalize(text)
    ex = extract(text)
    e = ex.entities
    if is_follow_up(text, ex):
        return IntentResult("follow_up", "rules", reason="no new intent named: keep the cached intent")
    if not domain_evidence(text, ex):
        return IntentResult("out_of_scope", "rules", reason="nothing about Formula 1 in the message")
    if unanswerable(text) and not re.search(r"\bpredict|\bpodium\b", t):
        return IntentResult("out_of_scope", "rules", reason="asks for something the graph does not hold")

    scores = {k: 0.0 for k in PATTERNS}
    for intent, pats in _COMPILED.items():
        for rx, w in pats:
            if rx.search(t):
                scores[intent] += w

    n_drivers = len(e["drivers"]) + sum(1 for a in ex.ambiguous if a["slot"] == "driver")
    has_team = bool(e["constructors"]) or any(a["slot"] == "constructor" for a in ex.ambiguous)
    # "he" / "they" point at the driver / team in the session cache
    if n_drivers == 0 and "driver" in ex.references:
        n_drivers = 1
    if not has_team and "constructor" in ex.references and n_drivers == 0:
        has_team = True
    has_season = e["season"] is not None or e["year_from"] is not None

    # numbers (points / wins / podiums) belong to whoever is named
    if _STAT.search(t):
        if n_drivers >= 2:
            scores["head_to_head"] += 1
        elif n_drivers == 1:
            scores["driver_stats"] += 2
        elif has_team:
            scores["constructor_stats"] += 2
        elif e["race"]:
            scores["race_result"] += 1
        elif e["circuit"]:
            scores["circuit_info"] += 1
        elif has_season:
            scores["season_history"] += 1
    if n_drivers >= 2:
        scores["head_to_head"] += 2
    if has_team and n_drivers == 0:
        scores["constructor_stats"] += 1
    if e["race"] and n_drivers == 0:
        scores["race_result"] += 1
    if e["circuit"] and not e["race"] and n_drivers == 0 and not e["years"]:
        scores["circuit_info"] += 1

    best = max(scores.values())
    if best == 0:
        return IntentResult("unclear", "rules", scores, "no pattern matched: ask the user to rephrase")
    order = list(PATTERNS)                     # ties go to the more specific intent (earlier in the list)
    label = min((k for k, v in scores.items() if v == best), key=order.index)
    return IntentResult(label, "rules", scores, "highest pattern score")
