// Member 1 checks: run in Neo4j Browser after Create_kg.py
// 1) node counts: Driver 861, Constructor 212, Circuit 77, Season 75, Status 139, Race 1125
MATCH (n) WHERE n:Driver OR n:Constructor OR n:Circuit OR n:Season OR n:Status OR n:Race
RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label;
// 2) relationships: AT_CIRCUIT 1125, IN_SEASON 1125
MATCH (:Race)-[r]->() RETURN type(r) AS rel, count(*) AS n;
// 3) types: every count below must be 0
MATCH (r:Race) WHERE NOT r.date IS :: DATE OR NOT r.year IS :: INTEGER OR NOT r.round IS :: INTEGER RETURN count(r) AS bad_races;
MATCH (d:Driver) WHERE (d.dob IS NOT NULL AND NOT d.dob IS :: DATE) OR (d.number IS NOT NULL AND NOT d.number IS :: INTEGER) RETURN count(d) AS bad_drivers;
MATCH (c:Circuit) WHERE NOT c.lat IS :: FLOAT OR NOT c.lng IS :: FLOAT OR NOT c.alt IS :: INTEGER RETURN count(c) AS bad_circuits;
MATCH (r:Race)-[:IN_SEASON]->(s:Season) WHERE r.year <> s.year RETURN count(r) AS season_mismatch;
// 4) no \N stored anywhere
MATCH (n) UNWIND keys(n) AS k WITH n, k WHERE n[k] = '\\N' OR n[k] = '' RETURN count(*) AS bad_values;
// 5) run Create_kg.py a second time, then repeat 1) and 2): counts must not change
