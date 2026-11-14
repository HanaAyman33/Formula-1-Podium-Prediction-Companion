//2. Query 2
MATCH (res:Result)-[:FOR_RACE]->(r:Race)
MATCH (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
MATCH (res)-[:HAS_STATUS]->(s:Status)
WHERE r.year >= 2014 AND r.year <= 2021
WITH c,
     count(res) AS Entries,
     sum(CASE WHEN s.status = 'Finished' OR s.status STARTS WITH '+' THEN 1 ELSE 0 END) AS Classified
WHERE Entries >= 100
RETURN c.constructor_name AS Constructor, Entries, Classified,
       toFloat(Classified) / Entries AS Reliability
ORDER BY Reliability DESC, Constructor ASC
