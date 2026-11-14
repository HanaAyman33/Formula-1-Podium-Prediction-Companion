//4. Query 4
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(:Circuit {circuitId: 6})
MATCH (res)-[:BY_DRIVER]->(d:Driver)
MATCH (res)-[:HAS_STATUS]->(s:Status)
WHERE r.date < date('2020-01-01')
  AND NOT s.status IN ['Did not qualify', 'Did not prequalify', 'Withdrew']
WITH d,
     count(DISTINCT r) AS Starts,
     avg(res.positionOrder) AS avgPos,
     sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS Wins
WHERE Starts >= 5
WITH d, Starts, round(avgPos, 4) AS Avg_Finish, Wins
RETURN d.forename + ' ' + d.surname AS Driver, Starts, Avg_Finish, Wins
ORDER BY Avg_Finish ASC, Wins DESC, Driver ASC
