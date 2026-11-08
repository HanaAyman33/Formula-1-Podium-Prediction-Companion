//1. Query 1
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(:Season {year: 2021})
MATCH (res)-[:BY_DRIVER]->(d:Driver)
WITH d,
     sum(coalesce(res.points, 0.0)) AS Points,
     sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS Wins
RETURN d.forename + ' ' + d.surname AS Driver, Points, Wins
ORDER BY Points DESC, Wins DESC, Driver ASC
