//1. Query 1
MATCH (res:Result)-[:FOR_RACE]->(:Race)-[:IN_SEASON]->(:Season {year: 2021})
MATCH (res)-[:BY_DRIVER]->(d:Driver)
RETURN d.forename + ' ' + d.surname AS Driver,
       sum(res.points) AS Points,
       sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS Wins
ORDER BY Points DESC, Wins DESC