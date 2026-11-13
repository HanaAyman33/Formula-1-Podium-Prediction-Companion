//3. Query 3
MATCH (h:Result)-[:BY_DRIVER]->(:Driver {driverId: 1})
MATCH (b:Result)-[:BY_DRIVER]->(:Driver {driverId: 822})
MATCH (h)-[:FOR_RACE]->(r:Race)<-[:FOR_RACE]-(b)
MATCH (h)-[:FOR_CONSTRUCTOR]->(c:Constructor)<-[:FOR_CONSTRUCTOR]-(b)
WITH c, r, min(h.positionOrder) AS hPosition, min(b.positionOrder) AS bPosition
RETURN c.constructor_name AS Constructor,
       count(*) AS Shared_Races,
       sum(CASE WHEN hPosition < bPosition THEN 1 ELSE 0 END) AS Hamilton_Ahead,
       sum(CASE WHEN bPosition < hPosition THEN 1 ELSE 0 END) AS Bottas_Ahead
ORDER BY Constructor ASC
