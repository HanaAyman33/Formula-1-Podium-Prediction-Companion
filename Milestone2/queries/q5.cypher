//5. Query 5
MATCH (res:Result)-[:FOR_RACE]->(r:Race)
WHERE r.year IN [2020, 2021]
MATCH (res)-[:BY_DRIVER]->(d:Driver)
MATCH (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WITH d, r.year AS Season, c, count(res) AS Entries
ORDER BY Entries DESC, c.constructorId ASC
WITH d, Season, collect(c)[0] AS PrimaryConstructor
WITH d,
     collect(CASE WHEN Season = 2020 THEN PrimaryConstructor END)[0] AS c2020,
     collect(CASE WHEN Season = 2021 THEN PrimaryConstructor END)[0] AS c2021
WHERE c2020 IS NOT NULL AND c2021 IS NOT NULL
  AND c2020.constructorId <> c2021.constructorId
WITH d.forename + ' ' + d.surname AS Driver
ORDER BY Driver ASC
RETURN count(Driver) AS Drivers_Changed, collect(Driver) AS Drivers
