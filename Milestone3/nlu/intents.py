"""The 10 intents: label, what it covers, and the example phrasings the embedding classifier is built from.

These examples are only for building the classifier. The labelled evaluation set
(data/intent_eval_set.csv) is worded differently on purpose and must never be copied in here.
"""

INTENTS = {
    "driver_stats": {
        "description": "A driver's own numbers: points, wins or podiums in a season, career totals, "
                       "or their record / average finish at one circuit.",
        "examples": [
            "How many points did Max Verstappen score in 2023?",
            "How many races did Alonso win in 2005?",
            "Total podiums for Lewis Hamilton in his career",
            "What is Sebastian Vettel's average finishing position at Suzuka?",
            "How many victories did Senna get at Monaco?",
            "Number of podium finishes for Charles Leclerc in 2019",
            "How many points has Lando Norris scored in total?",
            "What was Raikkonen's record at Spa?",
            "How did Kimi do in 2007 in terms of wins and points?",
            "Career wins of Michael Schumacher",
            "Average finish of Fernando Alonso between 2015 and 2018",
            "How many top three finishes did Bottas have in 2019?",
        ],
    },
    "constructor_stats": {
        "description": "A team's numbers: season points or wins, reliability (share of classified "
                       "finishes) or retirements over a period.",
        "examples": [
            "How many points did Ferrari score in 2018?",
            "How reliable were Mercedes from 2014 to 2020?",
            "How many races did Red Bull win in 2013?",
            "What percentage of races did McLaren finish between 2015 and 2017?",
            "How many retirements did Haas have in 2016?",
            "Number of DNFs for Toro Rosso in 2019",
            "Podiums for Williams in 2014",
            "Which share of Renault's entries were classified finishes from 2016 to 2020?",
            "Team points for Aston Martin in 2021",
            "How often did Sauber's cars reach the finish in 2015?",
            "Wins by Brawn in 2009",
            "How many times did Alpine's cars retire in 2022?",
        ],
    },
    "head_to_head": {
        "description": "Two drivers compared in the races they both entered: who finished ahead more "
                       "often, points in shared races, or qualifying head-to-head.",
        "examples": [
            "Hamilton vs Rosberg at Mercedes, who finished ahead more often?",
            "Compare Vettel and Webber in the races they both started",
            "Who beat whom more often, Leclerc or Sainz, at Ferrari?",
            "Head to head between Alonso and Massa",
            "Who outqualified the other more often, Perez or Verstappen?",
            "In shared races, who scored more points, Gasly or Ocon?",
            "Senna versus Prost: who was ahead more times?",
            "Norris against Piastri in 2023, who finished in front more?",
            "Qualifying battle between Russell and Latifi",
            "Bottas or Massa at Williams, who came out on top?",
            "Teammate comparison Button and Hamilton at McLaren",
            "Who had the upper hand in 2016, Hamilton or Rosberg?",
        ],
    },
    "race_result": {
        "description": "The outcome of one Grand Prix: the winner, the podium, the top N finishers, or "
                       "the fastest lap of that race.",
        "examples": [
            "Who won the 2016 Spanish Grand Prix?",
            "Who was on the podium at the 2021 Bahrain GP?",
            "Give me the top 5 of the 2012 Brazilian Grand Prix",
            "Which driver set the fastest lap at the 2020 British Grand Prix?",
            "Winner of the 2008 Singapore race",
            "Who finished second in the 2019 Austrian Grand Prix?",
            "Results of the 2010 Abu Dhabi Grand Prix",
            "Who came third at Monza in 2019?",
            "Podium finishers of the 2022 Saudi Arabian GP",
            "Who took victory at the 2005 Japanese Grand Prix?",
            "Full top ten of the 2023 Las Vegas Grand Prix",
            "Who won in Interlagos in 2008?",
        ],
    },
    "circuit_info": {
        "description": "Facts about a circuit: where it is, how many races it has held and when, "
                       "and who has won there the most.",
        "examples": [
            "Where is the Hungaroring?",
            "In which country is Interlagos?",
            "How many Grands Prix has Silverstone hosted?",
            "Which driver has the most wins at Monza?",
            "When was the last race held at Istanbul Park?",
            "When did Formula 1 first race at Spa?",
            "What city is Albert Park in?",
            "How many times has Imola held a Grand Prix?",
            "Who is the most successful driver at the Circuit Gilles Villeneuve?",
            "List the years in which Zandvoort hosted a race",
            "Location of the Yas Marina circuit",
            "Which team has won the most races at Suzuka?",
        ],
    },
    "qualifying": {
        "description": "Qualifying and the starting grid: pole positions, Q3 appearances, the pole "
                       "sitter of a race, or places gained from grid to finish.",
        "examples": [
            "How many pole positions did Hamilton take in 2019?",
            "Who was on pole at the 2021 Monaco Grand Prix?",
            "How many times did Norris reach Q3 in 2022?",
            "Who made up the most positions from the grid at the 2022 Hungarian GP?",
            "Which driver started first at the 2018 German Grand Prix?",
            "Number of Q3 appearances for Albon in 2020",
            "How many places did Perez gain from his starting position at the 2020 Sakhir GP?",
            "Poles for Leclerc in 2022",
            "Who qualified fastest for the 2017 Italian Grand Prix?",
            "Biggest climb from grid to finish at Silverstone in 2021",
            "What was Verstappen's qualifying position at the 2023 Japanese GP?",
            "How often did Vettel start from pole in 2011?",
        ],
    },
    "pit_stops": {
        "description": "Pit stops: the fastest stops in a race, how many stops drivers made, or a "
                       "driver's stops across a season.",
        "examples": [
            "What was the quickest pit stop at the 2021 Bahrain Grand Prix?",
            "How many pit stops did each driver make at the 2022 Monaco GP?",
            "Average number of stops per driver in the 2019 British Grand Prix",
            "How many times did Hamilton pit during 2020?",
            "Which team had the fastest stop at the 2023 Brazilian race?",
            "Slowest pit stop of the 2018 Australian Grand Prix",
            "How many stops did Verstappen make in Abu Dhabi 2021?",
            "Total pit stops made by Ricciardo in 2016",
            "Which race in 2022 had the most pit stops?",
            "Shortest time spent in the pit lane at the 2020 Turkish Grand Prix",
            "Did Sainz stop twice at the 2023 Singapore GP?",
            "Pit stop durations at the 2019 German Grand Prix",
        ],
    },
    "season_history": {
        "description": "A whole season or the change between seasons: the points leader or champion, "
                       "how many drivers raced, team changes, or the constructors' points order.",
        "examples": [
            "Who scored the most points in 2016?",
            "Who was the champion in 2008?",
            "How many drivers took part in the 2019 season?",
            "Which drivers moved to a different team from 2018 to 2019?",
            "Rank the teams by points in 2020",
            "Who led the championship in 2010?",
            "List the drivers who switched teams between 2021 and 2022",
            "How many different drivers raced in 1985?",
            "Which constructor scored the most points in 2014?",
            "Final points table of the 2012 season",
            "Who topped the drivers' standings in 2003?",
            "Team changes going into the 2024 season",
        ],
    },
    "out_of_scope": {
        "description": "Anything the race database cannot answer (salaries, lap-by-lap timing, "
                       "sprints, weather, tyres, news, opinions) or anything not about Formula 1.",
        "examples": [
            "How much does Lewis Hamilton earn per year?",
            "What tyre compound did Verstappen use in Abu Dhabi?",
            "Who won the 2023 sprint race in Baku?",
            "What will the weather be at Silverstone?",
            "Who won the Champions League in 2020?",
            "What is the capital of Australia?",
            "Write me a poem about racing",
            "Is Ferrari a good company to invest in?",
            "What was Alonso's lap time on lap 23 in Monaco 2012?",
            "Who is Charles Leclerc dating?",
            "How do I make pasta carbonara?",
            "Which F1 driver is the most handsome?",
        ],
    },
    "predict_podium": {
        "description": "A prediction: will a driver finish in the top three at a given race "
                       "(sends the user to the prediction view).",
        "examples": [
            "Predict Leclerc at the 2024 Monaco Grand Prix",
            "Will Hamilton finish on the podium at Silverstone 2024?",
            "What are Norris's chances of a podium in Miami 2024?",
            "Podium probability for Verstappen at the 2024 Japanese GP",
            "Can Sainz get a top three at the 2024 Australian Grand Prix?",
            "Run the podium model for Piastri at the 2024 Italian GP",
            "How likely is Russell to make the podium in Canada 2024?",
            "Forecast Perez's podium odds for the 2024 Saudi Arabian Grand Prix",
            "Would Alonso reach the podium at the 2023 Bahrain GP starting from 5th?",
            "Predict whether Ocon finishes top 3 at Monaco 2024",
            "Is Gasly expected to be on the podium in Brazil 2024?",
            "Model prediction for Hamilton starting 2nd at the 2024 British Grand Prix",
        ],
    },
}

LABELS = list(INTENTS)
