# Code used in my blog post https://artiebits.com/blog/predicting-football-results-with-statistical-modelling

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson


def create_model(home_team, away_team, home_goals, away_goals):
    model_data = pd.concat(
        [
            pd.DataFrame(
                data={
                    "team": home_team,
                    "opponent": away_team,
                    "goals": home_goals,
                    "home": 1,
                }
            ),
            pd.DataFrame(
                data={
                    "team": away_team,
                    "opponent": home_team,
                    "goals": away_goals,
                    "home": 0,
                }
            ),
        ]
    )

    return smf.glm(
        formula="goals ~ home + team + opponent",
        data=model_data,
        family=sm.families.Poisson(),
    ).fit()


def simulate_match(model, home_team, away_team, max_goals=8):
    df = pd.DataFrame()

    home_team = home_team.values
    away_team = away_team.values

    for i in range(0, len(home_team)):
        exp_home_goals = model.predict(
            pd.DataFrame(
                data={"team": home_team[i], "opponent": away_team[i], "home": 1},
                index=[1],
            )
        ).values[0]

        exp_away_goals = model.predict(
            pd.DataFrame(
                data={"team": away_team[i], "opponent": home_team[i], "home": 0},
                index=[1],
            )
        ).values[0]

        team_pred = [
            [poisson.pmf(i, team_avg) for i in range(0, max_goals)]
            for team_avg in [exp_home_goals, exp_away_goals]
        ]

        matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

        home_team_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_team_win = np.sum(np.triu(matrix, 1))

        temp_df = pd.DataFrame(
            data={
                "home_team": home_team[i],
                "away_team": away_team[i],
                "home_team_win": home_team_win,
                "draw": draw,
                "away_team_win": away_team_win,
                "exp_home_goals": exp_home_goals,
                "exp_away_goals": exp_away_goals,
            },
            index=[1],
        )

        df = df.append(temp_df).reset_index(drop=True).round(2)

    return df


data = pd.read_csv("data/Premier-League.csv")

fixtures = data.tail(10)
data = data.head(-10)

model = create_model(data.Home, data.Away, data.HomeGoals, data.AwayGoals)

prediction = simulate_match(model, fixtures.Home, fixtures.Away)

print(prediction)
