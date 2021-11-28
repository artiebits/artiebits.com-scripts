# Code used in my blog post https://artiebits.com/blog/improving-poisson-model-using-time-weighting

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime


def weights_dc(dates, xi=0.0019):
    date_diffs = datetime.now() - dates
    date_diffs = date_diffs.dt.days
    return np.exp(-1 * xi * date_diffs)


data = pd.read_csv("data/Premier-League.csv").assign(
    Date=lambda df: pd.to_datetime(df.Date)
)

# For more details about creating model data check this blog post
# https://artiebits.com/blog/predicting-football-results-with-statistical-modelling
model_data = pd.concat(
    [
        pd.DataFrame(
            data={
                "team": data.Home,
                "opponent": data.Away,
                "goals": data.HomeGoals,
                "home": 1,
            }
        ),
        pd.DataFrame(
            data={
                "team": data.Away,
                "opponent": data.Home,
                "goals": data.AwayGoals,
                "home": 0,
            }
        ),
    ]
)

weights = weights_dc(data.Date)
model_weights = pd.concat([weights, weights])

model = smf.glm(
    formula="goals ~ home + team + opponent",
    data=model_data,
    family=sm.families.Poisson(),
    var_weights=model_weights,
).fit()

# fixtures = data = pd.read_csv("data/Premier-League-fixtures.csv").assign(
#     Date=lambda df: pd.to_datetime(df.Date)
# )
# model = create_model(data.Home, data.Away, data.HomeGoals, data.AwayGoals)
# prediction = simulate_match(model, fixtures.Home, fixtures.Away)

# Check this blog post https://artiebits.com/blog/predicting-football-results-with-statistical-modelling
# to learn how to implement `create_model` and `simulate_match` functions.
