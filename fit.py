import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datasets import load_dataset
from models import *

# negative log-likelihood loss
def loss(parameters, choices, option_A, option_B, agg):
    def negative_log_likelihood(choices, probability_B):
        if agg == 'sum':
            loss = -(choices * np.log(probability_B) + (1 - choices) * np.log(1 - probability_B)).sum()
        elif agg == 'none':
            loss = -(choices * np.log(probability_B) + (1 - choices) * np.log(1 - probability_B))
        return loss

    choice_probability_B = model(parameters, option_A, option_B)
    return negative_log_likelihood(choices, choice_probability_B)

# load data and get test participants
df = pd.read_csv('data/exp1.csv')
test_data = load_dataset("marcelbinz/Psych-101-test")['test']
test_data = test_data.filter(lambda example: example['experiment'] == "hilbig2014generalized/exp1.csv")
test_participants = test_data['participant']
test_participants = [int(a) for a in test_participants]

model_string = model_string_gecco
NUM_PARAMETERS = None
exec(model_string)

nll = np.zeros((len(test_participants), df['trial'].max() + 1))
model_parameter_bounds = [[0, 10], [0, 1]] # TODO implement LLM2Code
for k, participant_id in enumerate(test_participants):
    # prepare data
    df_participant = df[df['participant'] == participant_id]
    choices = df_participant['choice'].values
    option_A = np.stack(df_participant["stimulus_0"].apply(lambda x: np.array(x[1:-1].split(" "), dtype=np.float32)).values)
    option_B = np.stack(df_participant["stimulus_1"].apply(lambda x: np.array(x[1:-1].split(" "), dtype=np.float32)).values)

    # GeCCo model fitting procedure
    x0 = [np.random.uniform(model_parameter_bounds[i][0], model_parameter_bounds[i][1]) for i in range(len(model_parameter_bounds))]
    res = minimize(loss, x0, method='BFGS', args=(choices, option_A, option_B, 'sum'), options={'gtol': 1e-6})
    nll[k] = loss(res.x, choices, option_A, option_B, 'none')

AIC = 2 * NUM_PARAMETERS + 2 * nll.sum(-1)
print(AIC.sum())
