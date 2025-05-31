import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datasets import load_dataset
import torch
import re
from unsloth import FastLanguageModel
from transformers import pipeline
from models import *

# TODO careful with this
used_model_id = 2
start_simulation_id = 8

# helper function for printing
def until_nth_occurrence(s, substring, n, num_chars_to_print=72):
    count = 0
    index = 0
    while index < len(s):
        index = s.find(substring, index)
        if index == -1:
            break
        count += 1
        if count == n:
            return s[index + len(substring) -num_chars_to_print:index + len(substring) + 1]

        index += len(substring)

    assert False, "should not happen"

# load data and get test participants
df = pd.read_csv('data/exp1.csv')
test_data = load_dataset("marcelbinz/Psych-101-test")['test']
test_data = test_data.filter(lambda example: example['experiment'] == "hilbig2014generalized/exp1.csv")
test_participants = test_data['participant']
test_participants = [int(a) for a in test_participants]

# negative log-likelihood of Centaur
nll_centaur = torch.stack(torch.load('data/log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth', weights_only=True))[test_participants].float().numpy()

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

# load LLM
llm, tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'unsloth/Qwen3-32B-bnb-4bit',
    max_seq_length = 32768,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(llm)

generator = pipeline('text-generation', model=llm, tokenizer=tokenizer, device_map='auto', return_full_text=False)


for model_id, model_string_init in enumerate([model_string_ew, model_string_wadd, model_string_ttb]):
    if model_id == used_model_id:
        simulation = start_simulation_id
        while simulation < 10:
            nll = np.zeros((len(test_participants), df['trial'].max() + 1))
            NUM_PARAMETERS = None
            model = None
            model_string = model_string_init
            for iteration in range(5):
                # model fitting
                exec(model_string)

                for k, participant_id in enumerate(test_participants):
                    # prepare data
                    df_participant = df[df['participant'] == participant_id]
                    choices = df_participant['choice'].values
                    option_A = np.stack(df_participant["stimulus_0"].apply(lambda x: np.array(x[1:-1].split(" "), dtype=np.float32)).values)
                    option_B = np.stack(df_participant["stimulus_1"].apply(lambda x: np.array(x[1:-1].split(" "), dtype=np.float32)).values)

                    # model fitting
                    x0 = 0.01 * np.random.randn(NUM_PARAMETERS)
                    res = minimize(loss, x0, method='BFGS', args=(choices, option_A, option_B, 'sum'), options={'gtol': 1e-6})
                    nll[k] = loss(res.x, choices, option_A, option_B, 'none')

                AIC = 2 * NUM_PARAMETERS + 2 * nll.sum(-1)
                print(AIC.sum())

                # scientific regret minimization
                nll_delta = nll - nll_centaur
                threshold = 0.05
                #num_prints = 10
                #threshold = -np.sort(-nll_delta.flatten())[num_prints]
                srm_string = ""
                num_srm_points = 0
                for k, participant_id in enumerate(test_participants):
                    # load prompt for participant
                    participant_data = test_data.filter(lambda example: example['participant'] == str(participant_id))

                    # print data points
                    for trial_id in range(nll_delta.shape[1]):
                        substring = until_nth_occurrence(participant_data[0]['text'], "<<", trial_id + 1)
                        if nll_delta[k, trial_id].item() > threshold:
                            num_srm_points += 1
                            product_names = re.findall(r'Product (\w) ratings', substring)
                            substring = substring.replace(f'Product {product_names[0]}', 'Product A', 1)
                            substring = substring.replace(f'Product {product_names[1]}', 'Product B', 1)
                            substring = substring.replace(f'<<{product_names[0]}', '<<A', 1)
                            substring = substring.replace(f'<<{product_names[1]}', '<<B', 1)

                            srm_string += '* ' + substring.replace('You press <<', 'Participant choice: ') + '\n'
                print(num_srm_points)

                prompt = "I am studying human behavior in a multi-attribute decision-making experiment.\n" \
                    "In this experiment, participants encounter a number of trials, in which they have to choose between two options labelled A and B.\n" \
                    "These options are fictitious products that are each characterized by four features.\n" \
                    "Each feature corresponds to a binary rating of an expert, either approving of the product (1) or not (0).\n" \
                    "The four experts are ordered based on their validity (taking values of 90%, 80%, 70%, and 60%), with the first feature corresponding to the ratings from the highest validity expert.\n" \
                    "In each trial, people have to predict which of the shown options is superior in terms of quality based on the presented information.\n\n" \
                    "I have the following computational model that is currently my best guess for how people make decisions in this experiment:\n"
                prompt += model_string
                prompt += '\n\nThis model does capture human behavior reasonably well overall, but there are the following data points in which it does not capture human behavior yet:\n\n'
                prompt += srm_string
                prompt += '\nCan you suggest an improved model that is able to capture human behavior in the listed situations?\n'
                prompt += 'Please structure your answer as follows:\n'
                prompt += '* Keep the structure of the function exactly the same.\n'
                prompt += '* Do not change the docstring.\n'
                prompt += '* State the number of free parameters before the model function using the NUM_PARAMETERS variable.\n'
                prompt += '* Do not write any text besides that and do not elaborate any further.'

                np.savez('data/srm_model_' + str(model_id) + '_run_' + str(simulation) + '_iteration_' + str(iteration) + '.npz', nll=nll, num_parameters=NUM_PARAMETERS, prompt=prompt, model_string=model_string)

                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ]
                model_outputs = generator(messages, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, return_full_text=False, max_new_tokens=16384)

                model_string = model_outputs[0]['generated_text'].split("</think>")[-1]
                print(model_string)

            simulation += 1
