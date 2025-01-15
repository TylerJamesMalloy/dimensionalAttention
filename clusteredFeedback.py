import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import tqdm

import numpy as np 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from Environments import multiAttribute
from pyibl import Agent 
from Models import FRLAgent 
from sklearn.feature_selection import mutual_info_regression


def get_choices():
    num_choices = 3
    attributes = ['Attribute 0', 'Attribute 1', 'Attribute 2']
    values = ['Value 0', "Value 1", 'Value 2']

    choices = [{} for _ in range(num_choices)]
    for attribute in attributes :
        np.random.shuffle(values)
        for choice_idx in range(num_choices):
            choices[choice_idx][attribute] = values[choice_idx]
    return choices

def Train(args):
    columns = ["Timestep", "Correct", "Model"]
    df = pd.DataFrame([], columns=columns)
    delayedResponses = []
    choices = []
    reward_sum = 0
    window = 10
    attributes = ['Attribute 0', 'Attribute 1', 'Attribute 2']
    values = ['Value 0', "Value 1", 'Value 2']
    weights = np.ones(3)
    alpha = 0.75
    env = multiAttribute(args)
    num_agents = 200
    num_resets = 2
    num_timesteps = 50
    num_models = 4
    pbar = tqdm.tqdm(total=num_models*num_timesteps*num_resets*num_agents)
    for model in ["IBL", "WIBL", "FRL", "WFRL"]:
        for _ in range(num_agents):
            memory = []
            if(model == "IBL" or model == "WIBL"):
                a = Agent(name="name", attributes=['Attribute 0', 'Attribute 1', 'Attribute 2'], mismatch_penalty=1, default_utility=0.5)
            if(model == "FRL" or model == "WFRL"):
                a = FRLAgent(args=args, env=env)
            for reset in  [0,1]:
                for ts in range(num_timesteps):
                    pbar.update(1)
                    choices = get_choices()
                    choice = a.choose(choices)
                    if(isinstance(choice, tuple)):
                        choice = choice[0]
                    if(reset):
                        if(choice['Attribute 0'] == 'Value 1'):
                            reward = np.random.choice([0,1], 1, p=[0.25, 0.75])[0] #+ np.random.normal(0,.1)
                            correct = 1
                        else:
                            reward = np.random.choice([0,1], 1, p=[0.75, 0.25])[0] #+ np.random.normal(0,.1)
                            correct = 0
                    else:
                        if(choice['Attribute 0'] == 'Value 1'):
                            reward = np.random.choice([0,1], 1, p=[0.25, 0.75])[0] #+ np.random.normal(0,.1)
                            correct = 1
                        else:
                            reward = np.random.choice([0,1], 1, p=[0.75, 0.25])[0] #+ np.random.normal(0,.1)
                            correct = 0
                    reward_sum += reward
                    if(model == "IBL" or model == "WIBL"):
                        delayedResponses.append(a.respond())

                    if(model == "WFRL"):
                        a.updateWeights()
                    if(model == "WIBL"):
                        memVals = [values.index(list(choice.values())[attr_idx]) for attr_idx in range(3)]
                        memVals.append(reward)
                        memory.append(memVals)

                        if(len(memory) > 5):
                            data = np.array(memory)
                            #last_twenty_slice = slice(-20, None)
                            #sample = data[last_twenty_slice]
                            sample = data
                            
                            X = sample[:, :3]  # First three columns
                            y = sample[:, -1]    # Last column
                            
                            mi = np.array(mutual_info_regression(X, y, discrete_features=True))
                            weights = weights + (alpha * (mi - weights))
                            for attribute, weight in zip(attributes, weights):
                                weight = np.clip(weight, 0, 1)
                                a.similarity([attribute], weight=weight*100)
                    

                    d = pd.DataFrame([[ts + (reset * num_timesteps), correct, model]], columns=columns)
                    df = pd.concat([d, df], ignore_index=True)

                    if(ts % window == 0):
                        if(model == "WIBL"):
                            for delayedResponse in delayedResponses:
                                delayedResponse.update(reward_sum/window)
                        elif(model == "IBL"):
                            for delayedResponse in delayedResponses:
                                delayedResponse.update(reward_sum/window)
                        else:
                            a.respond(reward_sum)
                        delayedResponses = []
                        reward_sum = 0
    pbar.close()
    df.to_pickle("./Results/Clustered_Intra.pkl")
    df['Timestep'] = round(df['Timestep'] / 50) * 50
    sns.lineplot(df, x="Timestep", y="Correct", hue="Model")
    plt.show()

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    parser.add_argument("--env", type=str, default="multiAttribute", help="Environment to load by name")
    parser.add_argument("--agent", type=str, default="IBLAgent", help="Agent to load by name")
    parser.add_argument("--timesteps", type=int, default=50, help="Timesteps to train the agent.")
    parser.add_argument("--weight-updating", type=bool, default=True, help="Whether or not to use weight updating")
    parser.add_argument("-nd", "--num-dims", dest="nd", type=int, default=3, help="Attribute dimensions")
    parser.add_argument("-nc", "--num-choices",dest="nc", type=int, default=3, help="Number of choices")
    parser.add_argument("-na", "--num-attributes", dest="na", type=int, default=3, help="Number of attributes")
    parser.add_argument("-id", "--intra-dimensional",dest="id", type=bool, default=True, help="Intradimensional shift")
    parser.add_argument("-ed", "--extra-dimensional",dest="ed", type=bool, default=False, help="Extradimensional shift")
    parser.add_argument("-cf", "--counter-factual", dest="cf", type=bool, default=False, help="Counterfactual feedback condition")
    parser.add_argument("-df", "--delayed-feedback", dest="df", type=bool, default=False, help="Delayed feedback condition")
    parser.add_argument("--default-utility", type=float, default=0.5, help="Default value")
    parser.add_argument("--name", type=str, default="IBL+W", help="Name of agent to display")
    parser.add_argument("--saveFolder", type=str, default="./Results/", help="Name of agent to display")
    parser.add_argument("--plot", type=bool, default=True, help="Whether to plot results or not")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--seed", dest="seed", type=int, default=1, help="Random seed")
    parser.add_argument("--decay", dest="decay", type=float, default=0.99, help="FRL model decay")
    parser.add_argument("--risky", dest="risky", type=bool, default=False, help="Whether to use risky rewards")
    parser.add_argument("--softTemp", dest="softMaxInverseTemp", type=float, default=0.01, help="FRL model softmax inverse temperature")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=0.9, help="FRL model learning rate")
    parser.add_argument("--feedback", dest="feedback", type=str, default="Clustered", help="FRL model decay")
                        
    args = parser.parse_args()

    Train(args)