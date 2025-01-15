import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
import random 

from Models import IBLAgent, FRLAgent
from Environments import multiAttribute

def Train(args):
    cols = ["Name", "Timestep", "Reward", "Correct", "Resets", "IntraDimensional", "ExtraDimensional", "Feedback"]
    df = pd.DataFrame([], columns=cols)
    for agentIdx in range(args.agents):
        args.seed = args.seed + agentIdx # Different seed for each agent but same set of seeds across agent population
        env = multiAttribute(args)
        if(args.model == "FRLAgent"):
            agent = FRLAgent(args, env)
        elif(args.model == "IBLAgent"):
            agent = IBLAgent(args, env)
        else: 
            print("Unrecognized agent type")
            assert(False)
        choices = env.get_choices()
        reward_sum = 0
        for resets in range(2):
            for ts in range(args.timesteps): 
                choice, details = agent.choose(choices)
                next_choices, reward, correct = env.step(choice)
                if(args.feedback == "Immediate"):
                    agent.respond(reward)
                elif(args.feedback == "Clustered"):
                    reward_sum += reward
                    if(ts % 5 == 0):
                        agent.respond(reward_sum)
                        reward_sum = 0
                    else:
                        agent.respond(None)
                elif(args.feedback == "Additional"):
                    reward = agent.respond(reward)
                else:
                    print("Feedback method not recognized")
                    assert(False)

                if(isinstance(reward, list)):
                    assert(False)

                agent.updateWeights()
                choices = next_choices
                timestep = ts + (resets * args.timesteps)
                timestep = int((timestep / 2)) * 2
                d = pd.DataFrame([[args.name, timestep, reward, correct, resets, args.id, args.ed, args.feedback]], columns=cols)
                df = pd.concat([df, d], ignore_index=True)
            if(resets == 0):
                choices = env.reset()

    return df


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

    args = parser.parse_args()

    cols = ["Name", "Timestep", "Reward", "Correct", "Resets", "IntraDimensional", "ExtraDimensional"]
    df = pd.DataFrame([], columns=cols)
    iterations = 0
    #for feedback in ["Additional", "Immediate", "Clustered"]:
    for feedback in ["Clustered"]:
        args.feedback = feedback
        #for weight_updating, name, model in zip([True, True, False, False], ["WIBL", "WFRL", "IBL", "FRL"], ["IBLAgent", "FRLAgent", "IBLAgent", "FRLAgent"]):
        for weight_updating, name, model in zip([True], ["WIBL"], ["IBLAgent"]):
            for shift, id, ed in zip(["IntraDimensional", "ExtraDimensional"], [False, True], [True, False]):
                iterations += 1
                print("Iteration: ", iterations)
                args.id = id
                args.ed = ed 
                args.name = name
                args.model = model 
                args.weight_updating = weight_updating
                d = Train(args)
                fileName = model + "_" + feedback + "_" + shift + ".pkl"
                #df.to_pickle(args.saveFolder + fileName)
                df = pd.concat([df, d], ignore_index=True)

    #df.to_pickle(args.saveFolder + "All_Results.pkl")
    
    p = sns.relplot(df, x="Timestep", y="Correct", col="Feedback", row="IntraDimensional", hue="Name", kind="line")
    plt.show()
            