import pyactup
import numpy as np
import random 
from pyibl import Agent 
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy, mode 

class IBLAgent:
    def __init__(self, args, env):
        """
        Initialize the IBL agent with a name and possible actions.

        :param agent_name: The name of the agent.
        :param actions: A list of possible actions the agent can take.
        """
        self.args = args
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.attributes = ["Attribute " + str(attr) for attr in range(self.args.na)]
        self.values = ["Value " + str(value) for value in range(2*self.args.nd)]
        self.agent = Agent(name="name", attributes=self.attributes, default_utility=self.args.default_utility, mismatch_penalty=1)
        for attribute in self.attributes:
            self.agent.similarity([attribute], lambda x,y: 1 if x == y else 0)
        self.weights = np.ones(len(self.attributes))
        self.memory = []
        self.alpha = .5
        self.env = env
        self.delayedResponses = []
        self.delay = 0


    def reset(self):
        self.weights = np.ones(len(self.attributes))
        self.agent.reset()
        self.memory = []

    def respond(self, response):
        reward = None 
        if(response == None):
            self.delay += 1
            delayedResponse = self.agent.respond()
            self.delayedResponses.append(delayedResponse)
        elif(isinstance(response, list)):
            (indx, pending_choices, queries, utilities) = self.agent._pending_decision
            for choice, addReward in zip(pending_choices, response):
                choice_idx = pending_choices.index(choice)
                self.agent._pending_decision = (choice_idx, pending_choices, queries, utilities)
                self.agent.respond(addReward)
                values = [self.values.index(list(choice.values())[attr_idx]) for attr_idx in range(self.args.na)]
                values.append(addReward)
                self.memory.append(values)
            reward = response[indx]
        else:
            # Need to do something different here for delayed feedback
            values = [self.values.index(list(self.pending_choice.values())[attr_idx]) for attr_idx in range(self.args.na)]
            values.append(response)
            self.memory.append(values)
            self.agent.respond(response)
            if(self.delay != 0):
                if(self.args.weight_updating and len(self.memory) > 5):
                    data = np.array(self.memory)
                    last_five_slice = slice(-5, None)
                    sample = data[last_five_slice]
                    X = sample[:, :3] 
                    
                    valuedDimPrediction = np.argmax(self.weights)
                    mostCommonValuedDim = mode(X[:, valuedDimPrediction])
                    for delayedResponse in self.delayedResponses:
                        baseResponse = response / self.delay
                        responseWeight = 1
                        for attributeIndex, attribute in enumerate(delayedResponse._attributes):
                            if(attribute == mostCommonValuedDim):
                                responseWeight += np.abs(self.weights[attributeIndex])
                        delayedResponse.update(baseResponse * responseWeight)
                else:
                    for delayedResponse in self.delayedResponses:
                        if(self.delay > 0):
                            delayedResponse.update(response / self.delay)
                        self.delay = 0 
        return reward
    
    def choose(self, choices=None, details=False):
        """
        Get the agent's predicted action based on the provided context.

        :param context: A dictionary representing the context for the agent.
        :return: The predicted action.
        """
        choice = self.agent.choose(choices, details=details)
        if(details):
            self.pending_choice = choice[0]
        else:
            self.pending_choice = choice

        return choice, None 

    def updateWeights(self):
        if(not self.args.weight_updating):
            return None 
        
        if(len(self.memory) > 5):
            data = np.array(self.memory)
            #last_twenty_slice = slice(-20, None)
            #sample = data[last_twenty_slice]
            sample = data
            
            X = sample[:, :3]  # First three columns
            y = sample[:, -1]    # Last column
            
            mi = np.array(mutual_info_regression(X, y, discrete_features=True))
            self.weights = self.weights + (self.alpha * (mi - self.weights))
            for attribute, weight in zip(self.attributes, self.weights):
                weight = np.clip(weight, 0, 1)
                self.agent.similarity([attribute], weight=weight*100)

            #print(self.weights, " With relevant attribute: ", self.env.valued_attribute)

            return self.weights
        return None 
        