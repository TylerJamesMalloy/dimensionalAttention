import pyactup
import numpy as np
import random 
from pyibl import Agent 
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy
import copy 

class HFRLAgent:
    def __init__(self, args, env):
        """
        self.args.default_utility
        self.args.na
        self.args.nd
        """
        self.args = args
        #np.random.seed(args.seed)
        #random.seed(args.seed)
        self.attributes = ["Attribute " + str(attr) for attr in range(self.args.na)]
        self.values = ["Value " + str(value) for value in range(2*self.args.nd)]
        self.env = env

        self.f_table = np.ones((self.args.na, 2*self.args.nd)) * self.args.default_utility
        self.weights = np.ones(self.args.na)
        self.pending_values = []
        self.pending_attributes = []
        self.unchosen_values = []
        self.unchosen_attributes = []
        self.pending_error = 0
        self.pending_weight = 0
        self.alpha = 0.5

    def reset(self):
        self.f_table = np.ones((self.args.na, 2*self.args.nd)) * self.args.default_utility
        self.weights = np.ones(self.args.na)
        self.pending_values = []
        self.pending_attributes = []
        self.unchosen_values = []
        self.unchosen_attributes = []
        self.pending_error = 0
        self.pending_weight = 0
        return

    def respond(self, response):
        if(response == None):
            return 
        elif(isinstance(response, list)):
            for choice, addReward in zip(self.choices, response):
                self.pending_attributes = list(choice.keys())
                self.pending_values = list(choice.values())
                self.respond(addReward)
        else:
            for atr, val in zip(self.pending_attributes, self.pending_values):
                featureIdx = self.values.index(val)
                attributeIdx = self.attributes.index(atr)
                self.f_table[attributeIdx, featureIdx] = self.f_table[attributeIdx, featureIdx] + self.args.lr * (response - self.f_table[attributeIdx, featureIdx])
                self.pending_error = self.args.lr * (response - self.f_table[attributeIdx, featureIdx])
                self.pending_weight = attributeIdx

            for atr, val in zip(self.unchosen_attributes, self.unchosen_values):
                featureIdx = self.values.index(val)
                attributeIdx = self.attributes.index(atr)
                self.f_table[attributeIdx, featureIdx] *= self.args.decay
        return

    def choose(self, choices=None, details=False):
        choice_values = []
        for choice in choices:
            value = 0
            for attributeIdx, attribute in enumerate(self.attributes):
                featureIdx = self.values.index(choice[attribute])
                value += self.f_table[attributeIdx, featureIdx]
            choice_values.append(value)
        softmax = np.array(choice_values)
        exp_x = np.exp(choice_values - np.max(choice_values)) * self.args.softMaxInverseTemp
        softmax = exp_x / np.sum(exp_x)
        choiceIdx = np.argmax(softmax)
        chosen = choices[choiceIdx]
        self.pending_attributes = list(choices[choiceIdx].keys())
        self.pending_values = list(choices[choiceIdx].values())
        choicesCopy = copy.deepcopy(choices)
        choicesCopy.pop(choiceIdx)

        self.unchosen_attributes = list(choicesCopy[0].keys()) + list(choicesCopy[1].keys())
        self.unchosen_values = list(choicesCopy[0].values()) + list(choicesCopy[1].values())
        return chosen, None 

    def updateWeights(self):
        self.weights[self.pending_weight] = self.alpha * (self.pending_error - self.weights[self.pending_weight])
        return None 
        