import numpy as np
import random 
import copy

class delayedFeedback:
    def __init__(self, args):
        self.args = args
        #np.random.seed(args.seed)
        #random.seed(args.seed)

        self.num_dims = args.nd 
        self.num_attributes = args.na
        self.num_choices = args.nc 
        self.values = ["Value " + str(value) for value in range(self.num_dims)]
        self.default_values = copy.deepcopy(self.values)
        self.attributes = ["Attribute 0"]
        self.valued_dim = 0 #np.random.choice(range(self.num_dims))
        self.hasReset = False
        self.choices = []
        self.pending_choices = None 
        
    
    def get_choices(self):
        self.pending_choices = self.choices
        self.choices = [{} for _ in range(self.num_choices)]
        for attribute in self.attributes :
            np.random.shuffle(self.values)
            for choice_idx in range(self.num_choices):
                self.choices[choice_idx][attribute] = self.values[choice_idx]
        return self.choices 

    def step(self, action=None):
        highReward = np.random.choice([0,10], 1, p=[0.5, 0.5])[0]   # Higher reward risky option
        lowReward = 4                                               # Lower reward safe option

        correct = 0
        if(list(action.values())[0] == self.default_values[self.valued_dim]):
            reward = highReward
            correct = 1
        else:
            reward = lowReward

        return self.get_choices(), reward, correct