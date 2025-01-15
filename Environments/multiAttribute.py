import numpy as np
import random 
import copy

class multiAttribute:
    def __init__(self, args):
        self.args = args
        #np.random.seed(args.seed)
        #random.seed(args.seed)

        self.num_dims = args.nd 
        self.num_attributes = args.na
        self.num_choices = args.nc 
        self.values = ["Value " + str(value) for value in range(self.num_dims)]
        self.default_values = copy.deepcopy(self.values)
        self.attributes = ["Attribute " + str(attribute) for attribute in range(self.num_attributes)]
        self.valued_dim = np.random.choice(range(self.num_dims))
        self.valued_attribute = np.random.choice(range(self.num_attributes))
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
    
    def reset(self):
        if(self.hasReset):
            print("This environment does not support multiple resets.")
            assert(False)
        if(self.args.id):
            # New attribute values with the same dimension being relevant  
            self.values = ["Value " + str(value) for value in range(self.num_dims, 2*self.num_dims)]
            self.default_values = copy.deepcopy(self.values)
        if(self.args.ed):
            # New values 
            self.values = ["Value " + str(value) for value in range(self.num_dims, 2*self.num_dims)]
            self.default_values = copy.deepcopy(self.values)
            # New dimension being relevant 
            self.valued_dim = np.random.choice(range(self.num_dims))
            self.valued_attribute = np.random.choice(range(self.num_attributes))
        self.hasReset = True 
        return self.get_choices() 

    def step(self, action=None):
        if(self.args.risky):
            highReward = np.random.choice([0,10], 1, p=[0.5, 0.5])[0]   # Higher reward risky option
            lowReward = 4                                               # Lower reward safe option
        else:
            highReward = np.random.choice([0,1], 1, p=[0.25, 0.75])[0] + np.random.normal(0,.1)
            lowReward = np.random.choice([0,1], 1, p=[0.75, 0.25])[0] + np.random.normal(0,.1)

        correct = 0
        if(list(action.values())[self.valued_attribute] == self.default_values[self.valued_dim]):
            reward = highReward
            correct = 1
        else:
            reward = lowReward

        #print("Chose: ", list(action.values())[self.valued_attribute], " Valued is ", self.default_values[self.valued_dim], "Reward is ", reward)
        choice_idx = -1 
        for idx, choice in enumerate(self.choices):
            if(choice[self.attributes[self.valued_attribute]] == self.default_values[self.valued_dim]):
                choice_idx = idx

        if(self.args.feedback == "Additional"):
            reward = [lowReward for _ in range(len(self.choices))]
            reward[choice_idx] = highReward

        return self.get_choices(), reward, correct

    
    

