import pyactup
import numpy as np
import random 
from pyibl import Agent, DelayedResponse
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy


def similarity(x,y):
    print("In similarity x = " + str(x))
    print("In similarity y = " + str(y))
    print("X == Y: ", x == y)
    if(x==y):
        return 1
    else: 
        return 0
    
class HIBLAgent:
    def __init__(self, args, env):
        """
        Initialize the HIBL agent with a name and possible actions.

        :param agent_name: The name of the agent.
        :param actions: A list of possible actions the agent can take.
        """
        self.args = args
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.attributes = ["Attribute " + str(attr) for attr in range(self.args.na)]
        self.values = ["Value " + str(value) for value in range(self.args.nd)]
        self.agents = []
        for agent_idx in range(self.args.nd):  
            agent = Agent(name="name " + str(agent_idx), attributes=self.attributes, mismatch_penalty=1)
            for attribute in self.attributes:
                agent.similarity([attribute], similarity)
            weights = np.ones(len(self.attributes))
            print(self.values[agent_idx])
            agent.populate(choices=[{"Attribute 0": self.values[agent_idx]}], outcome=1000 + agent_idx)
            self.agents.append({"Agent":agent, "Weights": weights, "Memory": [], "Delayed Responses":[]})
        
        for agent in self.agents:
            print(agent['Agent']._memory.values())
        
        self.agent = Agent(name="name", default_utility=self.args.default_utility, mismatch_penalty=1)
        self.agentChoices = list(range(len(self.agents)))
        self.alpha = .5
        self.env = env
        self.pendingAgentChoice = None 
        self.delay = 0

    def respond(self, response):
        if(response == None):
            for agentIdx, agentDict in enumerate(self.agents):
                agent = agentDict['Agent']
                delayed = agent.respond()
                self.agents[agentIdx]['Delayed Responses'].append(delayed)
            self.agent.respond()
            self.delay += 1
        else:
            chosenAgent = self.agents[self.pendingAgentChoice]

            for delayedResponse in chosenAgent['Delayed Responses']:
                delayedResponse.update(response/self.delay)
                
            for agentIdx, agentDict in enumerate(self.agents):
                agent = agentDict['Agent']
                if(agentIdx != self.pendingAgentChoice):
                    for oldResponse, newResponse in zip(agentDict['Delayed Responses'], chosenAgent['Delayed Responses']):
                        updatedResponse = DelayedResponse(agent, newResponse._attributes, newResponse._expectation)
                        updatedResponse.update(response/self.delay)
                        del oldResponse
                agent.respond(response)
            
            self.delay = 0
            self.agent.respond(response)
        return 

    def choose(self, choices=None, details=False):
        agentChoices = []
        agentDetails = []
        for agentDict in self.agents:
            agent = agentDict['Agent']
            if(details):
                print("Agent: ", agent.name, " Memory is ", agent._memory.values())
                choice, details = agent.choose(choices, details=True)
                print(details)
                agentDetails.append(details)
            else:
                choice = agent.choose(choices)
            agentChoices.append(choice)

        assert(False)
        if(details):
            choice, details = self.agent.choose(self.agentChoices, details=True)
            #print("Main Agent Details: ", details)
            agentDetails.append(details)
        else:
            choice = self.agent.choose(self.agentChoices)
        
        print(agentChoices)
        assert(False)
        self.pendingAgentChoice = choice 
        choice = agentChoices[choice]
        return choice, details
    
    def updateWeights(self):
        return 