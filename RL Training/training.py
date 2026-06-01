from pathlib import Path

from fastapi import FastAPI
import torch
import torch.nn as nn
import numpy as np

RL_DIR = Path(__file__).resolve().parent

app = FastAPI() #Create FastAPI application instance -> webserver

class PolicyNet(nn.Module): # policy network class
    def __init__(self, obsDim = 10, actDim = 2): #network structure
        super().__init__() #constructor

        self.net = nn.Sequential( #Feed forward NN defined
            nn.Linear(obsDim, 128), #First linear layer -> mapping observations to hidden NN layer
            nn.ReLU(), #non linear activation
            nn.Linear(128,128), #Hidden Layer
            nn.ReLU()
        )

        self.mu = nn.Linear(128, actDim) #output layer for mean action values
        self.logStd = nn.Parameter(torch.zeros(actDim)) #learnable tensor to store logstd
        
    def forward(self, x):
        x = self.net(x) #Pass input thru hidden layers
        mu = torch.tanh(self.mu(x)) # Normalize range
        std = torch.exp(self.logStd) # e^{logStd} = Std
        return mu, std

        
    def act(self, obs): #sample action from policy
        obs = torch.tensor(obs, dtype = torch.float32) #Convert numpy obsv to pytorch tensor
        mu, std = self.forward(obs) #compute mu and stdev of action distribution
        dist = torch.distributions.Normal(mu, std) #Normal distribution for stochastic policy
        action = dist.sample() #sample an action from the distribution
        return action.detach().numpy() #reconvert Tens to np arr

policy = PolicyNet() # global policy object

#load in PPO weights from disk to policy network
policy.load_state_dict(torch.load(RL_DIR / "somefile.pth", map_location="cpu"))

policy.eval() #set model to evaluation mode

@app.post("/act") #define API endpoint
def act(obs: dict):
    observation = np.array(obs["state"], dtype = np.float32)
    action = policy.act(observation)
        
    return{"action": action.tolist()}