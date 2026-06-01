import torch
import numpy as np 

env = smartGridEnv()

policy = PolicyNet()
optimizer = torch.optim.Adam(policy.parameters(), lr = 3e-4) #define PDF over actions

#Hyperparams -> discount factor close to 1 to place high weightage on future
#but can be changed
#epsClip controls PPO stability -> prevents policy from changing too much
#epocs -> how many times can same data be used for learning -> arbitrarily
#chosen for now
gamma = 0.99
epsClip = 0.2
epochs = 4

obs, info = env.reset() #starts new day/episode

#storage buffers
states = []
actions = []
logprobs []
rewards = []
values = []
dones = []

done = False
#run loop till done
while not done:
    obsTensor = torch.tensor(obs, dtype = torch.float32) #converts arr to tens for nn

    mu, std = policy.forward(obsTensor) #action ~ N(mu,sd)

    dist = torch.distributions.Normal(mu, std) # create PDF -> turns policy into stochastic actor

    action = dist.sample() #randomly select an action
    logprob = dist.log_prob(action).sum() #compute log probability

    nextObs, reward, terminated, truncated info = env.step(
        torch.tanh(action).detach().numpy()) #step environment 
    #return next state, reward, whether day/episode has ended

    #trajectory dataset
    states.append(obs)
    actions.append(action.detach())
    logprobs.append(logprob.detach())
    rewards.append(reward)
    dones.append(terminated or truncated)

    
    obs = nextObs 
    done = terminated or truncated

    #compute returns
    returns = []
    G=0
    
    for r in reversed(rewards):
        G = r+gamma*G
        returns.insert(0,G)

        #R_t = r_t = y r_{t+1} + y^2 r_{t+2} + ...

    #normalization of returns
    returns = torch.tensor(returns, dtype = torch.float32)
    returns = (returns - returns.mean())/(returns.std()+1e-8)

    #convert to tensor
    states = torch.tensor(np.array(states), dtype = torch.float32)
    actions = torch.stack(actions)
    oldLogprobs = torch.stack(logprobs)

    #update loop
    for _ in range(epochs):

        #recompute updated weights
        mu, std = policy.forward(states)
        dist = torch.distributions.Normal(mu, std)

        #liklihood of old actions
        newLogprobs = dist.log_prob(actions).sum(dim = 1)

        # \frac{\pi_{new}}{pi_{old}} since \e^{\logx-\logy} = \frac{y}{x}
        ratio = torch.exp(newLogprobs-oldLogprobs)

        advantage = returns #simplification -> could use critic I leave that
        # up to u

        surr1 = ratio*advantage #Ver1 of PPO
        surr2 = torch.clamp(ratio, 1-epsClip, 1+epsClip)*advantage #Ver2 (clipped) of PPO

        loss = -torch.min(surr1, surr2).mean() #minimize loss -> so minus

        #Backprop -> update policy weights
        optimizer.zero_grad() #gradient
        loss.backward() 
        optimizer.step()
