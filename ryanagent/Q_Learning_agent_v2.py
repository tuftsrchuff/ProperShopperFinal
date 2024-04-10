#Adapted from Hang's Q_Learning_agent file
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization
import time

class QLAgent:
    def __init__(self, action_space, alpha=0.5, gamma=0.9, temp=1, epsilon=0.3, mini_epsilon=0.25, decay=0.9999):
        self.action_space = action_space 
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.qtable_norms = pd.DataFrame(columns=[i for i in range(self.action_space)])
        self.qtable_x = pd.DataFrame(columns=[i for i in range(self.action_space)])
        self.qtable_y = pd.DataFrame(columns=[i for i in range(self.action_space)])
    
    def trans(self, state, goal_x = None, goal_y = None, granularity=0.15):
        # You should design a function to transform the huge state into a learnable state for the agent
        #It should be simple but also contains enough information for the agent to learn
        player_info = state['observation']['players'][0]
        position = [int(player_info['position'][0] / granularity) * granularity, int(player_info['position'][1] / granularity) * granularity]
        if goal_x is not None:
            return  json.dumps({'position': position, 'goal_x': goal_x}, sort_keys=True)
        if goal_y is not None:
            return json.dumps({'position': position, 'goal_y': goal_y}, sort_keys=True)
        return json.dumps({'position': position}, sort_keys=True)


    
    def check_add(self, state, goal_x=None, goal_y=None):
        #check if state is in qtable and if not add it in
        serialized_state = self.trans(state)
        if serialized_state not in self.qtable_norms.index:
            #Always add norms for each state
            self.qtable_norms.loc[serialized_state] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
        if goal_x is not None:
            serialized_state_x = self.trans(state, goal_x=goal_x)
            if serialized_state_x not in self.qtable_x.index:
                self.qtable_x.loc[serialized_state_x] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
        if goal_y is not None:
            serialized_state_y = self.trans(state, goal_y=goal_y)
            if serialized_state_y not in self.qtable_y.index:
                self.qtable_y.loc[serialized_state_y] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])



    def learning(self, action, state, next_state,  goal_x = None, goal_y = None, reward_x = 0, reward_y =0 , reward_norms = 0):

        #Adding norms state
        self.check_add(state)
        self.check_add(next_state)
        if goal_x is not None:
            self.check_add(state, goal_x = goal_x)
            self.check_add(next_state, goal_x = goal_x)
        if goal_y is not None:
            self.check_add(state, goal_y = goal_y)
            self.check_add(next_state, goal_y = goal_y)
        
        #Update qvalue to add to the table
        q_sa = self.qtable_norms.loc[self.trans(state), action]
        max_next_q_sa = self.qtable_norms.loc[self.trans(next_state), :].max()
        new_q_sa = q_sa + self.alpha * (reward_norms + self.gamma * max_next_q_sa - q_sa)
        self.qtable_norms.loc[self.trans(state), action] = new_q_sa

        if reward_x != 0:
            #Update reward for non zero val on x axis
            q_sa = self.qtable_x.loc[self.trans(state, goal_x=goal_x), action]
            max_next_q_sa = self.qtable_x.loc[self.trans(next_state, goal_x=goal_x), :].max()
            new_q_sa = q_sa + self.alpha * (reward_x + self.gamma * max_next_q_sa - q_sa)
            self.qtable_x.loc[self.trans(state, goal_x=goal_x), action] = new_q_sa

        if reward_y != 0:
            #Update reward for non zero val on y axis
            q_sa = self.qtable_y.loc[self.trans(state, goal_y=goal_y), action]
            max_next_q_sa = self.qtable_y.loc[self.trans(next_state, goal_y=goal_y), :].max()
            new_q_sa = q_sa + self.alpha * (reward_y + self.gamma * max_next_q_sa - q_sa)
            self.qtable_y.loc[self.trans(state, goal_y=goal_y), action] = new_q_sa
    


    def action_prob(self, state, goal_x = None, goal_y = None, train = True):
        self.check_add(state)

        self.check_add(state, goal_x=goal_x)

        self.check_add(state, goal_y=goal_y)


        p = np.random.uniform(0, 1)
        self.epsilon *= 0.99

        #Only take truly random actions for training
        if p <= self.epsilon and train:
            return np.array([1/self.action_space for i in range(self.action_space)])
        else:
            #Grab q table vals from norms if no goals, otherwise combine q table vals
            if goal_x is None and goal_y is None:
                prob = F.softmax(torch.tensor(self.qtable_norms.loc[self.trans(state)].to_list(), dtype=torch.float), dim=0).detach().numpy()
            elif goal_x is None:
                #combine prob from three qtables
                prob = F.softmax(torch.tensor(self.qtable_y.loc[self.trans(state, goal_y=goal_y)].to_list(), dtype=torch.float), dim=0).detach().numpy()
                prob = prob / prob.sum()
            elif goal_y is None:
                prob = F.softmax(torch.tensor(self.qtable_x.loc[self.trans(state, goal_x=goal_x)].to_list(), dtype=torch.float), dim=0).detach().numpy()
                prob = prob / prob.sum()
            else:
                prob = ( F.softmax(torch.tensor(self.qtable_x.loc[self.trans(state, goal_x=goal_x)].to_list(), dtype=torch.float), dim=0).detach().numpy() 
                      + F.softmax(torch.tensor(self.qtable_y.loc[self.trans(state, goal_y=goal_y)].to_list(), dtype=torch.float), dim=0).detach().numpy() )
                prob = prob / prob.sum()
            return prob

    def choose_action(self, state, goal_x = None, goal_y = None, train = True):

        #Reduce randomness for non-training agent
        if not train:
            self.epsilon = self.mini_epsilon
        self.check_add(state)
        p = np.random.uniform(0, 1)
        if self.epsilon >= self.mini_epsilon:
            self.epsilon *= self.decay
        if p <= self.epsilon and train:
            #Only take truly random action in training scenario
            return np.random.choice([i for i in range(self.action_space)])
        else:
            prob = self.action_prob(state, goal_x=goal_x, goal_y=goal_y, train=train)
            action = np.random.choice([i for i in range(self.action_space)], p=prob)
            cnt = 0

            #Taking greedy q-val action for performance, not optimal with local max in Q-tables
            # if not train:
            #     #If all the same prob, take random action
            #     if np.all(np.isclose(prob, prob[0])):
            #         action = np.random.choice([i for i in range(self.action_space)])
            #     else:
            #         action = np.argmax(prob)
            #     print(f"Index {action} selected")

            #Norm violation for that action, take a different action
            while self.qtable_norms.loc[self.trans(state)].to_list()[action] < 0:
                cnt += 1
                action = np.random.choice([i for i in range(self.action_space)], p=prob)
                if cnt > 10:
                    action = np.random.choice([i for i in range(self.action_space)])
            return action
