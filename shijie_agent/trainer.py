#Author Hang Yu
# assignment done by Shijie Fang

# some notes:
# Please set the "mode" variable to eval for testing the trained agent, so that
# it won't modify the trained Q table.
# if you want to train the agent based on current pretrained policies, set "mode" to "continue training"
# for training a new policy and OVERWRITE the old one, set "mode" to "train"

# you can set the training target with ease by setting the "target object" to any object in "TARGET_CART_POSITION"
# of config.yaml file

# if the program can't find the yaml file, change the "cfg_path"

# HINT: if use_cart_hacking is set to true, the agent will send a hacking message to env when resetting
# this enables the agent to always have a cart after reset.
# the recommandation is to set this variable to True for every target apart from "exit" and "basketReturns"


# suitable with the --random_start parameter for the env


import json
import random
import socket
import numpy as np
import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

# make sure we can find Q_Learning_agent module
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Q_learner
from Q_learner import QLAgent  # Make sure to import your QLAgent class
import pickle
import time
import yaml


# some useful flags
switch_q_table_flag = False
got_cart_first_time_in_episode = True
task_complete_flag = False
save_q_table_flag = False

cart = False
exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 
violation_count = 0

def distance_to_cart(state):
    agent_position = state['observation']['players'][0]['position']
    if agent_position[0] > 1.5:
        cart_distances = [euclidean_distance(agent_position, cart_pos_right)]
    else:
        cart_distances = [euclidean_distance(agent_position, cart_pos_left)]
    return min(cart_distances)

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

# check whether the agent is at leaving position
# I looked into env code and got these positions
def check_leaving(agent_pos):
    return (agent_pos[0] <= 1 and 15 <= agent_pos[1] <= 15.8) or (agent_pos[0] <= 1 and (7 <= agent_pos[1] <= 7.8 or 3 <= agent_pos[1] <= 3.8))

# check whether the agent is getting closer to target
def check_getting_closer(previous_pos, current_pos, target):
    if euclidean_distance(current_pos,target) < euclidean_distance(previous_pos, target):
        return 1
    else:
        return 0

# creating a global "gradient-like" reward table:
# heading to correct direction result in positive reward
# I tried this but it doesn't work, the agent would be trapped in corner or wall 
# I'll leave the code to show my thoughts, though
def initialize_reward_table(granularity, original_target_pos):
    reward_table = np.zeros((200,200,4),dtype=float)
    # reward_table -= 3
    trans_target_pose = np.floor([pos/granularity for pos in original_target_pos])
    target_pos = np.array(trans_target_pose)
    for i in range(reward_table.shape[0]):
        for j in range(reward_table.shape[1]):
            pos = np.array([i,j])
            if target_pos[1] > pos[1]:
                reward_table[i][j][0] = 1
            if target_pos[1] < pos[1]:
                reward_table[i][j][1] = 1
            if target_pos[0] > pos[0]:
                reward_table[i][j][2] = 1
            if target_pos[0] < pos[0]:
                reward_table[i][j][3] = 1
    return reward_table

# TODO: compared to last assignment, I modified the reward function to make training better
# the main change is increasing reward of getting closer to target from 1 to 5
def calculate_reward(agent:QLAgent,previous_state, current_state, target, action_index, reward_table,violation_num):
    agent_position = current_state['observation']['players'][0]['position']
    global task_complete_flag
    global violation_count
    prev_agent_pos = previous_state['observation']['players'][0]['position']
    agent.explored(current_state)
    # if reaching target, give big reward and reset
    if euclidean_distance(agent_position,target)<=0.5:
        # print("reaching target!")
        task_complete_flag = True
        return 100
    # we don't want the agent to make use of random reset to get into a better starting pose
    # so punishment when agent want to exit the market
    elif check_leaving(agent_position):
        agent.set_explored(current_state)
        return -10
    # if violate norms, like running into things, give punishment
    elif violation_num > 0:
        # print("violation!")
        # agent.set_explored(current_state)
        violation_count += violation_num
        return -10
    # somehow when agent run into things, violation doesn't always trigger
    # so manually check whether the agent is able to move
    # if not we assume it run into things
    elif euclidean_distance(agent_position,prev_agent_pos)<=0.2:
        # print("running into things")
        agent.set_explored(current_state)
        return -10
    # give a positive reward when
    # 1. the agent is getting closer
    # 2. this state is not visited in this episode
    # to prevent the agent from hacking the reward function
    # and also encourage it to explore eays to get to target
    # Checking explored is crucial to training, it ensures that our system works
    elif check_getting_closer(prev_agent_pos, agent_position,target) and (agent.explored(current_state)==0):
        agent.set_explored(current_state)
        return 5
    # not used, turn out doesn't work and may pollute Q table
    # elif not  check_getting_closer(prev_agent_pos, agent_position,target):
    #     print("getting further")
    #     return -2

    # if not getting closer:
    # give 0 reward for new states
    # give -0.5 reward for visited states
    else:
        if agent.explored(current_state) ==0:
            agent.set_explored(current_state)
            return 0
        elif agent.explored(current_state)>5:
            agent.set_explored(current_state)
            return -5
        else: 
            agent.set_explored(current_state)
            return -0.5
    # not used, code for the "reward table" that failed to work
    # else:
    #     if agent.explored(current_state) == 0:
    #         agent.set_explored(current_state)
    #         print(reward_table[*agent.trans(previous_state)][action_index])
    #         return reward_table[*agent.trans(previous_state)][action_index]
    #     else:
    #         return -2


if __name__ == "__main__":
    
    cfg_path = "config.yaml"
    target_object = "apples"
    use_cart_hacking = True

    with open(cfg_path,'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    # action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']

    # for navigation, we only allow agent to move to simplify the problem
    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST']

    ############
    # please set this to train for training , and set this to eval for evaluating
    # when in eval mode, the Q table file won't be changed

    # !!! Please note: when you set mode to train, modify the file's name in the saving

    # part, or the trained Q table will be overwritten
    ############
    mode = "continue training"
    # Initialize Q-learning agent
    granularity = 0.45
    # agent move 0.15 with a single action  - 11
    single_action_change = 0.15
    action_space = len(action_commands)   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space,granularity=granularity,epsilon=0.5, mini_epsilon=0.1,decay=0.999)
    

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    #agent.qtable = pd.read_json('qtable.json')
####################

    exit_mall_flag = False
    steps_count = 0
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    # target = [1,17.5] #target pos
    target = yaml_data["TARGET_CART_POSITION"][target_object]
    # agent.q_table = np.load('trained_qtable_{}_{},{}.npy'.format(target_object,target[0],target[1]))
    # target = list(target)
    if mode == "eval" or mode == "continue training":
        agent.q_table = np.load('trained_qtable_{}_{},{}.npy'.format(target_object,target[0],target[1]))
        agent.epsilon = 0.1
    print(target)
    training_time = 100
    episode_length = 2000
    violation_count = 0
    # we need accumulative violation count
    current_step_violations = 0
    reward_table = initialize_reward_table(granularity, target)
    for i in range(training_time):
        print(i)
        task_complete_flag = False
        # reset the explored states everytime we reset the env
        agent.restore_explored_table()
        if use_cart_hacking:
            sock_game.send(str.encode("0 HACK"))  # reset the game
        else:
            sock_game.send(str.encode("0 RESET"))
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        # sock_game.send(str.encode("0 HACK"))
        # recv_socket_data(sock_game)
        steps_count = 0
        # state = recv_socket_data(sock_game)
        # state = json.loads(state)
        cnt = 0
        print("starting!")
        violation_count = 0
        while not state['gameOver']:
            cnt += 1
            agent_position = state['observation']['players'][0]['position']
            # By default the env exit if the agent leave
            # here we reset the env manually if the agent is leaving
            # crucial for the training to run
            if check_leaving(agent_position) or task_complete_flag==True:
                print("reset!")
                time.sleep(1)
                if use_cart_hacking:
                    sock_game.send(str.encode("0 HACK"))
                else:
                    sock_game.send(str.encode("0 RESET"))
                state = recv_socket_data(sock_game)
                state = json.loads(state)
                # sock_game.send(str.encode("0 HACK"))
                agent.restore_explored_table()
                steps_count = 0
                violation_count = 0
                task_complete_flag = False
                # recv_socket_data(sock_game)
                # state = json.loads(state)
            # Choose an action based on the current state
            action_index = agent.choose_action(state)
            # print(action_index)
            action = "0 " + action_commands[action_index]
            # prevent env from exiting early
            
            # print("Sending action: ", action)

            # instead of moving 0.15 at a time, we move granularity at a time by repeatly sending command
            # This is to shrink the observation space and make the learning simpler
            current_step_violations = 0
            for _ in range(int(granularity/single_action_change)):
                sock_game.send(str.encode(action))  # send action to env
                next_state = recv_socket_data(sock_game)
                next_state = json.loads(next_state)
                if len(next_state["violations"]) != 0:
                    # print(next_state["violations"])
                    # modified this so that it doesn't misscount the violations
                    current_step_violations += len(next_state["violations"])
                time.sleep(0.004)
            
            next_agent_pose = next_state['observation']['players'][0]['position']
            steps_count += 1
            # Define the reward based on the state and next_state
            reward = calculate_reward(agent, state, next_state,target, action_index,reward_table,current_step_violations)  # You need to define this function
            if task_complete_flag == True:
                print("completing task in {} steps, with {} norms violated".format(steps_count,violation_count))
                if mode == "train" or mode == "continue training":
                    #######
                    # change here to prevent overwritting!
                    #######
                    np.save('trained_qtable_{}_{},{}.npy'.format(target_object,target[0],target[1]), agent.q_table)
                    time.sleep(0.5)
            if reward > 0:
                pass

            agent.learning(action_index, reward, state, next_state)
            # Update state
            exit_mall_flag = False
            state = next_state

            if cnt > episode_length:
                break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()

