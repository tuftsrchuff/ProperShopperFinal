#Training for target list of possible shopping areas
#Adapted from Hang's training

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent_v2 import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd
import numpy as np

reach_cnt = 0
global min_x 
global min_y 
cart = False

shelfPos = {
    'milk': [5.5, 3.6], 
    'chocolate milk': [9.5, 3.6], 
    'strawberry milk': [13.5, 3.6], 
    'apples': [5.5, 7.6], 
    'oranges': [7.5, 7.6], 
    'banana': [9.5, 7.6], 
    'strawberry': [11.5, 7.6], 
    'raspberry': [13.5, 7.6], 
    'sausage': [5.5, 11.6], 
    'steak': [7.5, 11.6], 
    'chicken': [11.5, 11.6], 
    'ham': [13.5, 11.6], 
    'brie cheese': [5.5, 15.6], 
    'swiss cheese': [7.5, 15.6], 
    'cheese wheel': [9.5, 15.6], 
    'garlic': [5.5, 19.6], 
    'leek': [7.5, 19.6], 
    'red bell pepper': [9.5, 19.6], 
    'carrot': [11.5, 19.6], 
    'lettuce': [13.5, 19.6], 
    'avocado': [5.5, 23.35], 
    'broccoli': [7.5, 23.35], 
    'cucumber': [9.5, 23.35], 
    'yellow bell pepper': [11.5, 23.35], 
    'onion': [13.5, 23.35],
    'prepared foods': [16.8, 5.5],
    'fresh fish': [16.8, 11.7]
}

cart_loc = [1.2, 17.7]
start_loc = [1.2, 15.6]


def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def getCartAndGoToStart(sock_game):
    sock_game.send(str.encode("0 NOP"))
    state = recv_socket_data(sock_game)
    state = json.loads(state)
    # print("State")
    # print(state)
    while state["observation"]["players"][0]["position"][0] < cart_loc[0]:
        sock_game.send(str.encode("0 EAST"))
        state = recv_socket_data(sock_game)  # get observation from env
        state = json.loads(state)
    
    while state["observation"]["players"][0]["position"][1] < cart_loc[1]:
        sock_game.send(str.encode("0 SOUTH"))
        state = recv_socket_data(sock_game)  # get observation from env
        state = json.loads(state)
    
    sock_game.send(str.encode("0 INTERACT"))
    state = recv_socket_data(sock_game)  # get observation from env
    state = json.loads(state)

    while state["observation"]["players"][0]["position"][0] > start_loc[0]:
        sock_game.send(str.encode("0 WEST"))
        state = recv_socket_data(sock_game)  # get observation from env
        state = json.loads(state)
    
    while state["observation"]["players"][0]["position"][1] > start_loc[1]:
        sock_game.send(str.encode("0 NORTH"))
        state = recv_socket_data(sock_game)  # get observation from env
        state = json.loads(state)

def calculate_reward(previous_state, current_state, goal_x = None, goal_y = None, norms = None):

    global min_x
    global min_y

    reward_x = -1
    reward_y = -1
    reward_norms = 0
    if norms is not None and norms != '':
        reward_norms -= 100 * len(norms)
        reward_x -= 100 * len(norms)
        reward_y -= 100 * len(norms)
    
    if goal_x is not None:
        dis_x = abs(current_state['observation']['players'][0]['position'][0] - goal_x)
        dis_y = abs(current_state['observation']['players'][0]['position'][1] - goal_y)
        if dis_x < min_x:
            min_x = dis_x
            reward_x = 10
        if dis_y < min_y:
            min_y = dis_y
            reward_y = 10

    if abs(current_state['observation']['players'][0]['position'][0] - goal_x) < 0.2 and abs(current_state['observation']['players'][0]['position'][1] - goal_y) < 0.2:
        reward_x = 100000
        reward_y = 100000
        global reach_cnt
        reach_cnt += 1
        print("Goal reached:", reach_cnt)

    return reward_x, reward_y, reward_norms

if __name__ == "__main__":
    

    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    agent.qtable_norms = pd.read_json('qtable_norms.json')
    agent.qtable_x = pd.read_json('qtable_x.json')
    agent.qtable_y = pd.read_json('qtable_y.json')
####################
    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    sock_game.send(str.encode("0 RESET"))  # reset the game

    #Build initial list of locations

    output = recv_socket_data(sock_game)  # get observation from env
    output = json.loads(output)
    shelves = output["observation"]["shelves"]
    # for shelf in shelves:
    #     if shelf["food_name"] not in shelfPos:
    #         modShelfX = shelf["position"][0]
    #         modShelfY = shelf["position"][0] - 2.1
    #         shelfPos[shelf["food_name"]] = [modShelfX, modShelfY]
    
    print(shelfPos)


    for key in shelfPos:
        training_time = 1000
        episode_length = 1000
        for i in range(training_time):
            print(f"Training round {i} for {key}")
            global min_x
            global min_y
            min_x = 1000
            min_y = 1000
            sock_game.send(str.encode("0 RESET"))  # reset the game
            output = recv_socket_data(sock_game)
            state = json.loads(output)
            #print(state)
            #episode_length = 100 + i
            cnt = 0
            agent.qtable_norms.to_json('qtable_norms.json')
            agent.qtable_x.to_json('qtable_x.json')
            agent.qtable_y.to_json('qtable_y.json')
            # create a switch that randomly chooses 0 or 1
            switch = random.randint(0, 1)
            reward_cnt = 0
            getCartAndGoToStart(sock_game)
            while not state['gameOver']:
                cnt += 1
                # Choose an action based on the current state
                if abs(state['observation']['players'][0]['position'][0] - shelfPos[key][0]) < 0.2:
                    switch = 1
                if abs(state['observation']['players'][0]['position'][1] - shelfPos[key][1]) < 0.2:
                    switch = 0 

                if switch == 0:
                    action_index = agent.choose_action(state, goal_x=shelfPos[key][0])  
                else:
                    action_index = agent.choose_action(state, goal_y=shelfPos[key][1])

                action = "0 " + action_commands[action_index]

                # print("Sending action: ", action)
                sock_game.send(str.encode(action))  # send action to env

                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]


                # Define the reward based on the state and next_state
                reward_x, reward_y, reward_norms = calculate_reward(state, next_state, shelfPos[key][0], shelfPos[key][1], norms)  # You need to define this function
                # print("------------------------------------------")
                # print(reward_x, reward_y, min_x, min_y, action_commands[action_index], state['observation']['players'][0]['position'], next_state['observation']['players'][0]['position'])    
                # print("------------------------------------------")
                # Update Q-table
                if reward_x < 0 and reward_y < 0:
                    reward_cnt += 1
                
                if reward_cnt > 20:
                    reward_cnt = 0
                    switch = 1 - switch

                agent.learning(action_index,  state, next_state, shelfPos[key][0], shelfPos[key][1], reward_x, reward_y, reward_norms)

                # Update state
                state = next_state


                if (abs(state['observation']['players'][0]['position'][0] - shelfPos[key][0]) < 0.2 and abs(state['observation']['players'][0]['position'][1] - shelfPos[key][1]) < 0.2):
                    print("Destination reached")
                    break
                    
                if cnt > episode_length:
                    print("Episode length reached")
                    break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()

