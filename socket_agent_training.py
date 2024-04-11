#Author Hang Yu
# Modified by Isabella Bianchi for HW4

import ujson as json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agnet import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd
import numpy as np
import math

reach_cnt = 0
global min_x 
global min_y 
cart = False

# target_list = [[1, 17.5], [10, 18.5], [5.5, 1.5], [9.5, 1.5]]
# target_list = [[10, 18.5], [5.5, 1.5], [9.5, 1.5]]
exit_pos = [1, 3.5] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 17.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 

# Define the grocery items
grocery_items = [
    "milk", "milk", "chocolate milk", "chocolate milk", "strawberry milk",
    "apples", "oranges", "banana", "strawberry", "raspberry",
    "sausage", "steak", "steak", "chicken", "ham",
    "brie cheese", "swiss cheese", "cheese wheel", "cheese wheel", "cheese wheel",
    "garlic", "leek", "red bell pepper", "carrot", "lettuce",
    "avocado", "broccoli", "cucumber", "yellow bell pepper", "onion",
    "prepared foods", "fresh fish"
]

# The x coordinates for the grocery items
x_values = [
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    17.5, 17.5
]

# First aisle they could be got in
first_aisle_values = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    6, 7
]

# Second aisle they could be got in
second_aisle_values = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    5, 5, 5, 5, 5,
    6, 7
]

# What shelf are they on?
shelf_values = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    6, 6, 6, 6, 6,
    0, 0
]

# What are the y values of each shelf?
aisle_y_values = [3.75, 7.8, 11.7, 15.6, 19.65, 23, 8.5, 14.5]

reached_item = False
need_cart = False

# A list of all the grocery items
grocery_list = []
for item, x, aisle1, aisle2, shelf in zip(grocery_items, x_values, first_aisle_values, second_aisle_values, shelf_values):
    grocery_list.append({
        'item': item,
        'x': x,
        'first_aisle': aisle1,
        'second_aisle': aisle2,
        'shelf': shelf
    })

# A list of all the locations we should be trained on
target_list = []
for t in grocery_list:
    target_list.append([t['x'], aisle_y_values[t['first_aisle']]])
    target_list.append([t['x'], aisle_y_values[t['second_aisle']]])

target_list.append(exit_pos)

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


def calculate_reward(previous_state, current_state, goal_x = None, goal_y = None, norms = None):
    global min_x 
    global min_y 
    reward_x = -1
    reward_y = -1
    reward_norms = 0
    #print("Norms: ", norms)
    if norms is not None and norms != '':
        reward_norms -= 100 * len(norms)
        reward_x -= 100 * len(norms)
        reward_y -= 100 * len(norms)
        # print("Violation")
    
    if goal_x is not None:
        # dis_x = -abs(current_state['observation']['players'][0]['position'][0] - goal_x) + abs(previous_state['observation']['players'][0]['position'][0] - goal_x)
        # dis_y = -abs(current_state['observation']['players'][0]['position'][1] - goal_y) + abs(previous_state['observation']['players'][0]['position'][1] - goal_y)
        dis_x = abs(current_state['observation']['players'][0]['position'][0] - goal_x)
        dis_y = abs(current_state['observation']['players'][0]['position'][1] - goal_y)
        if dis_x < min_x:
            min_x = dis_x
            reward_x = 10
        if dis_y < min_y:
            min_y = dis_y
            reward_y = 10
        # if dis_x != 0:
        #     reward_x += dis_x * 100 - 1
        # if dis_y != 0:
        #     reward_y += dis_y * 100 - 1 

    if abs(current_state['observation']['players'][0]['position'][0] - goal_x) < 0.2 and abs(current_state['observation']['players'][0]['position'][1] - goal_y) < 0.2:
        reward_x = 1000
        reward_y = 1000
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

    
    training_time = 100000
    episode_length = 1000
    for i in range(training_time):
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
        reward_cnt = 0

        original_x = state['observation']['players'][0]['position'][0]
        original_y = state['observation']['players'][0]['position'][1]


        # THIS HAS THE PLAYER GO GET A CART BEFORE THEY TRAIN
        # while True:
        #     action = "0 " + action_commands[1]
        #     sock_game.send(str.encode(action))  # send action to env
        #     output = recv_socket_data(sock_game)  # get observation from env
        #     next_state = json.loads(output)
        #     norms = next_state["violations"]
        #     if next_state['observation']['players'][0]['position'][0] > cart_pos_right[0]:
        #         action = "0 " + action_commands[3]
        #         sock_game.send(str.encode(action))  # send action to env
        #         output = recv_socket_data(sock_game)  # get observation from env
        #         next_state = json.loads(output)
        #         norms = next_state["violations"]
            
        #     if next_state['observation']['players'][0]['position'][0] < cart_pos_right[0] and next_state['observation']['players'][0]['position'][1] > 17:
        #         sock_game.send(str.encode("0 TOGGLE_CART"))  # send action to env
        #         output = recv_socket_data(sock_game)  # get observation from env
        #         next_state = json.loads(output)
        #         norms = next_state["violations"]
                
        #         state = next_state
        #         break

        #     print("------------------------------------------")
        #     print("Going to get cart [", cart_pos_right[0], ",", cart_pos_right[1], "]")
        #     print("Current location:", "[", next_state['observation']['players'][0]['position'][0], ",", next_state['observation']['players'][0]['position'][1], "]")
        #     print("x difference:", next_state['observation']['players'][0]['position'][0] - cart_pos_right[0])
        #     print("y difference:", next_state['observation']['players'][0]['position'][1] - cart_pos_right[1])
        #     print("------------------------------------------")
        
        # Shuffle the list so that we get every combination
        random.shuffle(target_list)
        # target_list = [original_x, original_y] + target_list
        for t in target_list:
            og_dis_x = state['observation']['players'][0]['position'][0]
            og_dis_y = state['observation']['players'][0]['position'][1]
            og_dis = math.sqrt((og_dis_x * og_dis_x) + (og_dis_y * og_dis_y))
            while not state['gameOver']:
                switch = random.randint(0, 1)
                cnt += 1
                # Choose an action based on the current state
                if abs(state['observation']['players'][0]['position'][0] - t[0]) < 0.2:
                    switch = 1
                if abs(state['observation']['players'][0]['position'][1] - t[1]) < 0.2:
                    switch = 0 

                if switch == 0:
                    action_index = agent.choose_action(state, goal_x=t[0])  
                else:
                    action_index = agent.choose_action(state, goal_y=t[1])

                action = "0 " + action_commands[action_index]

                # print("Sending action: ", action)
                sock_game.send(str.encode(action))  # send action to env

                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]


                # Define the reward based on the state and next_state
                reward_x, reward_y, reward_norms = calculate_reward(state, next_state, t[0], t[1], norms)  # You need to define this function
                # print("------------------------------------------")
                # print(reward_x, reward_y, min_x, min_y, action_commands[action_index], state['observation']['players'][0]['position'], next_state['observation']['players'][0]['position'])    
                # print("------------------------------------------")
                # Update Q-table
                if reward_x < 0 and reward_y < 0:
                    reward_cnt += 1
                
                if reward_cnt > 20:
                    reward_cnt = 0
                    switch = 1 - switch

                agent.learning(action_index,  state, next_state, t[0], t[1], reward_x, reward_y, reward_norms)

                # Update state
                state = next_state
                current_dis_x = abs(state['observation']['players'][0]['position'][0] - t[0])
                current_dis_y = abs(state['observation']['players'][0]['position'][1] - t[1])
                current_dis = math.sqrt((current_dis_x * current_dis_x) + (current_dis_y * current_dis_y))
                distance_travelled = og_dis - current_dis
                print("------------------------------------------")
                print("Looking [", t[0], ",", t[1], "]")
                print("Current location:", "[", state['observation']['players'][0]['position'][0], ",", state['observation']['players'][0]['position'][1], "]")
                print("Training time:", i)
                print("Percent travelled:")
                print((distance_travelled / og_dis)* 100, "%")
                print("------------------------------------------")


                if cnt > episode_length or (abs(state['observation']['players'][0]['position'][0] - t[0]) < 0.2 and abs(state['observation']['players'][0]['position'][1] - t[1]) < 0.2):
                    print("------------------------------------------")
                    if(cnt > episode_length):
                        print("Time exceeded")
                    else:
                        print("Reached goal")
                        print(t)
                    print("------------------------------------------")
                    break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()