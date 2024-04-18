#Author: Ryan Huffnagle

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent_v2 import QLAgent  # Make sure to import your QLAgent class
import pandas as pd
from planner import Planner

locations = {
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
    "exit_pos": [-0.8, 15.6],
    "cart_pos": [1.2, 17.7],
    'basket_pos': [3.15, 17.8],
    'checkout': [2.25, 13.0],
    'prepared foods': [16.8, 5.5],
    'fresh fish': [16.8, 11.7]
}


if __name__ == "__main__":
    
    #Just set up socket and pass in shopping and list quantity to planner
    action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'RESET']
    # rand_actions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)
    agent.qtable_norms = pd.read_json('qtable_norms.json')
    agent.qtable_x = pd.read_json('qtable_x.json')
    agent.qtable_y = pd.read_json('qtable_y.json')

    playerNum = input("Player number")
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    sock_game.send(str.encode(playerNum + " RESET"))  # reset the game

    output = recv_socket_data(sock_game)  # get observation from env
    output = json.loads(output)
    
    shoppingList = output["observation"]["players"][0]["shopping_list"]
    listQuant = output["observation"]["players"][0]["list_quant"]
    planner = Planner(shoppingList, listQuant, locations, sock_game, playerNum)
    planner.executePlan()

        

