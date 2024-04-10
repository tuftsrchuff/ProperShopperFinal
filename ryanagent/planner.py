#Author: Ryan Huffnagle

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv

from Q_Learning_agent_v2 import QLAgent  # Make sure to import your QLAgent class
import pandas as pd
import time
from utils import recv_socket_data

#Just for reference
# locations = {
#     'milk': [5.5, 3.6], 
#     'chocolate milk': [9.5, 3.6], 
#     'strawberry milk': [13.5, 3.6], 
#     'apples': [5.5, 7.6], 
#     'oranges': [7.5, 7.6], 
#     'banana': [9.5, 7.6], 
#     'strawberry': [11.5, 7.6], 
#     'raspberry': [13.5, 7.6], 
#     'sausage': [5.5, 11.6], 
#     'steak': [7.5, 11.6], 
#     'chicken': [11.5, 11.6], 
#     'ham': [13.5, 11.6], 
#     'brie cheese': [5.5, 15.6], 
#     'swiss cheese': [7.5, 15.6], 
#     'cheese wheel': [9.5, 15.6], 
#     'garlic': [5.5, 19.6], 
#     'leek': [7.5, 19.6], 
#     'red bell pepper': [9.5, 19.6], 
#     'carrot': [11.5, 19.6], 
#     'lettuce': [13.5, 19.6], 
#     'avocado': [5.5, 23.35], 
#     'broccoli': [7.5, 23.35], 
#     'cucumber': [9.5, 23.35], 
#     'yellow bell pepper': [11.5, 23.35], 
#     'onion': [13.5, 23.35],
#     "exit_pos": [-0.8, 15.6],
#     "cart_pos": [1.2, 17.7],
#     'basket_pos': [3.15, 17.8],
#     'checkout': [2.25, 13.0],
#     'prepared foods': [16.8, 5.5],
#     'fresh fish': [16.8, 11.7]
# }

exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 17.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 

class Planner:
    def __init__(self, shoppingList, listQuant, locations, socket):
        self.shoppingList = shoppingList
        self.listQuant = listQuant
        self.locations = locations
        self.socket = socket
        self.action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'RESET']
        self.action_space = len(self.action_commands) - 1
        self.agent =  QLAgent(self.action_space)
        self.agent.qtable_norms = pd.read_json('qtable_norms.json')
        self.agent.qtable_x = pd.read_json('qtable_x.json')
        self.agent.qtable_y = pd.read_json('qtable_y.json')
        self.replay_buffer = []
        self.cart_replay_buffer = []
        self.shelves = {}
        self.populateShelves()
        self.hasCart = True

    #Populate shelf locations from environment for navigation to them
    def populateShelves(self):
        self.socket.send(str.encode("0 NOP"))  # send action to env

        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)
        shelves = state["observation"]["shelves"]
        for shelf in shelves:
            if shelf["food_name"] not in self.shelves:
                self.shelves[shelf["food_name"]] = shelf["position"]
        
        print(self.shelves)

    def euclidean_distance(self, pos1, pos2):
        # Calculate Euclidean distance between two points
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def executePlan(self):
        #Execute shopper for full shopping list and select basket or cart

        #List that works well for demonstration, full shopping list can fail
        # self.shoppingList = ['sausage', 'prepared foods', 'broccoli']
        # self.listQuant = [2,3,1]
        print("Full shopping list")
        print(self.shoppingList)
        print("List quantity")
        print(self.listQuant)
        totalList = sum(self.listQuant)
        print(f"List quantity {totalList}")
        if totalList < 6:
            self.hasCart = False
            self.get("basket")
        else:
            self.hasCart = True
            self.get("cart")
        while self.shoppingList:
            #Grab each item then return home and remove from list
            print(f"Grabbing item {self.shoppingList[-1]}")
            # time.sleep(5)
            self.goTo(self.locations[self.shoppingList[-1]])
            self.leaveCartAndInteract(food=self.shoppingList[-1], num=self.listQuant[-1])
            self.goHome()
            self.shoppingList.pop()
            self.listQuant.pop()
        
        self.checkout()
        self.exitStore()
        
        print("Go to checkout and exit")
        

        
    def checkout(self):
        #Hard coded vals to move from home to checkout and pay for items
        self.cart_replay_buffer = []

        #Cart vs basket specific distances
        if self.hasCart:
            moveX = 6
            moveY = 7
        else:
            moveX = 0
            moveY = 6
        state = {}
        for _ in range(18):
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
        
        self.socket.send(str.encode("0 TOGGLE_CART"))
        

        for _ in range(moveX):
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
            self.cart_replay_buffer.append("EAST")
        
        for _ in range(moveY):
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
            self.cart_replay_buffer.append("NORTH")
        
        for _ in range(3):
            self.socket.send(str.encode("0 INTERACT"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)

        
        #Replay buffer to replay all actions        
        self.replayActions(self.cart_replay_buffer)


        #Return to cart after payment
        if self.hasCart:
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
        else:
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
        
        self.socket.send(str.encode("0 TOGGLE_CART"))
        state = recv_socket_data(self.socket)
        state = json.loads(state)

    
    def exitStore(self):
        #Hard coded vals to go from checkout to exit the store
        for _ in range(17):
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
        
        for _ in range(40):
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
        
        for _ in range(30):
            self.socket.send(str.encode("0 WEST"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)


    def goHome(self):
        #Use replay buffer to go home using the previous actions
        self.replayActions(self.replay_buffer)
        self.socket.send(str.encode("0 NOP"))
        state = recv_socket_data(self.socket)
        state = json.loads(state)
        
        direction = state['observation']['players'][0]['direction']
        
        if direction != 2:
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)

        #Overcome minor differences in return to home actions
        self.recoverToHomePos()

    def replayActions(self, buffer):
        #Take in buffer and reverse all actions to replay
        currAction = None
        state = {}
        mapDir = {
            "NORTH": "SOUTH",
            "SOUTH": "NORTH",
            "WEST": "EAST",
            "EAST": "WEST"
        }
        while len(buffer) > 0:
            prevAction = buffer.pop()
            currAction = mapDir[prevAction]
            currAction = "0 " + currAction
            self.socket.send(str.encode(currAction))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
        
        


    def goTo(self, location):
        #Use RL agent to go to location
        self.replay_buffer = []
        goalDest = location
        self.socket.send(str.encode("0 NOP"))
        state = recv_socket_data(self.socket)
        state = json.loads(state)

        print(state["observation"]["players"][0]["position"])
        # time.sleep(5)

        #RL agent go to function
        while not state['gameOver']:
            # Choose an action based on the current state
            action_index = self.agent.choose_action(state, goal_x = goalDest[0], goal_y = goalDest[1], train=False)
            action = "0 " + self.action_commands[action_index]
            self.socket.send(str.encode(action))  # send action to env
            self.replay_buffer.append(self.action_commands[action_index])

            next_state = recv_socket_data(self.socket)  # get observation from env
            next_state = json.loads(next_state)

            # Update state
            state = next_state
            # agent.qtable.to_json('qtable.json')
            # End episode when agent within 0.2 distance from goal
            playerPosCurr = state["observation"]["players"][0]["position"]
            if self.euclidean_distance(playerPosCurr, goalDest) < 0.2:
                print("Goal reached!")
                break

    def get(self, cartType):
        #Get either the cart or basket
        location = [0,0]
        #Grab basket or cart depending on size
        if cartType == "basket":
            moveX = 14
            moveY = 16
        else:
            moveX = 0
            moveY = 16
        
        print(f"Going to {location}")
        for _ in range(moveX):
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
        
        for _ in range(moveY):
            self.socket.send(str.encode("0 SOUTH"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
        
        self.socket.send(str.encode("0 INTERACT"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)

        for _ in range(moveY):
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
        
        for _ in range(moveX):
            self.socket.send(str.encode("0 WEST"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

    def leaveCartAndInteract(self, food=None, num = 1):
        #Interact with either shelf or counter item to put them into basket/cart
        self.cart_replay_buffer = []

        if food in ['fresh fish', 'prepared foods']:
            #Side counter, needs specific plan
            for _ in range(num):
                print(f"Round {_}")
                print(self.hasCart)
                self.grabCounterItem()
            return

        #Plan changes whether there is a basket or a cart
        if not self.hasCart:
            for _ in range(4):
                self.socket.send(str.encode("0 EAST"))
                state = recv_socket_data(self.socket)  # get observation from env
                state = json.loads(state)
                self.replay_buffer.append("EAST")
        
        self.socket.send(str.encode("0 NOP"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)
        direction = state['observation']['players'][0]['direction']
    
        if direction != 0:
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)
            state = json.loads(state)
        
        for _ in range(num):
            self.grabItem(food)
        
        
    def grabCounterItem(self):
        #Actions specific for counter item pickup
        self.socket.send(str.encode("0 NOP"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)
    
        #Actions differ for cart and basket pickup from counter
        if not self.hasCart:
            if state['observation']['players'][0]['direction'] != 3:
                self.socket.send(str.encode("0 WEST"))
                state = recv_socket_data(self.socket)
                state = json.loads(state)
            
            self.socket.send(str.encode("0 TOGGLE_CART"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

            #Move to location of counter where agent can grab item
            while state["observation"]["players"][0]["position"][0] < 17.25:
                self.socket.send(str.encode("0 EAST"))
                state = recv_socket_data(self.socket)
                state = json.loads(state)
                self.cart_replay_buffer.append("EAST")
        
        else:
            if state['observation']['players'][0]['direction'] != 0:
                self.socket.send(str.encode("0 NORTH"))
                state = recv_socket_data(self.socket)
                state = json.loads(state)
               
            
            self.socket.send(str.encode("0 TOGGLE_CART"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

            #Move to location of counter where agent can grab item
            while state["observation"]["players"][0]["position"][0] < 17.25:
                self.socket.send(str.encode("0 EAST"))
                state = recv_socket_data(self.socket)
                state = json.loads(state)
                self.cart_replay_buffer.append("EAST")
        
        self.socket.send(str.encode("0 INTERACT"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)

        #Keep attempting to grab food until success state
        while state["observation"]["players"][0]['holding_food'] is None:
            self.socket.send(str.encode("0 INTERACT"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

        #Go back to cart using replay buffer
        self.replayActions(self.cart_replay_buffer)

        if not self.hasCart:
            while state['observation']['players'][0]['direction'] != 3:
                self.socket.send(str.encode("0 WEST"))
                state = recv_socket_data(self.socket)
                state = json.loads(state)
        else:
            print("Trying to turn north")
            while state['observation']['players'][0]['direction'] != 0:
                print("In while loop")
                self.socket.send(str.encode("0 NORTH"))
                state = recv_socket_data(self.socket)
                state = json.loads(state)
                print("Cart and now turning NORTH")


        #Keep trying to grab food until agent gets it, in while loop because action fails sometimes
        while state["observation"]["players"][0]['holding_food'] is not None:
            self.socket.send(str.encode("0 INTERACT"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
            direction = state['observation']['players'][0]['direction']
            print(state["observation"]["players"][0]["position"])
            print(self.hasCart)
        
            
        self.socket.send(str.encode("0 TOGGLE_CART"))
        state = recv_socket_data(self.socket)
        state = json.loads(state)


    
    def grabItem(self, food):
        #Grabbing item from the shelf 
        self.socket.send(str.encode("0 TOGGLE_CART"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)

        
        if self.hasCart:
            toX = state["observation"]["players"][0]["position"][0] + 0.6
            toY = self.shelves[food][1] + 1.4
            self.grabFromShelf(state, toX, toY)

            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

        else:
            toX = state["observation"]["players"][0]["position"][0]
            toY = self.shelves[food][1] + 1.4
            self.grabFromShelf(state, toX, toY)

            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

        for _ in range(2):
                self.socket.send(str.encode("0 INTERACT"))
                state = recv_socket_data(self.socket)  # get observation from env
                state = json.loads(state)

        self.socket.send(str.encode("0 TOGGLE_CART"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)

        if not self.hasCart:
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

    def grabFromShelf(self, state, toX, toY):
        #Specific grab action that takes in amount of X and Y to move, agnostic to cart
        #or basket
        while state["observation"]["players"][0]["position"][0] < toX:
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
            self.cart_replay_buffer.append("EAST")
        
        while state["observation"]["players"][0]["position"][1] > toY:
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
            self.cart_replay_buffer.append("NORTH")
        
        self.socket.send(str.encode("0 INTERACT"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)

        #Keep attempting to grab food until success state
        while state["observation"]["players"][0]['holding_food'] is None:
            self.socket.send(str.encode("0 INTERACT"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)

        #Go back to cart using opposite of previous actions
        self.replayActions(self.cart_replay_buffer)
    
    def recoverToHomePos(self):
        #More sturdy return to home position, accounts for minor differences
        #in path backwards
        self.socket.send(str.encode("0 NOP"))
        state = recv_socket_data(self.socket)  # get observation from env
        state = json.loads(state)

        posX = state["observation"]["players"][0]["position"][0]
        posY = state["observation"]["players"][0]["position"][1]

        #Less than X
        while posX < 1.15:
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
            posX = state["observation"]["players"][0]["position"][0]
        
        while posX > 1.25:
            self.socket.send(str.encode("0 WEST"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
            posX = state["observation"]["players"][0]["position"][0]
        
        while posY < 15.55:
            self.socket.send(str.encode("0 SOUTH"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
            posY = state["observation"]["players"][0]["position"][1]
        
        while posY > 15.65:
            self.socket.send(str.encode("0 NORTH"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)
            posY = state["observation"]["players"][0]["position"][1]

        direction = state['observation']['players'][0]['direction']
        if direction != 2:
            self.socket.send(str.encode("0 EAST"))
            state = recv_socket_data(self.socket)  # get observation from env
            state = json.loads(state)