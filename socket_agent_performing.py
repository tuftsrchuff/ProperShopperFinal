#Author: Hang Yu
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


x_values = [
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    6, 8, 10, 12, 14,
    17.5, 17.5
]


first_aisle_values = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    0, 0
]


second_aisle_values = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    5, 5, 5, 5, 5,
    0, 0
]

shelf_values = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    6, 6, 6, 6, 6,
    0, 0
]

aisle_y_values = [3.75, 7.8, 11.7, 15.6, 19.65, 23, 8.5, 14.5]

reached_item = False
need_cart = False


grocery_list = []
for item, x, aisle1, aisle2, shelf in zip(grocery_items, x_values, first_aisle_values, second_aisle_values, shelf_values):
    grocery_list.append({
        'item': item,
        'x': x,
        'first_aisle': aisle1,
        'second_aisle': aisle2,
        'shelf': shelf
    })


def generate_shopping_info(shopping_list, quantities):
    shopping_info = []
    for index, item in enumerate(shopping_list):
        for grocery_item in grocery_list:
            if item.lower() == grocery_item['item'].lower():
                for i in range(quantities[index]):
                    shopping_info.append({
                        'item': item,
                        'x': grocery_item['x'],
                        'first_aisle': grocery_item['first_aisle'],
                        'second_aisle': grocery_item['second_aisle'],
                        'shelf': grocery_item['shelf']
                    })
                break
    return shopping_info


# This function finds the southmost aisle in a given shopping list
def find_max_first_aisle(shopping_list):
    max_first_aisle = float('-inf')  # Initialize with negative infinity to handle empty lists
    for item_info in shopping_list:
        first_aisle = item_info['first_aisle']
        if first_aisle > max_first_aisle:
            max_first_aisle = first_aisle
    return max_first_aisle


# OPTIMIZING PATH FUNCTION
def find_target_list(state):
    shopping_list = state['observation']['players'][0]['shopping_list']
    quantities = state['observation']['players'][0]['list_quant']
    shopping_list = generate_shopping_info(shopping_list, quantities)
    print(shopping_list)

    sorted_list = []
    current_aisle = find_max_first_aisle(shopping_list)
    aisles_needed = [current_aisle]

    while shopping_list:
        delete_list = []
        for t in shopping_list:
            print("Current aisle:", current_aisle)
            # If the item is in the aisle we're in, add it
            if(t['first_aisle'] == current_aisle or t['second_aisle'] == current_aisle):
                sorted_list.append({
                        'item': t['item'],
                        'x': t['x'],
                        'aisle': current_aisle,
                        'shelf': t['shelf']
                    })
                print("Adding", t["item"], "for aisle", current_aisle)
                delete_list.append(t)
            else:
                print("Skipping", t["item"], "because", t['first_aisle'], "and", t['second_aisle'], "isn't the same as", current_aisle)
        while delete_list:
            shopping_list.remove(delete_list[0])
            delete_list.remove(delete_list[0])
        current_aisle = find_max_first_aisle(shopping_list)
        aisles_needed.append(current_aisle)

    # This defines the snaking behavior
    def custom_sort(item):
        aisle = aisles_needed.index(item['aisle'])
        x = item['x']
        # If it's the first/third/fifth aisle we visit, go left to right. Second/fourth, go right to left
        if aisle % 2 == 0:  
            return (-item['aisle'], x)  # Sort ties by x value in ascending order
        else:  # If second aisle is odd
            return (-item['aisle'], -x)  # Sort ties by x value in descending order
            

    # Sort the shopping list using the custom sorting function
    sorted_list = sorted(sorted_list, key=custom_sort)
    
    print(sorted_list)
    last_aisle = 10
    for s in sorted_list:
        if(last_aisle != s['aisle']):
            last_aisle = s['aisle']
    return sorted_list
    


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
    training_time = 30
    episode_length = 1000
    for i in range(training_time):
        global min_x
        global min_y
        min_x = 1000
        min_y = 1000
        reached_item = False
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

        # First of all, get the optimized shopping list
        target_list = find_target_list(state)
        # If we have enough items, we need a cart
        if(len(target_list) > 6):
            need_cart = True

        #Get a cart
        if(need_cart):
            print("Going to get a cart")
            # Go right a few
            for i in range(2):
                sock_game.send(str.encode("0 " + action_commands[2]))
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]

            # Go down a bit
            for i in range(10):
                sock_game.send(str.encode("0 " + action_commands[1]))
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]
                print("Going down", i)

            sock_game.send(str.encode("0 INTERACT")) # get cart
            print("Getting cart!")
            output = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(output)
            norms = next_state["violations"]
            sock_game.send(str.encode("0 INTERACT"))
            output = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(output)
            norms = next_state["violations"]
        else:
            print("Going to get a basket")
            # Go right a few
            for i in range(7):
                sock_game.send(str.encode("0 " + action_commands[2]))
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]

            # Go down a bit
            for i in range(10):
                sock_game.send(str.encode("0 " + action_commands[1]))
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]
                print("Going down", i)

            sock_game.send(str.encode("0 INTERACT")) # get item
            print("GETTING ITEM!")
            output = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(output)
            norms = next_state["violations"]
            sock_game.send(str.encode("0 INTERACT")) # get item
            print("GETTING ITEM!")
            output = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(output)
            norms = next_state["violations"]

        # Now we can start shopping
        for t in target_list:
            switch = random.randint(0, 1)

            t_x = t['x']
            if t['item'] == "prepared foods":
                t_y = aisle_y_values[6]
            elif t['item'] == "fresh fish":
                t_y = aisle_y_values[7]
            else:
                t_y = aisle_y_values[t['aisle'] - 1]
            t_y_actual = (t['shelf'] - 1) * 4 + 1.5

            counter_time = t['aisle'] == 0


            while not state['gameOver']:
                cnt += 1

                print (target_list)
                
                # Choose an action based on the current state
                if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2:
                    switch = 1
                if abs(state['observation']['players'][0]['position'][1] - t_y) < 0.2:
                    switch = 0 
                # If we've reached it, then it's time to grab it
                if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2 and abs(state['observation']['players'][0]['position'][1] - t_y) < 0.2:
                    switch = 2
                
                # If we're going to a food counter and we've reached it
                if counter_time:
                    if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2 and abs(state['observation']['players'][0]['position'][1] - t_y) < .2:
                        switch = 3
                        print("We have to go to a counter")

                if switch == 0:
                    action_index = agent.choose_action(state, goal_x=t_x)  
                if switch == 1:
                    action_index = agent.choose_action(state, goal_y=t_y)
                
                if switch == 0 or switch == 1:
                    action = "0 " + action_commands[action_index]
                    # print("Sending action: ", action)
                    sock_game.send(str.encode(action))  # send action to env
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]


                # Grab the item!
                if switch == 2:
                    print("------------------------------------------")
                    print("RETRIEVING:", t['item'])

                    # Go right one
                    sock_game.send(str.encode("0 " + action_commands[2]))
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    # Drop cart
                    if need_cart:
                        sock_game.send(str.encode("0 TOGGLE_CART"))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        print("Current y:", next_state['observation']['players'][0]['position'][1])

                    num_moves = 5
                    agent_position_y = next_state['observation']['players'][0]['position'][1]
                    agent_below = agent_position_y > t_y_actual
                    if(agent_below):
                        # Go up
                        for _ in range(num_moves):
                            sock_game.send(str.encode("0 " + action_commands[0]))
                            print("Going up!")
                            output = recv_socket_data(sock_game)  # get observation from env
                            next_state = json.loads(output)
                            norms = next_state["violations"]
                            print("Current y:", next_state['observation']['players'][0]['position'][1])
                    else:
                        # Go down
                        for _ in range(num_moves):
                            sock_game.send(str.encode("0 " + action_commands[1]))
                            print("Going down!")
                            output = recv_socket_data(sock_game)  # get observation from env
                            next_state = json.loads(output)
                            norms = next_state["violations"]
                            print("Current y:", next_state['observation']['players'][0]['position'][1])
                    
                    sock_game.send(str.encode("0 INTERACT")) # get item
                    print("GETTING ITEM!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    print("Putting item in cart!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    if(agent_below):
                        # Go up
                        for _ in range(num_moves):
                            sock_game.send(str.encode("0 " + action_commands[1]))
                            print("Going down!")
                            output = recv_socket_data(sock_game)  # get observation from env
                            next_state = json.loads(output)
                            norms = next_state["violations"]
                            print("Current y:", next_state['observation']['players'][0]['position'][1])
                    else:
                        # Go down
                        for _ in range(num_moves):
                            sock_game.send(str.encode("0 " + action_commands[0]))
                            print("Going up!")
                            output = recv_socket_data(sock_game)  # get observation from env
                            next_state = json.loads(output)
                            norms = next_state["violations"]
                            print("Current y:", next_state['observation']['players'][0]['position'][1])

                    sock_game.send(str.encode("0 " + action_commands[2]))
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    print("Putting item in cart!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    if need_cart:
                        sock_game.send(str.encode("0 TOGGLE_CART")) # get item
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        print("------------------------------------------")
                    reached_item = True

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    print("Putting item in cart!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                # FOOD COUNTERS
                if switch == 3:
                    print("------------------------------------------")
                    print("RETRIEVING:", t['item'])

                    # Go left one
                    sock_game.send(str.encode("0 " + action_commands[3]))
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    # Drop cart
                    if need_cart:
                        sock_game.send(str.encode("0 TOGGLE_CART"))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        print("Current y:", next_state['observation']['players'][0]['position'][1])

                    num_moves = 6
                    agent_position_y = next_state['observation']['players'][0]['position'][1]
                    # Go up
                    for _ in range(num_moves):
                        sock_game.send(str.encode("0 " + action_commands[0]))
                        print("Going up!")
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        print("Current y:", next_state['observation']['players'][0]['position'][1])
                    
                    for _ in range(2):
                        sock_game.send(str.encode("0 " + action_commands[2]))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                    
                    sock_game.send(str.encode("0 INTERACT")) # get item
                    print("GETTING ITEM!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    # Go down
                    for _ in range(num_moves):
                        sock_game.send(str.encode("0 " + action_commands[1]))
                        print("Going down!")
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        print("Current y:", next_state['observation']['players'][0]['position'][1])

                    sock_game.send(str.encode("0 " + action_commands[3]))
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 INTERACT")) # put item away
                    print("Putting item in cart!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    if need_cart:
                        sock_game.send(str.encode("0 TOGGLE_CART")) # get item
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        print("------------------------------------------")
                    reached_item = True

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    print("Putting item in cart!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                print("------------------------------------------")
                print("Looking for:", t['item'], "located at [", t_x, ",", t_y, "]")
                print("Current location:", "[", state['observation']['players'][0]['position'][0], ",", state['observation']['players'][0]['position'][1], "]")
                print("Training time:", i)
                print("x difference:", state['observation']['players'][0]['position'][0] - t_x)
                print("y difference:", state['observation']['players'][0]['position'][1] - t_y)
                print("Percent travelled:")
                # print((distance_travelled / og_dis)* 100, "%")
                print("------------------------------------------")

                # Update state
                state = next_state


                if cnt > episode_length:
                    break
                elif reached_item:
                    reached_item = False
                    break
        # Additional code for end of episode if needed
                
        # CHECK OUT TIME
        while not state['gameOver']:
            switch = random.randint(0, 1)
            t_x = 3.5
            t_y = 12.5

            # Choose an action based on the current state
            if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2:
                switch = 1
            if abs(state['observation']['players'][0]['position'][1] - t_y) < 0.2:
                switch = 0 
            if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2 and abs(state['observation']['players'][0]['position'][1] - t_y) < 0.2:
                switch = 2

            if switch == 0:
                action_index = agent.choose_action(state, goal_x=t_x)  
            if switch == 1:
                action_index = agent.choose_action(state, goal_y=t_y)
            
            if switch == 0 or switch == 1:
                action = "0 " + action_commands[action_index]
                # print("Sending action: ", action)
                sock_game.send(str.encode(action))  # send action to env
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]

            if switch == 2:
                # Go up some
                    for _ in range(4):
                        sock_game.send(str.encode("0 " + action_commands[0]))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]

                    sock_game.send(str.encode("0 " + action_commands[1]))
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    # Drop cart
                    if need_cart:
                        sock_game.send(str.encode("0 TOGGLE_CART"))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        print("Current y:", next_state['observation']['players'][0]['position'][1])

                    
                    for _ in range(1):
                        sock_game.send(str.encode("0 " + action_commands[3]))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                    
                    sock_game.send(str.encode("0 INTERACT")) # get item
                    print("GETTING ITEM!")
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 INTERACT")) # get item
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    sock_game.send(str.encode("0 " + action_commands[2]))
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]

                    if need_cart:
                        sock_game.send(str.encode("0 " + action_commands[1]))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        sock_game.send(str.encode("0 " + action_commands[1]))
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                        sock_game.send(str.encode("0 TOGGLE_CART")) # get item
                        output = recv_socket_data(sock_game)  # get observation from env
                        next_state = json.loads(output)
                        norms = next_state["violations"]
                    reached_item = True

            print("------------------------------------------")
            print("Checking out: at [", t_x, ",", t_y, "]")
            print("Current location:", "[", state['observation']['players'][0]['position'][0], ",", state['observation']['players'][0]['position'][1], "]")
            print("x difference:", state['observation']['players'][0]['position'][0] - t_x)
            print("y difference:", state['observation']['players'][0]['position'][1] - t_y)
            print("Training time:", i)
            print("Percent travelled:")
            print("Switch:", switch)
            # print((distance_travelled / og_dis)* 100, "%")
            print("------------------------------------------")

            # Update state
            state = next_state
            if(reached_item):
                reached_item = False
                switch = random.randint(0, 1)
                break

        # CART/BASKET RETURN TIME
        while not state['gameOver']:
            if need_cart:
                t_x = 1.5
            else:
                t_x = 3
            t_y = 18

            # Choose an action based on the current state
            if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2:
                switch = 1
            if abs(state['observation']['players'][0]['position'][1] - t_y) < 0.2:
                switch = 0 
            if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2 and abs(state['observation']['players'][0]['position'][1] - t_y) < 1:
                switch = 2

            if switch == 0:
                action_index = agent.choose_action(state, goal_x=t_x)  
            if switch == 1:
                action_index = agent.choose_action(state, goal_y=t_y)
            
            if switch == 0 or switch == 1:
                action = "0 " + action_commands[action_index]
                # print("Sending action: ", action)
                sock_game.send(str.encode(action))  # send action to env
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]

            if switch == 2:
                sock_game.send(str.encode("0 " + action_commands[1]))
                print("Going down!")
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]
                print("Current y:", next_state['observation']['players'][0]['position'][1])

                sock_game.send(str.encode("0 INTERACT")) # get item
                print("Putting cart away!")
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]

                sock_game.send(str.encode("0 INTERACT")) # get item
                print("Putting cart away!")
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]
                reached_item = True

            print("------------------------------------------")
            print("Putting cart back: at [", t_x, ",", t_y, "]")
            print("Current location:", "[", state['observation']['players'][0]['position'][0], ",", state['observation']['players'][0]['position'][1], "]")
            print("x difference:", state['observation']['players'][0]['position'][0] - t_x)
            print("y difference:", state['observation']['players'][0]['position'][1] - t_y)
            print("Training time:", i)
            print("Percent travelled:")
            # print((distance_travelled / og_dis)* 100, "%")
            print("------------------------------------------")

            # Update state
            state = next_state
            if(reached_item):
                reached_item = False
                switch = random.randint(0, 1)
                break

        # TIME TO LEAVE
        while not state['gameOver']:
            print("Time to exit the store")
            t_x = exit_pos[0]
            t_y = exit_pos[1]
            # Choose an action based on the current state
            if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2:
                switch = 1
            if abs(state['observation']['players'][0]['position'][1] - t_y) < 0.2:
                switch = 0 
            if abs(state['observation']['players'][0]['position'][0] - t_x) < 0.2 and abs(state['observation']['players'][0]['position'][1] - t_y) < 0.2:
                switch = 2

            if switch == 0:
                action_index = agent.choose_action(state, goal_x=t_x)  
            if switch == 1:
                action_index = agent.choose_action(state, goal_y=t_y)
            
            if switch == 0 or switch == 1:
                action = "0 " + action_commands[action_index]
                # print("Sending action: ", action)
                sock_game.send(str.encode(action))  # send action to env
                output = recv_socket_data(sock_game)  # get observation from env
                next_state = json.loads(output)
                norms = next_state["violations"]

            if switch == 2:
                for _ in range(3):
                    action = "0 " + action_commands[3]
                    # print("Sending action: ", action)
                    sock_game.send(str.encode(action))  # send action to env
                    output = recv_socket_data(sock_game)  # get observation from env
                    next_state = json.loads(output)
                    norms = next_state["violations"]
                    reached_item = True

            print("------------------------------------------")
            print("Exiting the store: at [", t_x, ",", t_y, "]")
            print("Current location:", "[", state['observation']['players'][0]['position'][0], ",", state['observation']['players'][0]['position'][1], "]")
            if state['observation']['players'][0]['position'][0] - t_x < 0.2:
                print("X IN RANGE")
            print("x difference:", state['observation']['players'][0]['position'][0] - t_x)
            if state['observation']['players'][0]['position'][1] - t_y < 0.2:
                print("Y IN RANGE")
            print("y difference:", state['observation']['players'][0]['position'][1] - t_y)
            print("Training time:", i)
            print("Percent travelled:")
            # print((distance_travelled / og_dis)* 100, "%")
            print("------------------------------------------")

            # Update state
            state = next_state

            if reached_item:
                break

    # Close socket connection
    sock_game.close()




