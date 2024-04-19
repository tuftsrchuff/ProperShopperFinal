# TODO:I ran a few trails, and it seems quite good, but since a lot of things are 
# hard-coded, it would be best if I could do a better error-handling part.
# sadly the deadline is coming :-(
# so for now, the whole operation would just fail and raise an Error to help you know which part it failed


# This file have all the shopping behaviors
# Shopping behavior is achived through combining RL and hard-coded actions
# basically, the navigation to critical positions are done through RL
# and other things, such as approaching, interacting etc. are hard-coded
import numpy as np
from trainer import*
import yaml
from utils import recv_socket_data
import copy


# This is the RL agent for navigating to critical objects
# The RL part just reach a place near the critical object, then hard-coded actions would take over
# Suppose we have a shelf eith multiple goods, then one of them would be count as "critical"
# List of critical object and posisions near them can be found in TARGET_CART_POSITION in config.yaml
# trainer.py file is responsible for training
class navigation_operator:
    def __init__(self, sock_game:socket.socket,target_object, config_file_path:str) -> None:
    
        with open(config_file_path) as f:
            yaml_data = yaml.safe_load(f)

        self.sock_game = sock_game

        # get target pose
        self.target_object = target_object
        self.target_pose = yaml_data["TARGET_CART_POSITION"][self.target_object]

        # load trained Q table
        self.qtable = np.load('trained_qtable_{}_{},{}.npy'.format(target_object,self.target_pose[0],self.target_pose[1]))

        self.steps_limit = yaml_data["navigation_steps_limit"]

        self.granularity = yaml_data["granularity"]
        self.single_action_change = 0.15
        self.navigation_accuracy = yaml_data["navigation_accuracy"]

        self.action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        self.action_space = len(self.action_commands)

        self.agent = QLAgent(self.action_space,granularity=self.granularity)
        self.agent.q_table = copy.deepcopy(self.qtable)
        # turn off the exploration
        self.agent.epsilon = 0

        # even if the agent is already trained, we still might want
        # the reward functions and keep the agent training while doing the 
        # naviagtion, since it could increse robustness and prevent the agent 
        # from being trapped in a loop
        self.reward_table = initialize_reward_table(self.granularity, self.target_pose)

    def reach_target_checker(self, state):
        agent_position = state['observation']['players'][0]['position']
        if euclidean_distance(self.target_pose, agent_position) <= self.navigation_accuracy:
            # print("successfully reach {}".format(self.target_object))
            return True
        else: return False

    # carry out the navigation
    # return -1 for navigation failure, 1 for success 
    def do_navigation(self):

        # send a NOP action to read current state
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)

        step_cnt = 0
        total_violations = 0

        while not state['gameOver']:

            if step_cnt > self.steps_limit:
                return -1
            
            if check_leaving(state['observation']['players'][0]['position']):
                raise RuntimeError("Navigation Error: agent tries to go out when navigating")
    
            # get action
            action_index = self.agent.choose_action(state)
            action = "0 " + self.action_commands[action_index]

            # moving
            current_step_violations = 0
            for _ in range(int(self.granularity/self.single_action_change)):
                self.sock_game.send(str.encode(action))
                next_state = recv_socket_data(self.sock_game)
                next_state = json.loads(next_state)
                if len(next_state["violations"]) != 0:
                    current_step_violations += len(next_state["violations"])
                total_violations += current_step_violations
                time.sleep(0.005)

            if self.reach_target_checker(next_state):
                print("successfullly reach {} in {} steps, with {} norm violations".format(self.target_object, step_cnt, total_violations))
                return 1
            reward = calculate_reward(self.agent, state, next_state,self.target_pose, action_index,self.reward_table,current_step_violations)
            self.agent.learning(action_index, reward, state, next_state)

            state = next_state
            step_cnt += 1

# we don't need a separate navigation to every target pos
# we can just move in one direction to go to shelves in a line
class simple_movement_operator:
    def __init__(self, sock_game:socket.socket, target_object, cart_or_basket,config_file_path):
        # assert target_object in TARGET_SIMPLE_CART_RELATIVE_RELATION, "simple_movement_operator: invalid target object {}".format(target_object)
        self.sock_game = sock_game
        self.target_object = target_object
        self.cart_or_basket = cart_or_basket
        self.config_file_path = config_file_path

        with open(config_file_path) as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)

        self.target_pos = yaml_data["SIMPLE_MOVE_OBJECTS"][target_object]

        # get the pose of the "critical" poses that we need navigte to before simple moves
        for prior_target in yaml_data["TARGET_SIMPLE_CART_RELATIVE_RELATION"].keys():
            if target_object in yaml_data["TARGET_SIMPLE_CART_RELATIVE_RELATION"][prior_target]:
                self.prior_target_pos = prior_target

        # It's just EAST, any non-shelf objects are not for simple move
        self.simple_move_direction = yaml_data["SIMPLE_MOVE_DIRECTION"]
        self.move_accuracy = 0.15

    # first we need to navigate to a position that can get to target position with simple move
    def do_simple_move(self):
        success_flag = 0

        # first navigate
        nav_op = navigation_operator(self.sock_game, self.prior_target_pos, self.config_file_path)
        success_flag = nav_op.do_navigation()
        if success_flag == -1:
            print("do_simple_move: navigation part failed!")
            return -1


        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        current_agent_pos = state['observation']['players'][0]['position']
        action = "0 " + self.simple_move_direction
        
        # just east, we care onl about x coordinate since we move horizontally
        if self.simple_move_direction == "EAST":
            axis_we_care = 0
        elif self.simple_move_direction == "SOUTH":
            axis_we_care = 1
        
        # move until reach target position
        while abs(current_agent_pos[axis_we_care] - self.target_pos[axis_we_care]) > self.move_accuracy:
            self.sock_game.send(str.encode(action))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            current_agent_pos = state['observation']['players'][0]['position']
        print("simple move: getting to {}, agent position{}".format(self.target_object, current_agent_pos))
        return 1

    # just for convenience in genral shopping part
    def do_navigation(self):
        self.do_simple_move()


# operator for the buying behavior after going near the object
# This operator also hands checkout procedure
# when navigated to a position near shelf: be on the correct direction
        # if shopping with cart:
            # 1. drop the cart, go to self, 
            # 2. pickup the target object, go back to cart, put target object in cart, 
            # 3. and pickup the cart
        # if shopping with basket:
            # 1. go to shelf
            # 2. interact to put things in basket
class purchase_operator:
    def __init__(self, sock_game:socket.socket, target_object, cart_or_basket,config_file_path) -> None:
        # assert target_object in DESIERED_DIRECTION, "purchase_operator: unknown target object {}".format(target_object)
        self.sock_game = sock_game
        self.target_object = target_object
        self.cart_or_basket = cart_or_basket
        with open(config_file_path) as f:
            self.yaml_data = yaml.load(f, Loader=yaml.FullLoader)

        # for some special goods, like counters and registers
        if self.target_object in self.yaml_data["WITH_SPECIAL_DIRECTION"]:
            self.initial_direction = self.yaml_data["WITH_SPECIAL_DIRECTION"][self.target_object][0]

        # for all goods on the shelves, the cart face south
        else:
            self.initial_direction = "SOUTH"
        self.interact_distance = 0.3
        self.max_try_time = 5
    
    # the function for purchase as a whole
    def do_purchase(self):

        # get current state
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        self.agent_pose_before_droping_cart = self.head_initial_direction()
        
        # determin how we are going to approach the object
        if self.initial_direction == "SOUTH":
            self.shelf_direction = "NORTH"
            axis_we_care = 1
        elif self.initial_direction == "WEST":
            self.shelf_direction = "EAST"
            axis_we_care = 0
        elif self.initial_direction == "EAST":
            self.shelf_direction = "WEST"

        # for object on the shelf, we minus 1 due to data in yaml was position of parking cart, and is 1 greter than the actuall shelf y pose
        if self.target_object not in self.yaml_data["WITH_SPECIAL_DIRECTION"]:
            if self.target_object in self.yaml_data["TARGET_CART_POSITION"].keys():
                shelf_interaction_pose = self.yaml_data["TARGET_CART_POSITION"][self.target_object][1] - 1
            else:
                shelf_interaction_pose = self.yaml_data["SIMPLE_MOVE_OBJECTS"][self.target_object][1] - 1

        # for registers and counts, we read the true interact pose from the yaml file, in this case, x coordinates
        else:
            shelf_interaction_pose = self.yaml_data["WITH_SPECIAL_DIRECTION"][self.target_object][1]

        # we pick up goods apart from checking out
        if self.target_object != "register":
            if self.cart_or_basket == "carts":
                self.pick_and_drop_cart()
                self.go_to_shelf(self.shelf_direction, shelf_interaction_pose, axis_we_care)
                self.pickup_goods() 
                self.put_good_to_cart(axis_we_care)
                self.pick_and_drop_cart()
            else:
                self.go_to_shelf(self.shelf_direction, shelf_interaction_pose, axis_we_care)
                self.pickup_goods() 
        # checking out is a different move
        elif self.target_object == "registers":
            self.checkout_good()

        return 1

# code for checkout, the code may looks werid due to some faulty thinking and bad code design :-(
    # checkout: 
    # if cart :1. park cart in interaction range of the register
    # 2. leave cart, head for the register
    # 3. interact to checkout
    # 4. return to cart
    # 5. pick up cart
    # if basket: just go to register and interact to checkout

    def checkout_good(self):

        action_north = "0 NORTH"
        action_south = "0 SOUTH"
        action_east = "0 EAST"
        action_west = "0 WEST"
        action_int = "0 INTERACT"

        shelf_interaction_pose = self.yaml_data["WITH_SPECIAL_DIRECTION"]["registers"][1]
        max_action_num = 30

        agent_checkout_pose = self.yaml_data["WITH_SPECIAL_DIRECTION"]["registers"][1:]
        # special actions for cart
        if self.cart_or_basket == "carts":
            self.sock_game.send(str.encode("0 NOP"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            disired_cart_pose = shelf_interaction_pose + self.yaml_data["extra_dist_with_cart"]["registers"]
            curr_cart_pose = state['observation']['carts'][0]['position']

            # push cart to interactive pose with the register
            cnt = 0
            while abs(curr_cart_pose[0] - disired_cart_pose) > self.interact_distance/2:
                self.sock_game.send(str.encode(action_west))
                state = recv_socket_data(self.sock_game)
                state = json.loads(state)
                curr_cart_pose = state['observation']['carts'][0]['position']
                if cnt >= max_action_num:
                    raise RuntimeError("can't push cart to interactive pose!")
                cnt += 1

            cnt = 0
            self.sock_game.send(str.encode("0 NOP"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            agent_pose_before_checkout = state['observation']['players'][0]['position']
            # drop the cart
            self.pick_and_drop_cart()
        
        # in case of basket we need to move west a few step to prevent from being stuck
        elif self.cart_or_basket == "baskets":
            cnt = 0
            self.sock_game.send(str.encode("0 NOP"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            curr_agent_pose = state['observation']['players'][0]['position']
            while abs(curr_agent_pose[0] - agent_checkout_pose[0]) >= self.interact_distance + 0.5:
                self.sock_game.send(str.encode(action_west))
                state = recv_socket_data(self.sock_game)
                state = json.loads(state)
                curr_agent_pose = state['observation']['players'][0]['position']
                if cnt >= max_action_num:
                    raise RuntimeError("can't go to counter!")
                cnt += 1


        # the agent going to register and checkout is same for both cart and bascket case
        cnt = 0
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        curr_agent_pose = state['observation']['players'][0]['position']
        

        # move upward and leftward for reaching register
        while abs(curr_agent_pose[1] - agent_checkout_pose[1]) >= self.interact_distance:
            self.sock_game.send(str.encode(action_north))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            curr_agent_pose = state['observation']['players'][0]['position']
            if cnt >= max_action_num:
                raise RuntimeError("can't go to counter!")
            cnt += 1
        
        cnt = 0
        while abs(curr_agent_pose[0] - agent_checkout_pose[0]) >= self.interact_distance:
            self.sock_game.send(str.encode(action_west))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            curr_agent_pose = state['observation']['players'][0]['position']
            if cnt >= max_action_num:
                raise RuntimeError("can't go to counter!")
            cnt += 1

        
        # interat to checkout
        cnt = 0
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        budget = state['observation']['players'][0]['budget']
        initial_budget = budget
        while budget == initial_budget:
            self.sock_game.send(str.encode(action_int))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            budget = state['observation']['players'][0]['budget']
            if cnt >= self.max_try_time:
                raise RuntimeError("can't do the checkout!")
            cnt += 1
        
        # goback and pick up if using carts
        if self.cart_or_basket == "carts":
            cnt = 0
            self.sock_game.send(str.encode("0 NOP"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            curr_agent_pose = state['observation']['players'][0]['position']

            # move right and down for going back to cart
            while curr_agent_pose[0] - agent_pose_before_checkout[0]  <= 0:
                self.sock_game.send(str.encode(action_east))
                state = recv_socket_data(self.sock_game)
                state = json.loads(state)
                curr_agent_pose = state['observation']['players'][0]['position']
                if cnt >= max_action_num:
                    raise RuntimeError("can't go back to cart!")
                cnt += 1
            
            cnt = 0
            
            while abs(curr_agent_pose[1] - agent_pose_before_checkout[1])  > self.interact_distance/2:
                self.sock_game.send(str.encode(action_south))
                state = recv_socket_data(self.sock_game)
                state = json.loads(state)
                curr_agent_pose = state['observation']['players'][0]['position']
                if cnt >= max_action_num:
                    raise RuntimeError("can't go back to cart!")
                cnt += 1
            
            # pickup the cart
            self.sock_game.send(str.encode(action_west))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            self.pick_and_drop_cart()

            
    # for the agent to face to correct direction before doing anything else
    def head_initial_direction(self):
        action = "0 " + self.initial_direction
        self.sock_game.send(str.encode(action))
        next_state = recv_socket_data(self.sock_game)
        next_state = json.loads(next_state)
        return next_state['observation']['players'][0]['position']

    # pick up or drop the cart
    def pick_and_drop_cart(self):

        action = "0 TOGGLE_CART"
        self.sock_game.send(str.encode(action))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)

        if state['observation']['players'][0]['curr_cart'] == -1:
            print("drop cart!")
        elif state['observation']['players'][0]['curr_cart'] != -1:
            print("pick cart!")
    
    # approach the shelf with simple moves
    def go_to_shelf(self, direction, shelf_interact_pose, axis_we_care):
        assert direction == "NORTH" or direction == "EAST" or direction == "WEST", "purchase_operator: heading to invalid direction!"
        action = "0 " + direction
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        current_agent_pos = state['observation']['players'][0]['position']
        while abs(current_agent_pos[axis_we_care] -  shelf_interact_pose) > self.interact_distance:
            self.sock_game.send(str.encode(action))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            current_agent_pos = state['observation']['players'][0]['position']
        print("agent at {}".format(current_agent_pos))
    
    def pickup_goods(self):
        # pick up the target object from shelf
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        # sometimes need multiple try
        pick_time = 0
        initial_quants = state['observation'][self.cart_or_basket][0]['contents_quant']
        while pick_time < self.max_try_time:
            if state['observation']['players'][0]['holding_food'] is not None or state['observation'][self.cart_or_basket][0]['contents_quant'] != initial_quants:
                print("got {}".format(state['observation']['players'][0]['holding_food']))
                break
            else:
                self.sock_game.send(str.encode("0 INTERACT"))
                state = recv_socket_data(self.sock_game)
                state = json.loads(state)
                # print(state['observation'][self.cart_or_basket][0]['contents_quant'])
            pick_time += 1
        if state['observation']['players'][0]['holding_food'] is None and state['observation'][self.cart_or_basket][0]['contents_quant'] == initial_quants:
            raise RuntimeError("purchase_operator: could not pickup good!")
        
    # staightly go back to cart
    def put_good_to_cart(self, axis_we_care):
        # direction is just the direction of which we park out cart
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        current_agent_pos = state['observation']['players'][0]['position']
        action = "0 " + self.initial_direction
        while abs(current_agent_pos[axis_we_care] - self.agent_pose_before_droping_cart[axis_we_care]) > self.interact_distance/2:
            self.sock_game.send(str.encode(action))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            current_agent_pos = state['observation']['players'][0]['position']
        
        put_back_time = 0
        initial_quants = state['observation'][self.cart_or_basket][0]['contents_quant']
        quants = initial_quants
        while put_back_time < self.max_try_time:
            if quants != initial_quants:
                print("put food to cart successfullly!")
                break
            else:
                self.sock_game.send(str.encode("0 INTERACT"))
                state = recv_socket_data(self.sock_game)
                state = json.loads(state)
                quants = state['observation'][self.cart_or_basket][0]['contents_quant']
                print(quants)
            put_back_time += 1

        if quants == initial_quants:
            raise RuntimeError("purchase_operator: cannot put good back to cart!")

# just given a liust of target to buy, deside how to do the complete buying action 
# basically just combing the naviagation/simple move operator and the purchasing operator
class general_shopping:
    def __init__(self, sock_game:socket.socket, target_object_list, cart_or_basket,config_file_path) -> None:
        self.sock_game = sock_game
        self.target_object_list = target_object_list
        with open(config_file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        self.navigation_target_list =  yaml_data["TARGET_CART_POSITION"].keys()
        self.simple_move_target_list = yaml_data["SIMPLE_MOVE_OBJECTS"].keys()
        self.navigation_trails_limit = yaml_data["navigation_trails_limit"]
        self.move_operator_list = []
        self.purchase_operator_list = []
        self.cart_or_basket = cart_or_basket
        for target_object in self.target_object_list:
            assert target_object in self.navigation_target_list or target_object in self.simple_move_target_list, "object doesn't exist in the super market, try again!"
            if target_object in self.navigation_target_list:
                move_operator = navigation_operator(self.sock_game, target_object, config_file_path)
            else:
                move_operator = simple_movement_operator(self.sock_game, target_object, cart_or_basket,config_file_path)
            purchase_op = purchase_operator(self.sock_game, target_object, cart_or_basket, config_file_path)
            self.move_operator_list.append(move_operator)
            self.purchase_operator_list.append(purchase_op)
    
    def run_general_shopping(self):
        for op1, op2 in zip(self.move_operator_list, self.purchase_operator_list):
            for i in range(self.navigation_trails_limit):
                success_flag = 0
                success_flag = op1.do_navigation()
                if success_flag == -1:
                    continue
                success_flag = op2.do_purchase()
                if success_flag == 1:
                    print("purchasing successful for item {}".format(op2.target_object))
                    break
            if success_flag != 1:
                raise RuntimeError("purchasing failed for item {}".format(op2.target_object))
            time.sleep(1)

# go to cart/basket return and pickup them
class start_shopping_operator:
    
    def __init__(self, sock_game:socket.socket,cart_or_basket, config_file_path) -> None:
        self.sock_game = sock_game
        self.cart_or_basket = cart_or_basket
        if self.cart_or_basket == "carts":
            self.target_object = "cartReturns"
        elif self.cart_or_basket == "baskets":
            self.target_object = "basketReturns"
        
        with open(config_file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        self.nav_op = navigation_operator(sock_game, self.target_object, config_file_path)

        self.interaction_dist = 0.15
        self.movement_limit = 20
        self.pickup_or_drop_trial_limit = 5

        
        self.desired_y_axis = yaml_data["RETURNS_Y_AXIS"][self.cart_or_basket]

    
    def get_cart_or_basket(self):
        # navigate to above the cart
        self.nav_op.do_navigation()

        # move down until in interact range
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        current_agent_pos = state['observation']['players'][0]['position']

        cnt = 0
        while abs(current_agent_pos[1] - self.desired_y_axis) >= self.interaction_dist:
            self.sock_game.send(str.encode("0 SOUTH"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            current_agent_pos = state['observation']['players'][0]['position']

            if cnt >= self.movement_limit:
                raise RuntimeError("get_cart_or_basket: can't move to {}".format(self.cart_or_basket))
        
        # pick up cart or basket
        cnt = 0
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        cart_or_basket_num = len(state['observation'][self.cart_or_basket])
        while cart_or_basket_num == 0:
            self.sock_game.send(str.encode("0 INTERACT"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            cart_or_basket_num = len(state['observation'][self.cart_or_basket])
            if cnt > self.pickup_or_drop_trial_limit:
                raise RuntimeError("get_cart_or_basket: can't get {}".format(self.cart_or_basket))
        print("got {} successfully!".format(self.cart_or_basket))

    def return_cart_or_basket(self):
        # navigate to above the returns
        self.nav_op.do_navigation()


        self.sock_game.send(str.encode("0 SOUTH"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)

        if self.cart_or_basket == "carts":
            curr_pose = state['observation']['carts'][0]['position']
        elif self.cart_or_basket == "baskets":
            curr_pose = state['observation']['players'][0]['position']
        
        cnt = 0
        while abs(curr_pose[1] - self.desired_y_axis) >= self.interaction_dist+0.15:
            self.sock_game.send(str.encode("0 SOUTH"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            if self.cart_or_basket == "carts":
                curr_pose = state['observation']['carts'][0]['position']
            elif self.cart_or_basket == "baskets":
                curr_pose = state['observation']['players'][0]['position']

            if cnt >= self.movement_limit:
                raise RuntimeError("get_cart_or_basket: can't move to {}".format(self.cart_or_basket))
        
        # put back cart or basket
        cnt = 0
        self.sock_game.send(str.encode("0 NOP"))
        state = recv_socket_data(self.sock_game)
        state = json.loads(state)
        cart_or_basket_num = len(state['observation'][self.cart_or_basket])
        while cart_or_basket_num != 0:
            self.sock_game.send(str.encode("0 INTERACT"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            cart_or_basket_num = len(state['observation'][self.cart_or_basket])
            if cnt > self.pickup_or_drop_trial_limit:
                raise RuntimeError("get_cart_or_basket: can't return {}".format(self.cart_or_basket))
            cnt += 1
        print("return {} successfully!".format(self.cart_or_basket))


# navigate to the exit, and go west to leave
class leave_operator:
    def __init__(self, sock_game:socket.socket, cfg_file_path):
        self.sock_game = sock_game
        self.exit_nav_operator = navigation_operator(sock_game, "exit", cfg_file_path)
        self.movement_limit = 20
    def leave(self):

        self.exit_nav_operator.do_navigation()

        # since env will exit when we go out, we don't need any special checker
        cnt = 0
        while cnt < self.movement_limit:
            self.sock_game.send(str.encode("0 WEST"))
            state = recv_socket_data(self.sock_game)
            state = json.loads(state)
            cnt += 1
        
        if cnt >= self.movement_limit:
            raise RuntimeError("leave_operator: can't exit the mall!")







        
# for testing
if __name__ == "__main__":
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    # sock_game.send(str.encode("0 HACK"))
    # recv_socket_data(sock_game)
    target_object = "milk"
    path = "/home/frank/ethics/propershopper/config.yaml"
    # operator = navigation_operator(sock_game, target_object, path)
    # operator.do_navigtion()
    # purchase = purchase_operator(sock_game, target_object, "carts", path)
    # purchase.do_purchase()
    # simple_move_target = "chocolate milk"
    # simple = simple_movement_operator(sock_game, simple_move_target, "carts", path)
    # simple.do_simple_move()
    # purchase = purchase_operator(sock_game, simple_move_target, "carts", path)
    # purchase.do_purchase()
    # general_purchase_list = ["milk", "oranges", "raspberry", "chicken", "cheese wheel", "red bell pepper", "broccoli","onion"]
    # general_purchase_list = ["milk","oranges", "raspberry"]
    general_purchase_list = ["milk","prepared foods"]
    cart_or_basket = "carts"
    pick_operate = start_shopping_operator(sock_game, cart_or_basket, path)
    pick_operate.get_cart_or_basket()
    genral_buy = general_shopping(sock_game, general_purchase_list, cart_or_basket, path)
    genral_buy.run_general_shopping()
    check_1 = navigation_operator(sock_game,"registers", path)
    check_1.do_navigation()
    checkout_purchase = purchase_operator(sock_game, "registers", cart_or_basket, path)
    checkout_purchase.checkout_good()   
    pick_operate.return_cart_or_basket()        


        
            
