About the performance:
During my tests, the agent could complete the task, but because there wasn't enough time to train optimal RL policies, the agent still ran into some things. You can continue training if you want. Instructions are in the trainer.py file
When you run the task, sometimes the agent seems to be going in circles. Don't worry, it isn't stuck. It would eventually get out after some (usually a few dozen, worst case a few hundred) looping.

I attached a video. The video is not the agent's best performance. At the end, the agent appeared to be stuck, but that was just an issue with the linux screen recording after the task was complete and the window closed.



How to run the task:
Just run the shopper.py after starting the env. 

About the files:
shopper.py: high-level shopping task
navigation_operator.py: all the low-level behaviors
trianer.py: for training the RL policies
config.yaml: Some important parameters and pieces of information for the program
socket_env.py: This is a modified version of the original one with some hacking. Please use the modified file for training the RL agent
"*.npy": trained policies name: "..._{object_name}_{x pos}_{y pos}"
