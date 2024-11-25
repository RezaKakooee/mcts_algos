class Config:
    def __init__(self):


        ## windy_gridworld_env config
        self.wgw_config = {
            'max_steps': 100, # Maximum number of steps before episode ends
            'width': 10, # Grid width
            'height': 7, # Grid height
            'start': (0, 3), # Starting position
            'goal': (7, 3), # Goal position
            'wind': [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], # Wind strengths for each column
            'action_str_to_int': {
                'up': 0,
                'down': 1,
                'left': 2,
                'right': 3
            },
            'goal_reward': 1.0, # Reward for reaching goal
            'timeout_penalty': -1.0, # Penalty for exceeding max steps
            'step_penalty': -0.01, # Penalty for each step taken
        }
        self.wgw_config['action_int_to_str'] = {
            i: s for s, i in self.wgw_config['action_str_to_int'].items()
        }




    def get_config(self):
        return self.__dict__
    


    

if __name__ == "__main__":
    config = Config()
    print(config.get_config())