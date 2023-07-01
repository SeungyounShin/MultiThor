import sys 
sys.path.append("/Users/seungyounshin/Desktop/RIL/Projects/multithor_new")

from env import ThorMultiEnv
import time

picknplace_config = {
    'PickTwoStatuePlaceDiningTable' : {
        'instruction' : 'Pick two statue and place it on the diningtable.',
        'scene' : 'FloorPlan3',
    }
}

class PicknPlace(ThorMultiEnv):

    def reset(self, num: str) -> None:
        """
        Move the object as expected.
        """
        list(picknplace_config.keys())
    
    def get_reward(self) -> int:
        """
        Returns the current reward.
        """
        pass 

if __name__ == "__main__":

    config_dict = {
        'controller_args':{
            "scene": "FloorPlan5",
            "renderInstanceSegmentation" : True,
            'renderDepthImage' : True,
            'gridSize': 0.25,
            'agentCount' : 2},
    }

    env = PicknPlace(config_dict)
    env.reset(1)

    # Go to the first reception
    #env.goto_recep('countertop1')
    #env.get_visible_objects() # ['Bottle', ..., 'Vase']
    #env.goto_recep('fridge1')
    
    '''for i in range(10):
        act = random.sample(env.get_all_goto_actions(),2)
        act_str = f'agent1 : {act[0]} , agent2 : {act[1]}'
        env.step(act_str, to_print=True)'''

    action_str = """agent1 : goto sink1,agent2 : goto countertop1"""
    env.step(action_str, to_print=True)
    action_str = """agent1 : take the lettuce"""
    env.step(action_str, to_print=True)
    action_str = """agent1 : put the lettuce on the sink1,agent2 : goto fridge1"""
    env.step(action_str, to_print=True)
    action_str = """agent2 : goto fridge1"""
    env.step(action_str, to_print=True)
    action_str = """agent2 : open fridge1"""
    env.step(action_str, to_print=True)
    time.sleep(5)