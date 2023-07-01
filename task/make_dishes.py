import sys 
sys.path.append("/Users/seungyounshin/Desktop/RIL/Projects/multithor_new")

from env import ThorMultiEnv
import time

task_config = {
    'FindEgg' : {
        'instruction' : 'Place a cup of coffee and an apple on diningtable1.',
        'scene' : 'FloorPlan7',
        'goal_conditions' : [
            {'Mug|+01.67|+00.90|-01.32' : {'isFilledWithLiquid' : True,
                                           'fillLiquid' : 'coffee',
                                           'parentReceptacles' :['DiningTable|-02.66|+00.00|+03.21'] }},
            {'Apple|-00.63|+00.96|-01.44' : {'parentReceptacles' : ['DiningTable|-02.66|+00.00|+03.21']}}
        ]
    }
}

class MakeDishes(ThorMultiEnv):

    def reset(self, num: str) -> None:
        """
        Move the object as expected.
        """
        self.config_key = list(task_config.keys())[num-1]
        self.config_dict = task_config[self.config_key]
        self.instruction = self.config_dict['instruction']

        self.init_state_frame = self.getObjectStateFrame()
        self.goal_conditions = self.config_dict['goal_conditions']

        # debug for task creation
        #df = self.getObjectStateFrame()
        #filtered_df = df[df['objectType'] == 'Apple']
        #print(filtered_df[['objectId','isFilledWithLiquid', 'fillLiquid','parentReceptacles', 'receptacleObjectIds']])
        #exit()
    
    def get_reward(self) -> int:
        """
        Returns the current reward.
        """
        
        df = self.getObjectStateFrame()

        total_reward = 0
        total_attributes = 0

        for goal_condition in self.goal_conditions:
            for goal_object, goal_attributes in goal_condition.items():
                filtered_df = df[df['objectId'] == goal_object]

                # Check each goal attribute
                for attribute, target_value in goal_attributes.items():
                    total_attributes += 1
                    if attribute in filtered_df.columns:
                        # Check if the current state meets the goal state
                        current_value = filtered_df[attribute].values[0]
                        if isinstance(current_value, list) and isinstance(target_value, list):
                            if set(current_value) == set(target_value):
                                total_reward += 1
                        else:
                            if current_value == target_value:
                                total_reward += 1

        # Normalize reward
        if total_attributes > 0:
            total_reward /= total_attributes

        return total_reward


if __name__ == "__main__":
    from env import ThorMultiEnv
    from gpt_module.gpt_prompt import GPT4Agent
    from const import *
    from utils.print_util import colorprint

    config_dict = {
        'controller_args':{
            "local_executable_path":"/Users/seungyounshin/Desktop/RIL/Projects/ai2thor/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-Thor",
            "scene": "FloorPlan7",
            "renderInstanceSegmentation" : True,
            'renderDepthImage' : True,
            'gridSize': 0.25,
            'agentCount' : 2},
    }

    env = MakeDishes(config_dict)
    env.reset(1)

    '''print(env.init_obs('Make coffee'))
    print(env.step('agent1 : goto countertop1'))
    print(env.step('agent1 : take mug'))
    print(env.step('agent1 : goto coffeemachine1'))
    print(env.step('agent1 : put mug on coffeemachine'))
    print(env.step('agent1 : toggle coffeemachine'))
    print(env.step('agent1 : toggle coffeemachine'))
    print(env.step('agent1 : take mug'))
    print(env.step('agent1 : goto diningtable1'))
    print(env.step('agent1 : put mug on diningtable1'))
    print(env.step('agent1 : goto sink1'))
    print(env.step('agent1 : take apple'))
    print(env.step('agent1 : goto diningtable1'))
    print(env.step('agent1 : put apple on diningtable1'))'''

    print(env.init_obs('Make coffee'))
    print(env.step('agent1 : goto countertop1, agent2 : goto countertop2'))
    print(env.step('agent1 : take mug'))
    print(env.step('agent1 : goto coffeemachine1'))
    print(env.step('agent1 : put mug on coffeemachine1, agent2 : goto coffeemachine1'))
    print(env.step('agent1 : toggle coffeemachine'))
    print(env.step('agent1 : toggle coffeemachine'))
    print(env.step('agent1 : take mug'))
    print(env.step('agent1 : goto diningtable1'))
    print(env.step('agent2 : take apple'))
    print(env.step('agent2 : goto diningtable1'))
    print(env.step('agent1 : put mug on diningtable1'))
    print(env.step('agent1 : goto countertop2'))
    print(env.step('agent2 : put apple on diningtable1'))

    print(env.get_reward())


    '''llm_agent = GPT4Agent(role_msg=GPT4_SYSTEM_PROMPT)
    #llm_agent = GPT4Agent(role_msg=GPT4_SYSTEM_PROMPT_FULL_DEMO)
    init_obs = env.init_obs(instruction = env.instruction)
    MAX_STEPS = 15
    
    colorprint(init_obs, color='gray', font='bold')
    act_type,content = llm_agent.act(init_obs)
    colorprint(llm_agent.getLastAction(), color='green', font='bold')
    obs = ''

    for i in range(MAX_STEPS):

        if act_type.lower() == 'think':
            obs = 'Ok.'
        
        elif act_type.lower() == 'action':
            obs = env.step(llm_agent.last_aciton)

        # print results
        print('Observation : ')
        colorprint(obs, color='gray', font='bold')

        act_type,content =  llm_agent.act(obs)
        print('Action : ')
        colorprint(llm_agent.getLastAction(), color='green', font='bold')

        colorprint(f'Reward : {env.get_reward()}', color='orange', font='bold')'''

