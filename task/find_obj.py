import sys 
sys.path.append("/Users/seungyounshin/Desktop/RIL/Projects/multithor_new")

from env import ThorMultiEnv
import time

findobj_config = {
    'FindEgg' : {
        'instruction' : 'Find a dishsponge',
        'scene' : 'FloorPlan5',
        'target_type' : 'DishSponge'
    }
}

class FindObj(ThorMultiEnv):

    def reset(self, num: str) -> None:
        """
        Move the object as expected.
        """
        self.config_key = list(findobj_config.keys())[num-1]
        self.config_dict = findobj_config[self.config_key]
        self.instruction = self.config_dict['instruction']

        self.init_state_frame = self.getObjectStateFrame()
        self.target_type = self.config_dict['target_type']
    
    def get_reward(self) -> int:
        """
        Returns the current reward.
        """
        # agent meta
        agentMeta = self.getAgentsMetadata()
        isFound = list()

        # 모든 타겟 type 을 get
        for i in range(self.agent_num):
            self.state_frame_i = self.getObjectStateFrame(agent_id=i)
            target_df = self.state_frame_i[self.state_frame_i['objectType'] == self.target_type]

            isFound.append(target_df[['visible']])

        is_visible = any(df['visible'].any() for df in isFound)

        return int(is_visible)


if __name__ == "__main__":
    from env import ThorMultiEnv
    from gpt_module.gpt_prompt import GPT4Agent
    from const import *
    from utils.print_util import colorprint

    config_dict = {
        'controller_args':{
            "scene": "FloorPlan5",
            "renderInstanceSegmentation" : True,
            'renderDepthImage' : True,
            'gridSize': 0.25,
            'agentCount' : 2},
    }

    env = FindObj(config_dict)
    env.reset(1)

    llm_agent = GPT4Agent(role_msg=GPT4_SYSTEM_PROMPT)
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

        colorprint(f'Reward : {env.get_reward()}', color='orange', font='bold')

