from env import ThorMultiEnv
from gpt_module.gpt_prompt import GPT4Agent
from const import *
from utils.print_util import colorprint
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")

    parser.add_argument("--scene", type=str, default="FloorPlan1", help="Scene name")
    parser.add_argument("--agentCount", type=int, default=2, help="Number of agents")
    parser.add_argument("--gridSize", type=float, default=0.25, help="Grid size")

    args = parser.parse_args()

    config_dict = {
        'controller_args': {
            "scene": args.scene,
            "agentCount": args.agentCount,
            'gridSize': args.gridSize
        },
    }

    # setup env, agent, init_obs
    env = ThorMultiEnv(config_dict)
    llm_agent = GPT4Agent(role_msg=GPT4_SYSTEM_PROMPT)
    init_obs = env.init_obs()
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
    

