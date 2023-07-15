import sys 
sys.path.append("/Users/seungyounshin/Desktop/RIL/Projects/multithor_new")

from utils.print_util import colorprint
from env import ThorMultiEnv
import time

import os
import importlib


def import_tasks(task_dir):
    tasks = {}  # Store all imported tasks here

    # List all .py files in the task directory
    for filename in os.listdir(task_dir):
        if filename.endswith('.py') and filename not in ["__init__.py","task_llm_wrapper.py"]:
            module_name = filename[:-3]  # Strip the .py extension

            # Dynamically import the module
            module = importlib.import_module(f"{task_dir}.{module_name}")
            
            # Store the classes and task_config in the tasks dictionary
            tasks[module_name] = {
                "class": [cls for name, cls in module.__dict__.items() if isinstance(cls, type) and name!='ThorMultiEnv'][0],
                "task_config": module.task_config if hasattr(module, "task_config") else None
            }

    return tasks

def config_wrapper(task_config=None,agentCount=2):

    assert task_config is not None 


    config_dict = {
        'controller_args':{
            "local_executable_path":"/Users/seungyounshin/Desktop/RIL/Projects/ai2thor/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-Thor",
            "scene": task_config['scene'],
            "renderInstanceSegmentation" : True,
            'renderDepthImage' : True,
            'gridSize': 0.25,
            'agentCount' : agentCount},
    }

    return config_dict

def llm_task_wrapper(task_env, 
                     llm_agent, 
                     max_steps :int = 50):
    
    # start of episode
    init_obs = task_env.init_obs(instruction = task_env.instruction)
    colorprint(init_obs, color='gray', font='bold')
    act_type,content = llm_agent.act(init_obs)
    colorprint(llm_agent.getLastAction(), color='green', font='bold')
    obs = ''

    # looping the eposide
    for i in range(max_steps):

        if act_type.lower() == 'think':
            obs = 'Ok.'
        
        elif act_type.lower() == 'action':
            obs = task_env.step(llm_agent.last_aciton)

        # print results
        print('Observation : ')
        colorprint(obs, color='gray', font='bold')

        act_type,content =  llm_agent.act(obs)
        print('Action : ')
        colorprint(llm_agent.getLastAction(), color='green', font='bold')

        #colorprint(f'Reward : {task_env.get_reward()}', color='orange', font='bold')
    
    colorprint(f'Reward : {task_env.get_reward()}', color='orange', font='bold')
    colorprint(f'Steps  : {task_env.steps}', color='orange', font='bold')
    return task_env.get_reward()


if __name__ == "__main__":
    from env import ThorMultiEnv
    from gpt_module.gpt_prompt import GPT4Agent
    from const import *

    tasks = import_tasks('task')
    task_class = tasks['make_dishes']['class']
    task_name = 'MakeCoffeePlaceAppleAtDiningTable'
    config = config_wrapper(task_config=tasks['make_dishes']['task_config'][task_name] ,agentCount=3)
    task_instance = task_class(config)
    task_instance.reset(task_name)
    llm_agent = GPT4Agent(role_msg=GPT4_SYSTEM_PROMPT)

    llm_task_wrapper(task_instance, llm_agent)