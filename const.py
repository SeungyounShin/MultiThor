RECEPTACLE_TYPE = [
    'CounterTop', 'SinkBasin', 'Microwave', 'Fridge', 'SideTable',
    'ArmChair', 'Sofa', 'GarbageCan','HandTowelHolder', 'TowelHolder',
    'BathtubBasin', 'Bed', 'Box', 'CoffeeMachine', 'CoffeeTable', 'Desk', 'DiningTable',
    'Dresser', 'Shelf', #'Cabinet', 'Drawer'
]

VISIBLE_OBJECT_EXCLUDE = [
    'CounterTop', 'SinkBasin', 'Fridge', 'Cabinet', 'Drawer','SideTable',
    'ArmChair', 'Sofa', 'GarbageCan','HandTowelHolder', 'TowelHolder',
    'BathtubBasin', 'Bed', 'Box','CoffeeTable', 'Desk', 'DiningTable',
    'Dresser', 'ShelvingUnit', 'Floor'
]

ACTION_PRIMITIVE = [
    'goto', 'take', 'put' , 'open', 'close' ,  'toggle', 
]

ACTION_TEMPLATE = {
    'goto': 'goto {recep} : navigating to {recep}',
    'take' : 'take {obj} : using manipualtor to pick up {obj} in the ego-centric view',
    'put' : 'put {obj} on {recep} : put held object (using manipualtor)',
    'open' : 'open {obj/recep} : open the {obj/recep} (using manipualtor)',
    'toggle' : 'toggle {obj/recep} : toggle the {obj/recep} (using manipualtor)'
}

ACTION_TEMPLATE_TOPRINT = "\n".join([f"{template}" for action, template in ACTION_TEMPLATE.items()])


GPT4_SYSTEM_PROMPT = f"""You are an intelligent assistant capable of controlling multiple agents in an AI2Thor environment.
The agents need to collaborate to solve tasks given in natural language instructions.
Your role is to understand these instructions and guide the agents to complete the tasks.

Valid Actions : 
{ACTION_TEMPLATE_TOPRINT}

Example Demo:
###Env: There are 2 agents in the scene. You are instructed to finish 'Make a coffee'
You see cabinet1-9, coffeemachine1, countertop1-3, drawer1-8, fridge1, garbagecan1, microwave1, sink1.
###Assistant:
<think>To make coffee, I need to find a mug first. Mugs can be found on countertops 1-3, in cabinets 1-9, and in drawers 1-8. I need to direct Agent1 to countertop1 and Agent2 to countertop2 to find mug first.</think>
###Env:
Ok.
###Assistant:
<action>agent1 : goto countertop1, agent2 : goto countertop2</action>
...
(made coffee)
...
###Assistant:
<done>Finish making coffee</done>
"""

GPT4_SYSTEM_PROMPT_FULL_DEMO = f"""You are an intelligent assistant capable of controlling multiple agents in an AI2Thor environment.
The agents need to collaborate to solve tasks given in natural language instructions.
Your role is to understand these instructions and guide the agents to complete the tasks.

Action Primitives: {ACTION_PRIMITIVE}

Example Demo:
###Env: There are 2 agents in the scene. You are instructed to finish 'Pick all mugs an put it on the countertop1'
You see cabinet1-9, coffeemachine1, countertop1-3, drawer1-8, fridge1, garbagecan1, microwave1, sink1.
###Assistant:
<think>I think (fill this to accomplish instruction) </think>
###Env:
Ok.
###Assistant:
<action>agent1 : goto drawer1, agent2 : goto drawer2</action>
###Env:
agent1 arrived at drawer1. The drawer1 is closed.
agent2 arrived at drawer2. The drawer2 is closed.
<think>I think I need to open the drawer in order to verify if the mug is inside.</think>
###Env:
Ok.
###Assistant:
<action>agent1 : open drawer1, agent2 : open drawer2</action>
###Env:
agent1 : agent1 open the drawer1. In it, you see mug, ....
agent2 : agent2 open the drawer2. In it, (~)
<think>Agent1 discovered the mug in drawer1, while agent2 did not find anything. Agent1 should proceed to retrieve the mug and place it on countertop1. Meanwhile, agent2 should search for alternative locations where the mug might be present. However, before doing so, drawer2 needs to be closed by agent2.</think>
###Env:
Ok.
<action>agent1 : take mug, agent2 : close drawer2</action>

...

"""