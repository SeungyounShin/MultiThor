## Getting Started (GPT4 refer to gpt_play.py)

1. Clone the repository:

```shell
git clone https://github.com/your-username/gpt-play.git
```

2. Install the required dependencies:
```shell
pip install -r requirements.txt
```

3. Run the code:
```shell
python gpt_play.py --scene FloorPlan1 --agentCount 2 --gridSize 0.25
```

4. This will print : 

<font color="gray">**Obs**: </font>
There are 2 agents in the scene. You are instructed to finish 'Put all utensil on the sink'.
You see cabinet1-9, coffeemachine1, countertop1-3, drawer1-8, fridge1, garbagecan1, microwave1, sink1.

**Action**: 

**Think**: I need to find all the utensils, and then make the agents go to the utensils to pick them up and put them on the sink. 

**Obs**: 

Ok.

**Action**: 

**agent1**: goto drawer1, **agent2**: goto drawer2

**Obs**: 

**agent1**: agent1 arrived at drawer1. The drawer1 is closed.
**agent2**: ...

...
```
