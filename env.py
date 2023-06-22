from ai2thor.controller import Controller
import cv2 
import networkx as nx
import matplotlib.pyplot as plt
import math,os,time
import random,re
from tqdm import tqdm
import json,copy
from collections import Counter,defaultdict

from utils.multiagent_path_finding import MAPF
from utils.top_down_utils import *
from utils.navigation_utils import *
from utils.print_util import colorprint
from const import *

class ThorMultiEnv():
    
    def __init__(self, config_dict):
        # Constructor code goes here
        self.config_dict = config_dict
        self.scene = config_dict['controller_args']['scene']
        self.controller = Controller(**config_dict['controller_args'])
        self.agent_num= config_dict['controller_args']['agentCount']
        self.log = ''
        
        # agent initial positions, rotations, horizons, and standing
        self.agent_init_meta = self.getAgentsMetadata()

        # recep
        self.recep_ids = self.get_all_recepIds()
        # mapping name to id
        self.best_recep_pose = dict()

        if os.path.exists(f'./bestposes/{self.scene}.json'):
            print(f'loading best recep pose...')
            with open(f'./bestposes/{self.scene}.json', 'r') as f:
                self.best_recep_pose = json.load(f)
        else:
            print(f'Pre-caching best recep pose...')
            #for recep_id in tqdm(self.recep_ids):
            self.best_recep_pose = self.getBestRecepClosestPos() #self.getBestRecepClosestPos(recep_id)
            #self.best_recep_pose[recep_id] = best_recep_closest_pos
            # save 
            with open(f'./bestposes/{self.scene}.json', 'w') as f:
                json.dump(self.best_recep_pose, f)
        self.recep_name2id, self.recepId2name = self.obj_mapping(list(self.best_recep_pose.keys()))

        # Create a graph
        self.G = nx.Graph()
        # Add nodes to the graph
        reachable_pos = self.getReachablePositions()
        for rp_i, rp in enumerate(reachable_pos):
            self.G.add_node(str(rp_i), **rp)
        for node1 in self.G.nodes:
            for node2 in self.G.nodes:
                if node1 != node2:
                    pos1 = self.G.nodes[node1]
                    pos2 = self.G.nodes[node2]
                    if is_valid_transition(pos1, pos2):
                        weight = self.euclidean_distance(pos1, pos2)
                        #print(weight)
                        self.G.add_edge(node1, node2, weight=weight)
        
        graph_representation = {}
        for node in self.G.nodes:
            graph_representation[node] = list(self.G.adj[node])
        self.G_dict = graph_representation

        # DEBUG : network
        #plt.figure(figsize=(8, 8))
        #pos = {node: (self.G.nodes[node]['x'], self.G.nodes[node]['z']) for node in self.G.nodes}
        #nx.draw(self.G, pos, node_color='b', with_labels=True, font_weight='bold')
        #nx.draw_networkx_nodes(self.G, pos, node_color='r', node_size=200)
        #plt.show()

        # save actions
        self.ongoing_actions = [None] * self.agent_num

        # cv2 window config
        window_offset=300
        cv2.namedWindow('top_down', cv2.WINDOW_NORMAL)
        cv2.moveWindow('top_down', 0, window_offset+30)  # Position for the 'top_down' window
        for agent_i in range(self.agent_num):
            cv2.namedWindow(f'frame_{agent_i}', cv2.WINDOW_NORMAL)
            cv2.moveWindow(f'frame_{agent_i}', window_offset*agent_i, 0)  # Position for the 'frame' window

        colorprint(f'Initialized Environment Done.', color='green', font='bold')

        self.top_down_view_show()
        self.show_egocentric()

    def reset(self, reset_dict):
        # Code to reset the scene goes here
        pass
    
    def get_all_goto_actions(self):
        recep_list = list(self.recep_name2id.keys())

        act_list = list()
        for key, template in ACTION_TEMPLATE.items():
            
            if key == 'goto':
                # for all recep
                for recep in recep_list:
                    act_list.append(template.format(recep=recep))
        
        return act_list

    def getAgentsMetadata(self):
        meta_list = []
        for i in range(self.agent_num):

            meta_list.append({
                'position': self.controller.last_event.events[i].metadata['agent']['position'],
                'rotation': self.controller.last_event.events[i].metadata['agent']['rotation'],
                'cameraHorizon': self.controller.last_event.events[i].metadata['agent']['cameraHorizon'],
                'isStanding': self.controller.last_event.events[i].metadata['agent']['isStanding']
            })
        return meta_list
    
    def getRecepString(self):
        dict_keys = list(self.recep_name2id.keys())
        # Sort the keys
        sorted_keys = sorted(dict_keys, key=lambda item: (re.match("([a-z]+)([0-9]+)", item).groups()[0], int(re.match("([a-z]+)([0-9]+)", item).groups()[1])))

        # Group items by type (e.g., 'cabinet', 'drawer', etc.)
        grouped_items = defaultdict(list)
        for key in sorted_keys:
            key_split = re.match("([a-z]+)([0-9]+)", key).groups()  # Split into word and number
            grouped_items[key_split[0]].append(int(key_split[1]))

        # Constructing the sentence
        sentence_parts = []
        for key, values in grouped_items.items():
            if len(values) == 1:
                sentence_parts.append(f"{key}{values[0]}")
            else:
                sentence_parts.append(f"{key}{values[0]}-{values[-1]}")

        sentence = "You see " + ", ".join(sentence_parts) + "."

        return sentence

    def obj_mapping(self, object_list):
        object_dict = {}
        reverse_object_dict = {}
        counter = {}

        for obj in object_list:
            obj_type = obj.split('|')[0].lower()
            if obj_type not in counter:
                counter[obj_type] = 1
            else:
                counter[obj_type] += 1
            key = f'{obj_type}{counter[obj_type]}'
            object_dict[key] = obj
            reverse_object_dict[obj] = key  # vice-versa mapping

        return object_dict, reverse_object_dict

    def get_recep_list(self):
        return list(self.recep_name2id.keys())
            # Simulate the effect of the actions on the agents' positions.
        # If the updated positions would result in a collision, return True.
        # Otherwise, return False.a
        pass

    def get_metadata(self):
        pass

    def get_all_recepIds(self):
        recep_ids = list()
        # print all object in the scene
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectType'] in RECEPTACLE_TYPE:
                recep_ids.append(obj['objectId'])
        return recep_ids

    def getObjMetabyId(self, obj_id):
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectId'] == obj_id:
                return obj

    def getAgentPosbyId(self, agent_id):
        return self.controller.last_event.events[agent_id].metadata['agent']['position']
    
    def getReachablePositions(self):
        return self.controller.step(action="GetReachablePositions").metadata["actionReturn"]

    def euclidean_distance(self, pos1, pos2):
        return math.sqrt((pos1['x'] - pos2['x']) ** 2 + (pos1['z'] - pos2['z']) ** 2)

    def getObjsOnRecep(self, recep_id):
        object_ids = list()
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectId'] == recep_id:
                return obj['receptacleObjectIds']
        return []
    
    def show_egocentric(self):
        for agent_i in range(self.agent_num):
            frame = self.controller.last_event.events[agent_i].frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2 show
            cv2.imshow(f'frame_{agent_i}', frame)
            cv2.waitKey(1)

    def closest_interactables(self, recep_meta, interactable):
        reachable_positions = self.getReachablePositions()
        
        closest_position = None
        min_distance = float('inf')
        for position in reachable_positions:
            distance = self.euclidean_distance(position, recep_meta['position'])
            if distance < min_distance:
                min_distance = distance
                closest_position = position
    
        # keep only the interactables that are closest to the closest position
        closest_interactables = []
        for i in interactable:
            distance = self.euclidean_distance(i, closest_position)
            if distance < 0.3:
                closest_interactables.append(i)

        return closest_interactables    

    def getBestRecepClosestPos(self):
        agent_init_pos = self.controller.last_event.metadata['agent']
        recep_bestpose_dict = dict()
        reachable_pos = self.getReachablePositions()

        # init recep_bestpose_dict
        for recep_id in self.recep_ids:
            recep_bestpose_dict[recep_id] = (0, None)

        for xyz_dict in tqdm(reachable_pos):
            for rot in list(range(0, 360, 90)):
                rot_dict = {'x': -0.0, 'y': rot, 'z': 0.0}
                for horizon in np.linspace(-30, 60, 4):
                    full_pose_dict = {
                        'x' : xyz_dict['x'],
                        'y' : xyz_dict['y'],
                        'z' : xyz_dict['z'],
                        'rotation' : rot_dict,
                        'horizon' : horizon,
                        'standing' : True,
                    }

                    # physically move robot to check
                    e = self.teleportfull(full_pose_dict, agent_id=0)
                    for objId_in_the_scene in list(e.events[0].instance_masks.keys()):
                        if objId_in_the_scene in self.recep_ids:
                            mask = e.events[0].instance_masks[objId_in_the_scene]
                            w,h = mask.shape
                            y_, x_ = np.where(mask)
                            center_of_mass = (x_.mean(), y_.mean())
                            center_deviation = math.sqrt((w//2-center_of_mass[0])**2 + (h//2-center_of_mass[1])**2)/max(w,h)
                            depth = e.events[0].depth_frame
                            mask_area = mask.sum()/(w*h)
                            mask_detph_mean = (mask*depth).sum()/mask.sum()
                            if mask_detph_mean > 1.25:
                                continue
                            if center_deviation > 0.32:
                                continue
                            if recep_bestpose_dict[objId_in_the_scene][0] < mask_area:
                                recep_bestpose_dict[objId_in_the_scene] = (mask_area, full_pose_dict)

        # assuming recep_bestpose_dict is your dictionary
        keys = list(recep_bestpose_dict.keys())
        num_keys = len(keys)

        # calculate number of rows and columns for subplots
        cols = 4  # define as many as you need
        rows = num_keys // cols 
        rows += num_keys % cols

        # create a position for each subplot
        position = range(1,num_keys + 1)

        # create main figure
        # fig = plt.figure(figsize=(20,20))

        filtered_poses = dict()
        for k, pos in zip(keys, position):
            if recep_bestpose_dict[k][1] is None:
                continue
            #recep_name_k = self.recepId2name[k]
            filtered_poses[k] = recep_bestpose_dict[k][1]
            '''e = self.teleportfull(recep_bestpose_dict[k][1], agent_id=0)
            frame = e.events[0].frame
            
            # add subplot
            ax = fig.add_subplot(rows,cols,pos)
            ax.imshow(frame) # if frame is an image or 2D array
            ax.set_title(f"{k.split('|')[0]}") # show shape as title

        plt.tight_layout()
        plt.savefig("./recep_test.png")'''
        return filtered_poses

    def teleportfull(self,pos, agent_id=0):
        e = self.controller.step(action="TeleportFull", x=pos['x'], 
                             y=pos['y'], 
                             z=pos['z'], 
                             rotation=pos['rotation'], 
                             horizon=pos['horizon'],
                            standing = pos['standing'], agentId = agent_id)
        return e

    def traverse_every_recep_and_plot(self):
        frames = dict()
        masks = dict()
        best_recep_closest_poses = dict()
        for recep in self.recep_name2id.keys():
            recep_id = self.recep_name2id[recep]
            best_recep_closest_pos = self.best_recep_pose[recep_id]

            e = self.controller.step(action="TeleportFull", x=best_recep_closest_pos['x'], 
                             y=best_recep_closest_pos['y'], 
                             z=best_recep_closest_pos['z'], 
                             rotation=best_recep_closest_pos['rotation'], 
                             horizon=best_recep_closest_pos['horizon'],
                            standing = best_recep_closest_pos['standing'])
            frame = e.frame
            frames[recep] = frame
            try:
                masks[recep] = e.instance_masks[recep_id]
            except:
                masks[recep] = np.zeros((300,300))
        
        # plot all
        plt.figure(figsize=(10, 10))
        for i, recep in enumerate(frames.keys()):
            plt.subplot(4, len(frames.keys())//4+1, i+1)
            plt.imshow(frames[recep])
            plt.imshow(masks[recep], alpha=0.5)
            plt.title(recep)
            plt.axis('off')
        plt.show()        

    def get_visible_objects(self, agent_id):
        visible_obj = self.controller.last_event.events[agent_id].metadata['objects']
        visible_obj = [i['objectType'] for i in visible_obj if i['visible']]
        visible_obj.sort()
        return visible_obj

    def get_visible_objectIds(self, agentId =0):
        visible_obj = self.controller.last_event.events[agentId].metadata['objects']
        return [i['objectId'] for i in visible_obj if i['visible']]

    def get_rotation_from_two_points(self, p1, p2):
        # p1 {'x': -1.0, 'y': 0.900999128818512, 'z': 1.0}
        # p2 {'x': -1.0, 'y': 0.900999128818512, 'z': 0.75}

        # get angle
        angle = math.atan2(p2['z'] - p1['z'], p2['x'] - p1['x'])
        # convert to degree
        angle = math.degrees(angle) * -1 + 90
        return angle 

    def get_egocentric_view(self):
        return self.controller.last_event.frame

    def pickup(self, obj_name):
        self.show_egocentric()
        objectIds = self.get_visible_objectIds()
        filtered = [i for i in objectIds if i.split('|')[0].lower() == obj_name.lower()]
        if len(filtered) == 0:
            self.log += f'\nNo {obj_name} in the scene'
            return False
        e = self.controller.step(action="PickupObject", objectId=filtered[0])

        if e.metadata['lastActionSuccess']:
            self.log += f'\nPicked up {obj_name} successfully'
        else:
            self.log += f'\nPicked up {obj_name} failed'
        self.show_egocentric()
        return e

    def put(self, recep_name):
        self.show_egocentric()
        objectIds = self.get_visible_objectIds()
        filtered = [i for i in objectIds if i.split('|')[0].lower() == recep_name]
        e = self.controller.step(action="PutObject", objectId=filtered[0], placeStationary=True)
        # placeStationary : True means it will not consider physics
        self.show_egocentric()
        return e

    def goto_recep(self , recep_name, agent_id=0):
        # get recep pos
        recep_id = self.recep_name2id[recep_name]
        recep_pos = self.getObjMetabyId(recep_id)['position']
        # get agent pos
        agent_pos = self.getAgentPosbyId(agent_id)
        # get reachable pos
        reachable_pos = self.getReachablePositions()
        best_recep_closest_pos = self.best_recep_pose[recep_id]

        # Create a graph
        G = nx.Graph()

        # Add nodes to the graph
        G.add_node('agent', **agent_pos)
        for rp_i, rp in enumerate(reachable_pos):
            G.add_node(str(rp_i), **rp)
        G.add_node('destination', **best_recep_closest_pos)

        # Add edges to the graph
        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 != node2:
                    pos1 = G.nodes[node1]
                    pos2 = G.nodes[node2]
                    if is_valid_transition(pos1, pos2):
                        weight = self.euclidean_distance(pos1, pos2)
                        G.add_edge(node1, node2, weight=weight)

        # Find the shortest path
        shortest_path = nx.dijkstra_path(G, 'agent', 'destination')
        pos = {node: (G.nodes[node]['x'], G.nodes[node]['z']) for node in G.nodes}
        shortest_path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

        self.top_down_view_show()
        self.show_egocentric()
        for pair in shortest_path_edges:
            now = G.nodes[pair[0]]
            nxt = G.nodes[pair[1]]
            
            # check if destination node
            if pair[1] == 'destination':
                rotation = best_recep_closest_pos['rotation']
            else:
                rotation = self.get_rotation_from_two_points(now, nxt)

            e = self.teleportfull({
                'x': nxt['x'],
                'y': nxt['y'],
                'z': nxt['z'],
                'rotation': rotation,
                'horizon': nxt['horizon'] if 'horizon' in nxt else 0,
                'standing': nxt['standing'] if 'standing' in nxt else True,
            })

            # top down vis
            self.top_down_view_show()
            self.show_egocentric()
            #plt.imshow(self.controller.last_event.frame)
            #plt.show()
        
        if e.metadata['lastActionSuccess']:
            self.log += f'\ngoto {recep_name} success'
        else:
            self.log += f'\ngoto {recep_name} failed'
            
        '''# Draw the graph
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, node_color='b', with_labels=True, font_weight='bold')
        nx.draw_networkx_nodes(G, pos, nodelist=['agent', 'destination'], node_color='r', node_size=200)

        # Draw the shortest path
        nx.draw_networkx_edges(G, pos, edgelist=shortest_path_edges, edge_color='r', width=2)
        
        plt.subplot(1, 2, 2)
        # teleport to the best_recep_closest_pos
        print(best_recep_closest_pos)
        self.controller.step(action="TeleportFull", x=best_recep_closest_pos['x'], 
                             y=best_recep_closest_pos['y'], 
                             z=best_recep_closest_pos['z'], 
                             rotation=best_recep_closest_pos['rotation'], 
                             horizon=best_recep_closest_pos['horizon'],
                            standing = best_recep_closest_pos['standing'] , agentId = agent_id)
        # open the recep
        plt.imshow(self.controller.last_event.events[agent_id].frame)
        plt.show()'''
        return e

    def stop(self):
        # Code to stop the simulation goes here
        pass

    def top_down_view_show(self):
        t = get_agent_map_data(self.controller)

        new_frame = add_agent_view_triangle(
            self.controller,
            self.agent_num,
            t["frame"],
            t["pos_translator"],
        )

        #ego_centric_view = self.controller.last_event.frame

        # show 
        cv2.imshow("top_down", cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def parse_actions(self,actions):
        actions = re.split(r"\n|,", actions)
        new_act = [None]*self.agent_num
        for ith,a in enumerate(actions):
            try:
                agent, action_str = a.split(":")
                num = int(re.findall(r'\d+', agent)[0])-1
                new_act[num] = action_str.strip()
            except:
                new_act[num] = 'Invalide Action'

        actions = new_act
        
        #if type(actions) is not list:
        #    return ['Invalid Action'] * self.agent_num

        _actions_ = list()
        for agent_i in range(self.agent_num):
            act = actions[agent_i]
            if act is None:
                _actions_.append("IDLE")
                continue
            try:
                opcode, operand = act.split(' ')[0], ' '.join(act.split(' ')[1:])
                opcode = opcode.strip().lower()
                operand = operand.strip().lower()
            except: 
                _actions_.append('Invalid Action')
                continue

            if opcode not in ACTION_PRIMITIVE:
                _actions_.append('Invalid Action')
                continue 
            else: 
                _actions_.append((opcode, operand))

        return _actions_

    def match_agent_node(self, nav_agents):
        
        # get agent meta
        agent_meta = self.getAgentsMetadata()

        # Create a dictionary to store mapping between agent number and node number
        agent_node_mapping = {}

        # Loop over the agent_meta list
        for i, agent in enumerate(agent_meta):
            agent_position = agent['position']
            # Find the node number that matches the agent's position
            for node in self.G.nodes:
                node_position = self.G.nodes[node]
                if agent_position == node_position:
                    agent_node_mapping[i] = node
                    break
        
        # filter out nav_agents
        filtered_agent_node = dict()
        for na in nav_agents:
            filtered_agent_node[na] = agent_node_mapping[na]

        return filtered_agent_node

    def match_recep_node(self, recep_names):
        recep_node_mapping = dict()
        for recep_name in recep_names:
            # get recep pos
            if recep_name in self.recep_name2id:
                recep_id = self.recep_name2id[recep_name]
                recep_pos = self.getObjMetabyId(recep_id)['position']
                best_recep_closest_pos = self.best_recep_pose[recep_id]

                # mapping recep_node to recep_name
                for node in self.G.nodes:
                    node_position = self.G.nodes[node]
                    if (node_position['x'] == best_recep_closest_pos['x']) and (node_position['z'] == best_recep_closest_pos['z']):
                        recep_node_mapping[recep_name] = node

            else: 
                recep_node_mapping[recep_name] = None

        return recep_node_mapping

    def delete_node(self,graph, node):
        if node in graph:
            del graph[node]  # Delete the node from dictionary keys

        # Delete the node from dictionary values (i.e., from lists of connected nodes)
        for key in graph.keys():
            if node in graph[key]:
                graph[key].remove(node)
                
        return graph

    def nav_plan_to_pair(self, path):
        pairs = []
        if len(path) ==1:
            # don't have to move but most of time this need rotation and horizon chagne
            # HARD-CODED :: path augment
            return [(path[0], path[0])]
        for i in range(len(path) - 1):
            pairs.append((path[i], path[i + 1]))
        return pairs
    
    def nav_basic_step(self, pair, agent_id, final=None):
        now = self.G.nodes[pair[0]]
        nxt = self.G.nodes[pair[1]]
        horizon = 0

        if final is not None:
            rotation = final['rotation']
            horizon = final['horizon']
        else:
            rotation = self.get_rotation_from_two_points(now, nxt)

        e = self.teleportfull({
                'x': nxt['x'],
                'y': nxt['y'],
                'z': nxt['z'],
                'rotation': rotation,
                'horizon': horizon,
                'standing': nxt['standing'] if 'standing' in nxt else True,
        }, agent_id=agent_id)

        return e
    
    def interact_step(self, interaction_action_tuple, agent_id):
        action = interaction_action_tuple[0]
        obj = None
        recep = None
        rest_part = interaction_action_tuple[1].strip()
        words_to_replace = ['the', 'at']
        for word in words_to_replace:
            rest_part = rest_part.replace(word, '')

        # Depending on the action we will parse the string differently
        if action in ['open','close','put']:
            # open/close {recep}
            # put {obj} in/on {recep}
            rest_parts = rest_part.split(' ')
            recep = rest_parts[-1].strip()

        elif action =='take':
            # We can further split the string on 'from' to separate them
            rest_parts = rest_part.split('from')
            obj = rest_parts[0].strip()
            if len(rest_parts) > 1:
                recep = rest_parts[1].strip()

        elif action in ['slice','toggle']:
            rest_parts = rest_part.split(' ')
            obj = rest_parts[-1].strip()

        # get objectId 
        if obj is not None:
            visible_objectIds_i = self.get_visible_objectIds(agent_id)
            gather_objId = list(filter(lambda visible_objId: obj.lower() in visible_objId.lower(), visible_objectIds_i))
            if len(gather_objId) > 0:
                obj = random.choice(gather_objId) # HARD-CODED
        # get recepId
        if recep is not None:
            try:
                recep = self.recep_name2id[recep]
            except:
                recep =None

        # perform action
        if action == 'take':
            e = self.controller.step(
                action="PickupObject",
                objectId=obj,
                forceAction=False,
                manualInteract=False,
                agentId = agent_id
            )
        elif action =='put':
            e = self.controller.step(
                action="PutObject",
                objectId=recep,
                forceAction=False,
                agentId = agent_id
            )
        elif action =='open':
            e = self.controller.step(
                action="OpenObject",
                objectId=recep,
                agentId = agent_id
            )
        elif action=='close':
            e = self.controller.step(
                action="CloseObject",
                objectId=recep,
                agentId = agent_id
            )
        elif action=='toggle':
            obj_toggle_meta = self.getObjMetabyId(obj)
            if obj_toggle_meta is None:
                return False
            if obj_toggle_meta['isToggled']:
                print('isToggled')
                e = self.controller.step(
                    action="ToggleObjectOff",
                    objectId=obj,
                    forceAction=False,
                    agentId = agent_id
                )
            else:
                e = self.controller.step(
                    action="ToggleObjectOn",
                    objectId=obj,
                    forceAction=False,
                    agentId = agent_id
                )
        elif action=='slice':
            pass

        return e

    def visible_object_template(self, agent_id , exclude=VISIBLE_OBJECT_EXCLUDE):
        visiable_objects_agent_i = self.get_visible_objects(agent_id)

        # Count the occurrences of each item type
        item_counts = Counter(visiable_objects_agent_i)

        sentence_parts = []
        for item, count in item_counts.items():
            if item in exclude:
                continue
            if count == 1:
                sentence_parts.append(f"a {item}")
            else:
                sentence_parts.append(f"{count} {item}s")
        if len(sentence_parts) <=0:
            return f'you see nothing'

        sentence = ', '.join(sentence_parts)

        return f'you see {sentence.lower()}'

    def interaction_obs_template(self, agent_id , action_tuple, agent_last_event_meta):
        act_name = action_tuple[0]

        # take 
        if act_name == 'take':
            obj_name = action_tuple[1].replace('the ','').strip()
            return f'You pick up the {obj_name}.'

        elif act_name == 'put':
            # You put the {obj id} on the {recep id}.
            rest_parts = action_tuple[1].replace('the ','').strip()
            rest_parts = rest_parts.split(' ')
            obj_name = rest_parts[0].strip()
            recep = rest_parts[-1].strip()
            return f'You put the {obj_name} on the {recep}.'

        # toggle
        elif act_name == 'toggle':
            if agent_last_event_meta['lastAction'] == 'ToggleObjectOn':
                return f'You turn the {action_tuple[1]} on.'
            else:
                return f'You turn the {action_tuple[1]} off.'
        
        elif act_name == 'open':
            #(a) You open the {recep id}. In it, you see a {obj1 id}, ... and a {objN id}.
            #(b) You open the {recep id}. The {recep id} is empty.
            goto_observation_objs = self.visible_object_template(agent_id)
            return f'You open the {action_tuple[1]}. In it, {goto_observation_objs}.'

        elif act_name == 'close':
            # You close the {recep id}.
            return f'You close the {action_tuple[1]}.'
            
    def step(self, actions, to_print=False):
        ################################################################
        #                                                              #
        #  actions : agent1 : goto countertop\nagent2 : goto fridge1   #
        #                                                              #
        ################################################################
        
        # init stat vis
        self.top_down_view_show()
        self.show_egocentric()

        #####################
        # preprocess action #
        #####################

        # output str 
        reaction_str = ['']*self.agent_num

        # 1. parse actions 
        actions = self.parse_actions(actions)

        # 2. preprocess actions to take 
        nav_agents, nav_recep_names = list(), list()
        interact_agents, interact_actions = list() , list()
        interact_action_dict = dict()
        idle_agents = list()
        for agent_i in range(self.agent_num):
            ## check ongoing actions 
            if self.ongoing_actions[agent_i] is not None:
                #print(f' ongoing ... {agent_i} {self.ongoing_actions[agent_i]}')
                #reaction_str[agent_i] = f"{agent_i+1} has ongoing action {self.ongoing_actions[agent_i]}"
                actions[agent_i] = self.ongoing_actions[agent_i]
                #if self.actions[agent_i] is not None:
                #    reaction_str[agent_i] = f"{agent_i+1} has ongoing action {self.ongoing_actions[agent_i]}"
            ## check actions invalid
            if actions[agent_i] == 'Invalid Action':
                print("!!!!!!! ",actions,agent_i )
                reaction_str[agent_i] = f"{agent_i+1} commanded with Invalid Action."
            ## divide nav and interaction agents
            if actions[agent_i][0] =='goto':
                # nav agent
                nav_agents.append(agent_i)
                nav_recep_names.append(actions[agent_i][1])
            elif actions[agent_i][0].lower() in ACTION_PRIMITIVE[1:]:
                # interaction agent
                interact_agents.append(agent_i)
                interact_actions.append(actions[agent_i])
                interact_action_dict[agent_i] = actions[agent_i]
            elif actions[agent_i][0] =='Invalid Action':
                continue 
            else:
                idle_agents.append(agent_i)

        # 3. take action
        min_action_len = 0

        action_plan = dict()
        # navigation
        # (1) find agent node 
        agent_nodes = self.match_agent_node(nav_agents)
        # (2) find recep node
        recep_nodes = self.match_recep_node(nav_recep_names)
        nav_error_agent = [False]*len(agent_nodes)
        # some utils for nav (match agentId and recep_nodes)
        navAgent2node= dict()
        for enum_i,(recep_name_i,nav_agent_i) in enumerate(zip(nav_recep_names,nav_agents)):
            # get recep node at Graph(G)
            node_num = recep_nodes[recep_name_i]
            if node_num is not None:
                navAgent2node[nav_agent_i] = (node_num, recep_name_i)
            else: 
                reaction_str[nav_agent_i] = f'There is no receptacles named {recep_name_i}'
                nav_error_agent[enum_i] = True
                idle_agents.append(nav_agent_i)
        
        nav_agents = [agent for i, agent in enumerate(nav_agents) if not nav_error_agent[i]]

        # (3) MAPF (Multi-Agent Path Finding)
        min_nav_len, multiagent_path_solution = 0,0
        if len(agent_nodes) > 0:
            #print("Check 1 : ",self.G_dict, list(agent_nodes.values()), list(recep_nodes.values()))
            # skip the errored nav agent
            agent_nodes_list = list()
            recep_nodes_list = list()
            for enum_i,(k,v) in enumerate(recep_nodes.items()):
                if v is not None: 
                    agent_nodes_list.append(list(agent_nodes.values())[enum_i])
                    recep_nodes_list.append(v)
            if len(agent_nodes_list)>0:
                # add all agent that is not a nav-agent (collision avoidance)
                
                '''for agent_i in range(self.agent_num):
                    if agent_i not in nav_agents:
                        agent_node_on_Graph = list(self.match_agent_node([agent_i]).values())[0]
                        agent_nodes_list.append(agent_node_on_Graph)
                        recep_nodes_list.append(agent_node_on_Graph)
                        occupied[agent_i] = True'''
                
                g_copy = copy.deepcopy(self.G_dict)
                for agent_i in range(self.agent_num):
                    if agent_i not in nav_agents:
                        agent_node_on_Graph = list(self.match_agent_node([agent_i]).values())[0] # e.g '21'
                        g_copy = self.delete_node(g_copy, agent_node_on_Graph)

                min_nav_len,multiagent_path_solution = MAPF(g_copy,agent_nodes_list,recep_nodes_list)
                #print(multiagent_path_solution)
        for ni,na in enumerate(nav_agents):
            if not nav_error_agent[ni]:
                action_plan[na] = self.nav_plan_to_pair(multiagent_path_solution[ni])
        
        # interaction
        for interact_agent_i in list(interact_action_dict.keys()):
            action_plan[interact_agent_i] = [interact_action_dict[interact_agent_i]]

        #######################
        # stepping the action #
        #######################
        # get action_plan min
        # TODO : ongoing action keep
        if len(action_plan.values())==0:
            action_min_len =1 
        else:
            action_min_len = len(min(action_plan.values(), key=len))
        agent_who_done = [False]*self.agent_num

        for idle_agent_i in idle_agents:
            agent_who_done[idle_agent_i] = True

        for i in range(action_min_len):
            for agent_id in range(self.agent_num):
            
                #### navigation action ####
                
                if agent_id in nav_agents:
                    nav_pair = action_plan[agent_id].pop(0)
                    # check final node 
                    final = dict()
                    if nav_pair[-1] == navAgent2node[agent_id][0]:
                        # rotation change to heading recep
                        recep_name_final = navAgent2node[agent_id][1]
                        recep_id = self.recep_name2id[recep_name_final]
                        best_recep_closest_pos = self.best_recep_pose[recep_id]
                        final['rotation'] = best_recep_closest_pos['rotation']
                        final['horizon'] = best_recep_closest_pos['horizon']
                        rtn_event = self.nav_basic_step(nav_pair,agent_id,final)
                    else:
                        rtn_event = self.nav_basic_step(nav_pair,agent_id) 
                    action_success_i = rtn_event.events[agent_id].metadata["lastActionSuccess"]

                    # if finish reaching
                    if len(action_plan[agent_id]) <= 0:
                        if action_success_i:
                            arrived_recep_id = self.recep_name2id[navAgent2node[agent_id][1]]
                            # check it's opennable 
                            recep_arrived_meta = self.getObjMetabyId(arrived_recep_id)
                            goto_observation_objs = self.visible_object_template(agent_id)
                            if not recep_arrived_meta['openable']:
                                # general case : not opennable (ex : countertop)
                                reaction_str[agent_id] = f'agent{agent_id+1} arrived at {actions[agent_id][-1]}. On the {actions[agent_id][-1]}, {goto_observation_objs}' # TODO : tell more about what it see
                            else:
                                # opennable 
                                if recep_arrived_meta['isOpen']:
                                    # RECEP opened
                                    reaction_str[agent_id] = f'agent{agent_id+1} arrived at {actions[agent_id][-1]}.  The {actions[agent_id][-1]} is open. On it, {goto_observation_objs}'
                                else:
                                    # RECEP closed
                                    reaction_str[agent_id] = f'agent{agent_id+1} arrived at {actions[agent_id][-1]}. The {actions[agent_id][-1]} is closed.'

                            agent_who_done[agent_id] = True
                        else:
                            reaction_str[agent_id] = f'agent{agent_id+1} failed to arrive at {actions[agent_id][-1]}' 
                
                #### interaction action ####

                if agent_id in list(interact_action_dict.keys()):
                    interaction_action_tuple = action_plan[agent_id][0]
                    # [] TODO : embodied action
                    rtn_event = self.interact_step(interaction_action_tuple,agent_id) 
                    if type(rtn_event) == bool and not rtn_event:
                        # failed
                        reaction_str[agent_id] = f'agent{agent_id+1} failed'
                    else:
                        if rtn_event.events[agent_id].metadata["lastActionSuccess"]:
                            # interaction success
                            reaction_str[agent_id] = self.interaction_obs_template(agent_id, interaction_action_tuple, rtn_event.metadata)
                        else:
                            full_iteraction_action_str = ' '.join(interaction_action_tuple)
                            reaction_str[agent_id] = f'Agent{agent_id+1} failed to execute the operation "{full_iteraction_action_str}".'
                            #reaction_str[agent_id] = f'agent{agent_id+1} failed to operate \'{full_iteraction_action_str}\''
                    agent_who_done[interact_agent_i] = True 

            # visualize
            self.top_down_view_show()
            self.show_egocentric()

        # keep ongoing action for next step 
        for i in range(self.agent_num):
            if not agent_who_done[i]:
                if (type(actions[i]) is not str) and (not actions[i] in ['Invalid Action','IDLE']):
                    self.ongoing_actions[i] = actions[i]
                # ongoing action 
                if reaction_str[i] == '':
                    if (type(actions[i]) is tuple) and (actions[i][0] == 'goto'):
                        #print(f'agent {i+1} heading to...')
                        reaction_str[i] = f'agent{i+1} is heading to {self.ongoing_actions[i][-1]}'
            else:
                self.ongoing_actions[i] = None

        # reaction for idle agent 
        for agent_i in idle_agents:
            if reaction_str[agent_i] == '':
                reaction_str[agent_i] = f'agent{agent_i+1} is currently idle and not engaged in any activities or tasks.'

        #time.sleep(3)
        obs_str = ''
        if to_print:
            colorprint("="*15, font='bold')
            # print action
            colorprint("-"*15, font='italic')
            for agent_i in range(self.agent_num):
                print(f' + agent{agent_i+1} : {actions[agent_i]}')
            colorprint("-"*15, font='italic')
            # print observation
            for agent_i in range(self.agent_num):
                print(f' + agent{agent_i+1} : {reaction_str[agent_i]}')
                obs_str += f'agent{agent_i+1} : {reaction_str[agent_i]}\n'
            colorprint("-"*15, font='italic')
            #print(reaction_str)
            #print(self.ongoing_actions)
        else:
            for agent_i in range(self.agent_num):
                obs_str += f'agent{agent_i+1} : {reaction_str[agent_i]}\n'

        return obs_str

    def init_obs(self, instruction="Put all utensil on the sink"):
        obs = f"There are {self.agent_num} agents in the scene. You are instructed to finish \'{instruction}\'\n"
        recep_names = list(self.recep_name2id.keys())
        obs += self.getRecepString()

        return obs

    def agent_simulation_hand_tunend(self):
        # floorplan 1

        command = {
            0: 'goto countertop1', 
            1: 'goto countertop2',
        }

        self.action_queue = {
            0: [],
            1: [],
        }

        agent_0_act_to_go = self.generate_nav_actions_by_recep(0 , 'sink1')


        min_len = min(len(queue) for _, queue in self.action_queue.items() if queue)
        print(f'min_len = {min_len}')

        t = 0 
        self.top_down_view_show()

        for t_local in range(min_len):
            
            print(f't = {t_local}')
            for agent_id, action_queue in self.action_queue.items():
                if action_queue:
                    next_action = action_queue[0]
                    event = self.controller.step(next_action, agentId=agent_id)
                    # print success
                    success_flag = event.metadata["lastActionSuccess"]
                    print(event.metadata['lastActionSuccess'])
                    print("====")
                    if success_flag:
                        action_queue.pop(0)
            
            self.top_down_view_show()
            t += 1

if __name__=="__main__":

    config_dict = {
        'controller_args':{
            "scene": "FloorPlan1",
            "renderInstanceSegmentation" : True,
            'renderDepthImage' : True,
            'gridSize': 0.25,
            'agentCount' : 5},
    }

    env = ThorMultiEnv(config_dict)
    
    # Go to the first reception
    #env.goto_recep('countertop1')
    #env.get_visible_objects() # ['Bottle', ..., 'Vase']
    #env.goto_recep('fridge1')
    
    '''for i in range(10):
        act = random.sample(env.get_all_goto_actions(),2)
        act_str = f'agent1 : {act[0]} , agent2 : {act[1]}'
        env.step(act_str, to_print=True)'''

    action_str = """agent3 : goto sink1, agent2 : goto countertop2"""
    env.step(action_str, to_print=True)
    time.sleep(5)