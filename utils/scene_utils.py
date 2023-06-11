import ai2thor 

def train_valid_test_scenes():
    kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

    room_types = [kitchens, living_rooms, bedrooms, bathrooms]

    train_scenes = []
    valid_scenes = []
    test_scenes = []

    for room in room_types:
        train_scenes.extend(room[:20])
        valid_scenes.extend(room[20:25])
        test_scenes.extend(room[25:30])

    scene_dict = {'train': train_scenes, 'valid': valid_scenes, 'test': test_scenes}
    return scene_dict

if __name__=="__main__":
    print(len(train_valid_test_scenes()['train']))
    print(len(train_valid_test_scenes()['valid']))
    print(len(train_valid_test_scenes()['test']))
    