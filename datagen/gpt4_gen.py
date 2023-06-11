import sys 
sys.path.append("/Users/seungyounshin/Desktop/RIL/Projects/multithor_new") # TODO : change this to os.environ

from utils.scene_utils import train_valid_test_scenes 


if __name__=="__main__":
    scene_split_dict = train_valid_test_scenes()
