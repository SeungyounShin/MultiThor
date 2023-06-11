from env import ThorMultiEnv
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

    env = ThorMultiEnv(config_dict)

    while True:
        input_str = input("Enter a command ('q' to exit): ")

        if input_str == 'q':
            break

        env.step(input_str)


