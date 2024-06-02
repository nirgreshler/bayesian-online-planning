import argparse
import os

from planners.bts import BTS
from planners.nmcts import NMCTS
from planners.tsts import TSTS
from procgen_wrapper.procgen_simulator import ProcGenSimulator


def parse_args():
    """
    Parse command line argument
    """

    parser = argparse.ArgumentParser(description='Run a close-loop simulation of ProcGen.')
    parser.add_argument('-e', '--env', help='The name of the environment, either "maze" or "leaper"',
                        choices=['maze', 'leaper'], type=str, required=True)
    parser.add_argument('-p', '--planner', help='The name of the planner, one of [nmcts, bts, tsts, tsts_det].',
                        choices=['nmcts', 'bts', 'tsts', 'tsts_det'], type=str, required=True)
    parser.add_argument('-m', '--model-path', help='A path to load the neural network model.', type=str, required=True)
    parser.add_argument('-s', '--seed', help='A seed for the ProcGen environment, default is 0', type=int, default=0)
    parser.add_argument('-k', '--time-steps', help='Number of close-loop time steps for evaluation, '
                                                   'default for maze is 100 and for leaper is 25.', type=int, default=0)
    parser.add_argument('-T', '--search-budget', help='Maximum number of iterations for the search, '
                                                      'default is 100.', type=int, default=100)
    parser.add_argument('-r', '--results-folder', help='An optional path to save environment photos.',
                        type=str, default='')

    args = parser.parse_args()

    if args.time_steps == 0:
        if args.env.lower() == 'maze':
            args.time_steps = 100
        else:
            args.time_steps = 25

    if args.time_steps < 0:
        raise ValueError('Please set a positive number of steps.')

    if args.search_budget < 0:
        raise ValueError('Please set a positive search budget.')

    return args


def get_planner(planner_name: str, simulator: ProcGenSimulator, nn_model_path: str):
    """
    Create a planner based on the given name.

    :param planner_name: the name of the planner
    :param simulator: the simulator used in the search process
    :param nn_model_path: the path to load a model for the NN.

    :return: an instance of the planner
    """

    if planner_name == 'nmcts':
        return NMCTS(simulator=simulator, nn_model_path=nn_model_path)
    if planner_name == 'bts':
        return BTS(simulator=simulator, nn_model_path=nn_model_path)
    if planner_name == 'tsts':
        return TSTS(simulator=simulator, nn_model_path=nn_model_path)
    else:
        raise ValueError(f'Invalid planner name "{planner_name}".')


def simulate_procgen(env: str, seed: int, planner_name: str, model_path: str, time_steps: int, results_folder: str = ""):
    """
    Run a close-loop simulation of ProcGen.

    :param env: the name of the environment.
    :param seed: the seed for the ProcGen environment.
    :param planner_name: the name of the planner.
    :param model_path: the path to load a model for the neural network.
    :param time_steps: the number of close-loop time steps for evaluation.
    :param results_folder: an optional path to save environment photos.
    """
    # Create a simulator
    simulator = ProcGenSimulator(env_name=env, rand_seed=seed)

    # Reset the simulator to retrieve the initial state
    next_state = simulator.reset()

    # Construct the planner
    planner = get_planner(planner_name, simulator, model_path)

    save_env = False
    if results_folder:
        save_env = True
        results_folder = os.path.join(results_folder, 'maze', f'{planner_name}', f'seed_{seed}')
        os.makedirs(results_folder, exist_ok=True)

    total_reward = 0

    for step in range(time_steps):
        if save_env:
            # Render the environment and save to file
            simulator.render_state(state=next_state, path=os.path.join(results_folder, f'{step}.png'))
        # Call the planner to get the committed action
        action = planner.plan(next_state, search_budget=args.search_budget)

        # Advance the environment using the committed action
        next_state, reward, is_terminal, info = simulator.step(next_state, action)
        total_reward += reward

        if next_state.is_solved:
            print(
                f'{planner_name.upper()} solved {env} seed {seed} in {step} steps ,accumulating reward of {total_reward}.')
            break


if __name__ == '__main__':
    args = parse_args()

    simulate_procgen(args.env, args.seed, args.planner, args.model_path, args.time_steps, args.results_folder)
