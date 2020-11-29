import argparse
import argcomplete
import torch

import logger
import utils
import train

from tasks.tasks import TASKS

# Default values for program arguments
RANDOM_SEED = 1000
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000


def init_arguments():
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()), default='copy',
                        help="Choose the task to train (default: copy)")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL,
                        help="Checkpoint interval (default: {}). "
                             "Use 0 to disable checkpointing".format(CHECKPOINT_INTERVAL))
    parser.add_argument('--checkpoint-path', action='store', default='./',
                        help="Path for saving checkpoint data (default: './')")
    parser.add_argument('--report-interval', type=int, default=REPORT_INTERVAL,
                        help="Reporting interval")
    parser.add_argument('--GPU', action='store_true', default=False, help="Use GPU")

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path.rstrip('/')

    return args


def main():
    logger.init_logging()

    # Initialize arguments
    args = init_arguments()
    args.GPU = args.GPU and torch.cuda.is_available()

    # Initialize random
    utils.init_seed(args.seed)

    # Initialize the model
    model = train.init_model(args)

    logger.LOGGER.info("Total number of parameters: %d", model.net.calculate_num_params())
    train.train_model(model, args)


if __name__ == '__main__':
    main()
