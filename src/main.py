# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from envs.bibtex import BibtexEnv
from envs.delicious import DeliciousEnv
from envs.media_mill import MediaMillEnv
from run_algorithms import RunAlgorithms
import torch
import utils
from utils import Logger
import os
import copy


########### types of generic environment ##############
"""
heart_disease, breast_cancer, 'dry_bean', 'wine_quality', 'mushroom', 

spambase, 'credit_approval', 'ocr_handwritten', 'covertype', 'ionosphere',

'statlog_credit', 'image_segmentation', 'diabetic_retiopathy'

"""


device = torch.device("cpu")
total_steps = 0


if __name__ == "__main__":

    # Training of a PPO model
    args = utils.get_parser().parse_args()

    logger = Logger(
        args,
        experiment_name=args.algorithm,
        environment_name=args.env_name,
        seed=str(args.seed),
        human_feedback=args.human_feedback,
        feedback_interval=args.feedback_interval,
        folder=args.folder,
    )
    logger.save_args(args)

    env_name = args.env_name

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    elif env_name.lower() == "bibtex":
        env = BibtexEnv()
        eval_env = copy.deepcopy(env)
    elif env_name.lower() == "delicious":
        env = DeliciousEnv()
        eval_env = copy.deepcopy(env)

    elif env_name.lower() == "media_mill":
        env = MediaMillEnv()
        eval_env = copy.deepcopy(env)
    else:
        print("no environment selected")

    torch.manual_seed(args.seed)

    RA = RunAlgorithms(env, args, logger)

    if args.algorithm.lower() == "ppo":
        RA.train_ppo()

    elif args.algorithm.lower() == "ppo-lstm":
        RA.train_ppolstm()

    elif args.algorithm.lower() == "reinforce":
        RA.train_reinforce()

    elif args.algorithm.lower() == "actor-critic":
        RA.train_actorcritic()

    elif args.algorithm.lower() == "linearucb":
        RA.train_linearucb()
    elif args.algorithm.lower() == "ee-net":
        RA.train_eenet()

    elif args.algorithm.lower() == "tamer":
        RA.train_tamer()

    elif args.algorithm.lower() == "bootstrapped-ts":
        RA.train_bootstrapped_ts()
