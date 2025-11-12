import argparse
import datetime
import os
import pprint
import sys
import time

# noinspection PyUnusedImports
import ale_py
import numpy as np
import torch
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import NetBase

from atari_network import ActionConcatenatedDQN, DQN
from atari_wrapper import make_atari_env


def _patch_collector_for_wsl_time():
    """Patch Collector to handle WSL time adjustment issues."""
    original_set_collect_time = CollectStats.set_collect_time

    def patched_set_collect_time(self, collect_time: float, update_collect_speed: bool = True) -> None:
        return original_set_collect_time(self, max(1e-6, collect_time), update_collect_speed)

    CollectStats.set_collect_time = patched_set_collect_time


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore different action-encoding methods for DDQN in Atari environments")
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb_project", type=str, default="ddqn.action-encoding")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--network", default="classic")
    return parser.parse_args()


def main(args: argparse.Namespace = get_args()) -> None:
    _patch_collector_for_wsl_time()

    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    # noinspection PyUnresolvedReferences
    args.state_shape = env.observation_space.shape or env.observation_space.n
    # noinspection PyUnresolvedReferences
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net: NetBase
    match args.network:
        case "dueling":
            q_params = v_params = {"hidden_sizes": [128]}
            net = DQN(*args.state_shape, args.action_shape, args.device, dueling_param=(q_params, v_params)).to(
                args.device)
        case "concat":
            net = ActionConcatenatedDQN(*args.state_shape, args.action_shape, args.device).to(args.device)
        case _:  # classic
            net = DQN(*args.state_shape, args.action_shape, args.device).to(args.device)
    # noinspection PyUnboundLocalVariable
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # noinspection PyTypeChecker
    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_only_last_obs` and `stack_num` can be removed together when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
    )
    # collector
    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dqn"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    train_start_time = time.time()

    def save_best_fn(p: BasePolicy) -> None:
        torch.save(p.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    # noinspection PyUnusedLocal
    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            elapsed = time.time() - train_start_time
            logger.write("train/env_step", env_step, {
                "train/eps": eps,
                "train/time_elapsed_stepwise": elapsed,
            })

    # noinspection PyUnusedLocal
    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps(args.eps_test)

    # noinspection PyUnusedLocal
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path

    # watch agent's performance
    def watch() -> None:
        print("Setup test envs ...")
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        collected: CollectStats
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            vrb = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
            )
            collector = Collector[CollectStats](policy, test_envs, vrb, exploration_noise=True)
            collected = collector.collect(n_step=args.buffer_size, reset_before_collect=True)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            collected = test_collector.collect(n_episode=args.test_num, render=args.render)
        collected.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    # test train_collector and start filling the replay buffer
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    main(get_args())
