from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Only imports the below statements during type checking
   from sb3_agent_service import EvaluationOptions, SelfPlayParameters
import utils

import threading
from uuid import UUID
import copy
from collections import deque
import random
import numpy as np
import gymnasium as gym
import requests
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.logger import HParam
from sb3_contrib import MaskablePPO


class Agent:
    """
    Acts like a wrapper for a SB3 agent, with further customizations.
    """
    agent_id: UUID
    baseAlgorithm: BaseAlgorithm
    self_play_polices: deque[BasePolicy] = None
    latest_policy: BasePolicy = None
    use_self_play: bool = False
    use_latest_policy = 0.2
    game_name: str
    agent_type: str
    best_win_rate: float = 0
    hyper_parameters: dict[str, Any] = {}
    # callbacks: list[BaseCallback] = []

    use_model_lock = threading.Lock()

    def __init__(self, agent_id: UUID, model: BaseAlgorithm, agent_type: str, game_name: str, hyper_parameters=None) -> None:
        if hyper_parameters is None:
            hyper_parameters = {}
        self.hyper_parameters = hyper_parameters
        self.agent_id = agent_id
        self.baseAlgorithm = model
        self.game_name = game_name
        self.agent_type = agent_type

    @classmethod
    def form_parameters(cls, agent_id, env: gym.Env, agent_type: str, game_name: str, base_parameters: dict, agent_parameters: dict,
                        network_parameters: dict) -> Agent:
        """
        Creates a new agent from parameters passed by GBG through the sb3_agent_service.
        """
        print(th.version.cuda)
        print(th.cuda.is_available())
        if th.cuda.is_available():
            print(f"Number of GPUs available: {th.cuda.device_count()}")
            for i in range(th.cuda.device_count()):
                print(f"GPU {i}: {th.cuda.get_device_name(i)}")
        else:
            print("No GPU devices available.")

        hyper_parameters = base_parameters | agent_parameters

        if "activation_fn" in network_parameters:
            activation_fn = None
            model: BaseAlgorithm
            match network_parameters["activation_fn"]:
                case "ReLU":
                    activation_fn = th.nn.ReLU
                case "LeakyReLU":
                    activation_fn = th.nn.LeakyReLU
                case "Sigmoid":
                    activation_fn = th.nn.Sigmoid
                case "Tanh":
                    activation_fn = th.nn.Tanh
            network_parameters["activation_fn"] = activation_fn

        print("network_parameters:")
        print(network_parameters)

        match agent_type:
            case "DQN":
                model = DQN("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **agent_parameters)
            case "PPO":
                model = PPO("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **agent_parameters)
            case "MPPO":
                model = MaskablePPO("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **agent_parameters)

        network_parameters_flatten: dict[str, int] = {}
        if "net_arch" in network_parameters:
            for i, layer in enumerate(network_parameters["net_arch"]):
                network_parameters_flatten[f"layer_{i}"] = layer
            hyper_parameters.update(network_parameters_flatten)

        print("Network:")
        print(model.policy)
        return cls(agent_id, model, agent_type, game_name, hyper_parameters)

    def learn(self, total_time_steps: int, evaluation_options: EvaluationOptions, self_play_parameters: [SelfPlayParameters | None] = None):
        """
        Starts the training process.
        """
        print(f"Training started for total time steps: {total_time_steps}")
        # prepare callbacks
        callbacks: list[BaseCallback] = []
        # update self play callback

        if self_play_parameters is not None:
            callbacks.append(self.prepare_for_self_play(self_play_parameters))
            self_play_parameters_dict: dict = {f"self_play_parameters/{key}": value for key, value in self_play_parameters.dict().items()}
            self.hyper_parameters.update(self_play_parameters_dict)

        if evaluation_options is not None:
            callbacks.append(self.prepare_for_evaluation(evaluation_options))
            evaluation_options_dict: dict = {f"evaluation_options/{key}": value for key, value in evaluation_options.dict().items()}
            self.hyper_parameters.update(evaluation_options_dict)


        print(f"Hyperparameters: {self.hyper_parameters}")

        callbacks.append(HParamCallback(self.hyper_parameters))
        # self.callbacks = callbacks

        self.baseAlgorithm.learn(total_time_steps, callback=callbacks)

        requests.post(f"http://{utils.get_gbg_ip()}:{utils.get_gbg_port()}/trainingFinished", "Training finished")
        print("Training complete.")

    # def train_one_episode_off_policy(self):
    #     rollout = self.baseAlgorithm.collect_rollouts(
    #         self.baseAlgorithm.env,
    #         train_freq=self.baseAlgorithm.train_freq,
    #         action_noise=self.baseAlgorithm.action_noise,
    #         callback=self.callbacks,
    #         learning_starts=self.baseAlgorithm.learning_starts,
    #         replay_buffer=self.baseAlgorithm.replay_buffer,
    #         log_interval=4,
    #     )
    #
    #     if self.baseAlgorithm.num_timesteps > 0 and self.baseAlgorithm.num_timesteps > self.baseAlgorithm.learning_starts:
    #         # If no `gradient_steps` is specified,
    #         # do as many gradients steps as steps performed during the rollout
    #         gradient_steps = self.baseAlgorithm.gradient_steps if self.baseAlgorithm.gradient_steps >= 0 else rollout.episode_timesteps
    #         # Special case when the user passes `gradient_steps=0`
    #         if gradient_steps > 0:
    #             self.baseAlgorithm.train(batch_size=self.baseAlgorithm.batch_size, gradient_steps=gradient_steps)
    #
    #
    # def train_one_episode_off_policy(self):
    #     continue_training = self.baseAlgorithm.collect_rollouts(self.baseAlgorithm.env, callback, self.baseAlgorithm.rollout_buffer, n_rollout_steps=self.baseAlgorithm.n_steps)
    #
    #     if not continue_training:
    #         break
    #
    #     iteration += 1
    #     self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
    #
    #     # Display training infos
    #     if log_interval is not None and iteration % log_interval == 0:
    #         assert self.ep_info_buffer is not None
    #         self.dump_logs(iteration)
    #
    #     self.train()

    def save_model_if_better(self, win_rate: float):
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.save()


    def save_eval_results_to_tensorboard(self, wins: int, ties: int, losses: int, average_reward: float):
        total_number_of_games = wins + losses + ties
        self.baseAlgorithm.logger.record("eval_results/total_number_of_games", total_number_of_games)
        self.baseAlgorithm.logger.record("eval_results/win_ratio", wins/total_number_of_games)
        self.baseAlgorithm.logger.record("eval_results/tie_ratio", ties/total_number_of_games)
        self.baseAlgorithm.logger.record("eval_results/lose_ratio", losses/total_number_of_games)
        self.baseAlgorithm.logger.record("eval_results/average_reward", average_reward)



    def prepare_for_self_play(self, self_play_parameters: SelfPlayParameters) -> EveryNTimesteps:
        self.use_self_play = True
        self.latest_policy = copy.deepcopy(self.baseAlgorithm.policy)
        self.self_play_polices = deque(maxlen=self_play_parameters.policyWindowSize - 1)
        for _ in range(self_play_parameters.policyWindowSize - 1):
            self.self_play_polices.append(self.latest_policy)
        add_to_self_play = AddToSelfPlay(self)
        add_to_self_play_callback = EveryNTimesteps(n_steps=self_play_parameters.addPolicyEveryXSteps,
                                                    callback=add_to_self_play)
        self.use_latest_policy = self_play_parameters.useLatestPolicy
        return add_to_self_play_callback

    def prepare_for_evaluation(self, evaluation_options: EvaluationOptions) -> BaseCallback:
        callback = EveryNTimesteps(n_steps=evaluation_options.evaluateEverySteps,
                                   callback=EvaluationCallback(self, evaluation_options))
        return callback

    def get_self_play_policy(self) -> BasePolicy:
        if random.random() < self.use_latest_policy:
            return self.latest_policy
        else:
            return random.choice(self.self_play_polices)

    def predict_values(self, observation: np.ndarray, policy: BasePolicy) -> np.ndarray:
        """
        Predicts the values for each action and returns them.
        """
        if type(self.baseAlgorithm) is DQN:
            with th.no_grad():
                tensor = policy.obs_to_tensor(observation)[0]
                values = policy.q_net(tensor)
                values = values.detach().to('cpu').numpy()
        elif type(self.baseAlgorithm) is PPO:
            with th.no_grad():
                tensor = policy.obs_to_tensor(observation)[0]
                distribution = policy.get_distribution(tensor)
                probs = distribution.distribution.probs
                values = probs.detach().to('cpu').numpy()
        elif type(self.baseAlgorithm) is MaskablePPO:
            with th.no_grad():
                tensor = policy.obs_to_tensor(observation)[0]
                distribution = policy.get_distribution(tensor)
                probs = distribution.distribution.probs
                values = probs.detach().to('cpu').numpy()

        return values[0]

    def predict(self, observation: np.ndarray, available_actions: np.ndarray, deterministic: bool = True, for_self_play: bool = False) -> tuple[int, list[float]]:
        """
        Predicts the best action and availableActions values for each action and returns them in a 2-tuple.
        First element of the tuple is the chosen action if deterministic is True takes the action with the best predicted value,
        if deterministic is False takes an action by probability represent by their value.
        The second element are the values of just the availableActions, all not available actions are cut out the order stays the same.
        """
        policy = self.baseAlgorithm.policy
        if for_self_play:
            policy = self.get_self_play_policy()

        values = self.predict_values(observation, policy)

        available_values: list[float] = []
        best_action: int = int(available_actions[0])

        for available_action in available_actions:
            if values[available_action] > values[best_action]:
                best_action = available_action
            available_values.append(float(values[available_action]))

        if not deterministic and len(available_actions) > 1:
            if type(self.baseAlgorithm) is DQN:
                probabilities = th.softmax(th.tensor(available_values), dim=0).numpy()
            elif type(self.baseAlgorithm) is PPO: # already done softmax
                probabilities = np.array(available_values)
                probabilities = probabilities / sum(probabilities)
            elif type(self.baseAlgorithm) is MaskablePPO: # already done softmax
                probabilities = np.array(available_values)
                probabilities = probabilities / sum(probabilities)

            best_action = np.random.choice(available_actions, p=probabilities)

        return int(best_action), available_values

    def save(self):
        """
        Saves the agents policy at the location specified in config.yaml file under "gbg_save_location: "
        """
        try:
            path = f"{utils.get_save_dir()}/{self.game_name}/SB3Agent/{self.agent_type}/{str(self.agent_id)}/{utils.get_date_and_time()}"
            self.baseAlgorithm.save(path)
            print(f"Policy saved: {path}")
        except Exception as err:
            print(f"Policy could not be saved:")
            print(f"{err}")


class AddToSelfPlay(BaseCallback):
    """
    This callback adds the current policy to the self_play_policies of the agent.
    """
    def _on_step(self) -> bool:
        self.agent.self_play_polices.append(self.agent.latest_policy)
        with self.agent.use_model_lock:
            self.agent.latest_policy = copy.deepcopy(self.model.policy)
        return True

    def __init__(self, agent: Agent, verbose: int = 0):
        super().__init__(verbose)
        self.agent = agent


class EvaluationCallback(BaseCallback):
    """
    This callback evaluates the current policy, for that it first uses the /eval endpoint og the GBG SB3 API to get the
    winrate of the current Agent against a specific opponent and then safe it to TensorBoard.
    """
    def __init__(self, agent: Agent, evaluation_options: EvaluationOptions, verbose: int = 0):
        super().__init__(verbose)
        self.agent = agent
        self.evaluation_options = evaluation_options

    def _on_step(self) -> bool:
        response = requests.post(f"http://{utils.get_gbg_ip()}:{utils.get_gbg_port()}/eval", json={
            "opponentName": self.evaluation_options.opponent,
            "numberOfGames": self.evaluation_options.numberOfGames # number of games for each player Postion. Total number of Game = numberOfgames * Player
        }).json()
        total_number_of_games = response["wins"] + response["losses"] + response["ties"]
        print(f"Time steps: {self.num_timesteps}")
        print(f"Results of last evaluation with {total_number_of_games} episodes: ")
        print(response)
        average_reward: float = response["averageReward"]
        if self.evaluation_options.saveBest:
            self.agent.save_model_if_better(average_reward)
        self.agent.save_eval_results_to_tensorboard(response["wins"], response["ties"], response["losses"], average_reward)
        return True


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    hyper_parameters: dict[str, Any]

    def __init__(self, hyper_parameters: dict[str, Any], verbose: int = 0):
        super().__init__(verbose)
        self.hyper_parameters = hyper_parameters

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        hparam_dict.update(self.hyper_parameters)
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
