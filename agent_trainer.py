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
            for i, layers in enumerate(network_parameters["net_arch"]):
                network_parameters_flatten[f"layer_{i}"] = layers
            hyper_parameters.update(network_parameters_flatten)

        print(model.policy)
        return cls(agent_id, model, agent_type, game_name, hyper_parameters)

    def learn(self, total_time_steps: int, evaluation_options: EvaluationOptions, self_play_parameters: [SelfPlayParameters | None] = None):
        """
        Starts the training process.
        """
        print(total_time_steps)
        # prepare callbacks
        callbacks: list[BaseCallback] = []
        # update self play callback

        if self_play_parameters is not None:
            callbacks.append(self.prepare_for_self_play(self_play_parameters))
            self.hyper_parameters.update(self_play_parameters.dict())

        if evaluation_options is not None:
            callbacks.append(self.prepare_for_evaluation(evaluation_options))
            self.hyper_parameters.update(evaluation_options.dict())


        print(self.hyper_parameters)

        callbacks.append(HParamCallback(self.hyper_parameters))

        self.baseAlgorithm.learn(total_time_steps, callback=callbacks)
        requests.post(f"http://127.0.0.1:{utils.get_gbg_port()}/trainingFinished", "Training finished")
        print("Training complete.")

    def save_model_if_better(self, win_rate: float):
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.save()


    def save_eval_results_to_tensorboard(self, win_rate: float):
        self.baseAlgorithm.logger.record("win_rate", win_rate)


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
        callback = EveryNTimesteps(n_steps=evaluation_options.evaluateEveryEpisodes,
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

        print(f"action values: {values[0]}")
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
            probabilities = np.array(available_values)
            probabilities = probabilities + abs(probabilities.min())
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
        response = requests.post(f"http://127.0.0.1:{utils.get_gbg_port()}/eval", json={
            "opponentName": self.evaluation_options.opponent,
            "numberOfGames": self.evaluation_options.numberOfGames
        })
        print("WINRATE: ")
        print(response.json())
        win_rate: float = response.json()["winRate"]
        if self.evaluation_options.saveBest:
            self.agent.save_model_if_better(win_rate)
        self.agent.save_eval_results_to_tensorboard(win_rate)
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