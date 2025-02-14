from datetime import datetime
import os
import signal
import threading
from uuid import UUID
from typing import Dict, Union, Optional, Tuple, List
import copy
from collections import deque
import random

import numpy as np
import gymnasium as gym
import requests
from pydantic import BaseModel, Field
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, EvalCallback
from stable_baselines3.common.preprocessing import preprocess_obs
import torch as th
from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.responses import JSONResponse
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import obs_as_tensor

# start with: uvicorn httpEnvironment:app --host 0.0.0.0 --port 8095
# start TensorBoard with: tensorboard --logdir ./logs

JAVA_PORT = 8094
PORT = 8095


# class HttpTest:
#     def __init__(self, observation_vector_size: int):
#         self.observation_vector_size = observation_vector_size
#
#     def test(self):
#         print("Http test round started.")
#         env = HttpEnv(self.observation_vector_size)
#         model = DQN("MlpPolicy", env, verbose=1)
#         model.learn(total_timesteps=1000, log_interval=4)
#         requests.post(f"http://127.0.0.1:{JAVA_PORT}/testComplete", "Http test round complete.")
#         print("Http test round complete.")
#         exit_app()


class SelfPlayParameters(BaseModel):
    """
    The Parameters used for self play.
    """
    policyWindowSize: int = Field(description="How many old policy's/ models should be saved.", gt=0)
    addPolicyEveryXSteps: int = Field(description="Add current the policy every X steps. Determines how old policy should be are.",
                                      gt=0)
    useLatestPolicy: float = Field(description="How often should the latest policy be used for self play.", ge=0, le=1)


class EvaluationOptions(BaseModel):
    """
    The Options used for evaluation.
    """
    evaluateEveryEpisodes: int = Field(description="How often should be evaluated.", gt=0)
    numberOfGames: int = Field(description="Number of games for evaluation.", gt=0)
    opponent: str = Field(description="Name of opponent used for evaluation.")
    saveBest: bool = Field(description="Save Policy if better")


def get_save_dir() -> str:
    return f"C:/Users/leonp/IdeaProjects/GBG/agents"


class Agent:
    agent_id: UUID
    baseAlgorithm: BaseAlgorithm
    self_play_polices: deque[BasePolicy] = None
    latest_policy: BasePolicy = None
    use_self_play: bool = False
    use_latest_policy = 0.2
    game_name: str
    agent_type: str

    use_model_lock = threading.Lock()

    def __init__(self, agent_id: UUID, model: BaseAlgorithm, agent_type: str, game_name: str):
        self.agent_id = agent_id
        self.baseAlgorithm = model
        self.game_name = game_name
        self.agent_type = agent_type

    @classmethod
    def form_parameters(cls, agent_id, env: gym.Env, agent_type: str, game_name: str, base_parameters: dict, agent_parameters: dict,
                        network_parameters: dict):

        if "activation_fn" in network_parameters:
            activation_fn = None
            model: BaseAlgorithm
            match network_parameters["activation_fn"]:
                case "ReLU":
                    activation_fn = th.nn.ReLU
                case "Linear":
                    activation_fn = th.nn.Linear
                case "LeakyReLU":
                    activation_fn = th.nn.LeakyReLU
                case "Sigmoid":
                    activation_fn = th.nn.Sigmoid
                case "Tanh":
                    activation_fn = th.nn.Tanh
            network_parameters["activation_fn"] = activation_fn

        # net_arch = network_parameters["net_arch"]
        # network_parameters["net_arch"] = dict(pi=net_arch, vf=net_arch)
        # TODO: make different net_arch for agents

        print(network_parameters)

        match agent_type:
            case "DQN":
                model = DQN("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **agent_parameters)
            case "PPO":
                model = PPO("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **agent_parameters)

        print(model.policy)
        return cls(agent_id, model, agent_type, game_name)

    def learn(self, total_time_steps: int, evaluation_options: EvaluationOptions, self_play_parameters: [SelfPlayParameters | None] = None):
        print(total_time_steps)
        # prepare callbacks
        callbacks: list[BaseCallback]= []
        # update self play callback
        if self_play_parameters is not None:
            callbacks.append(self.prepare_for_self_play(self_play_parameters))

        if evaluation_options is not None:
            callbacks.append(self.prepare_for_evaluation(evaluation_options))

        self.baseAlgorithm.learn(total_time_steps, callback=callbacks)
        requests.post(f"http://127.0.0.1:{JAVA_PORT}/trainingFinished", "Training finished")
        print("Training complete.")

    def save_model_if_better(self, win_rate: float):
        print(win_rate)
        # TODO

    def save_eval_results_to_tensorboard(self, win_rate: float):
        print(win_rate)
        # TODO

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

        print(f"action values: {values[0]}")
        return values[0]

    def predict(self, observation: np.ndarray, available_actions: np.ndarray, deterministic: bool = True, for_self_play: bool = False) -> tuple[int, list[float]]:
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

        print(f"availbae actions {available_actions}")
        print(values)
        print(available_values)
        print(best_action)

        return int(best_action), available_values

    def save(self):
        try:
            path = f"{get_save_dir()}/{self.game_name}/SB3Agent/{self.agent_type}/{str(self.agent_id)}/{get_date_and_time()}"
            self.baseAlgorithm.save(path)
            print(f"Policy saved: {path}")
        except Exception as err:
            print(f"Policy coud not be saved:")
            print(f"{err}")







class AddToSelfPlay(BaseCallback):
    def _on_step(self) -> bool:
        self.agent.self_play_polices.append(self.agent.latest_policy)
        with self.agent.use_model_lock:
            self.agent.latest_policy = copy.deepcopy(self.model.policy)
        return True

    def __init__(self, agent: Agent, verbose: int = 0):
        super().__init__(verbose)
        self.agent = agent

class EvaluationCallback(BaseCallback):
    def __init__(self, agent: Agent, evaluation_options: EvaluationOptions, verbose: int = 0):
        super().__init__(verbose)
        self.agent = agent
        self.evaluation_options = evaluation_options

    def _on_step(self) -> bool:
        response = requests.post(f"http://127.0.0.1:{JAVA_PORT}/eval", json={
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


# TODO: WHere put env and trainer?
agents: dict[UUID, Agent] = dict()


class EnvironmentParameters(BaseModel):
    """
    The Parameters used to creat a gym.env.
    """
    actionSpaceSize: int = Field(gt=0, description="Size of all actions that can be available.")
    observationRangeStarts: list[int] = Field(description="The starts of the integer values of the observationVector.")
    observationRangeSizes: list[int] = Field(
        description="Size of the ranges the values of the observationVector can fall in.")


class HttpEnv(gym.Env):
    def __init__(self, environment_parameters: EnvironmentParameters):
        # self.observation_vector_size = observation_vector_size

        # self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float32,
        #  shape=(self.observation_vector_size,))  # TODO: for differen obs
        self.observation_space = gym.spaces.MultiDiscrete(environment_parameters.observationRangeSizes,
                                                          start=environment_parameters.observationRangeStarts)
        self.action_space = gym.spaces.Discrete(environment_parameters.actionSpaceSize)

    def reset(self, *, seed: int | None = None, options=None, ):
        super().reset(seed=seed)

        print("1")
        response = requests.post(f"http://127.0.0.1:{JAVA_PORT}/reset")
        response_body = response.json()
        print("2")
        observation_vector = np.array(response_body["observationVector"])
        info = response_body["info"]
        # print("reset: ")
        # print(observation_vector)
        return observation_vector, info

    def step(self, action: int):
        response = requests.post(f"http://127.0.0.1:{JAVA_PORT}/step", json={"action": int(action)})
        response_body = response.json()
        # print(f"action: {action}")

        observation_vector = np.array(response_body["observationVector"])
        reward = response_body["reward"]
        terminated = response_body["terminated"] # TODO: BaseModel
        truncated = response_body["truncated"]
        info = response_body["info"]
        # print("step: ")
        # print(observation_vector)
        # print(reward)
        return observation_vector, reward, terminated, truncated, info


app = FastAPI()


class SB3Parameters(BaseModel):
    """
    The Parameters used to creat a gym.env and a stable baseline agent.
    """
    agent_id: UUID = Field(
        description="The agents Id used for loading and saving and also impotent when several agents are used.")
    agentType: str = Field(description="Type of sb3 agent that should be created.")  # TODO: enums
    gameName: str = Field(description="Name of the game. e.g: TicTacToe.")

    baseParameters: Dict[str, Union[str, int, float, bool, None]] = Field(
        description="Parameters every sb3 agent has in common.")
    agentParameters: Dict[str, Union[str, int, float, bool, None]] = Field(
        description="Special parameters specific to agentType.")
    networkParameters: Dict[str, Union[str, int, float, bool, list, None]] = Field(
        description="Parameter to creat the neural Netowrk inside the agent.")  # TODO: link sb3 webside with furthe information
    environmentParameters: EnvironmentParameters = Field(description="Parameters needed to creat the gym.Env")



@app.post("/agents")
async def create_agent(sb3_parameters: SB3Parameters):
    """

    :param sb3_parameters:
    :return:
    """
    global agents
    print(sb3_parameters.agent_id)
    print(sb3_parameters.environmentParameters.actionSpaceSize, sb3_parameters.agentType)
    print(sb3_parameters.baseParameters)
    print(sb3_parameters.agentParameters)
    print(sb3_parameters.networkParameters)
    env = HttpEnv(sb3_parameters.environmentParameters)
    agents[sb3_parameters.agent_id] = Agent.form_parameters(sb3_parameters.agent_id,
                                                            env,
                                                            sb3_parameters.agentType,
                                                            sb3_parameters.gameName,
                                                            sb3_parameters.baseParameters,
                                                            sb3_parameters.agentParameters,
                                                            sb3_parameters.networkParameters)
    return Response("Env created", media_type="text/plain")


class LearningParameters(BaseModel):
    totalTimeSteps: int = Field(gt=0)
    evaluationOptions: EvaluationOptions
    selfPlayParameters: Optional[SelfPlayParameters] = Field(
        "Parameters to use for self play. Only required if you want to use self play.")

# starts training Agent for total number of time steps
@app.post("/agents/{agent_id}/learn")
async def learn(agent_id: UUID, learning_parameters: LearningParameters, background_tasks: BackgroundTasks):
    """

    :param agent_id:
    :param learning_parameters:
    :param background_tasks:
    :return:
    """
    global agents
    print("--- Learning Parameters ---")
    print(f"total episodes: {learning_parameters.totalTimeSteps}")
    print(learning_parameters.selfPlayParameters)
    print(learning_parameters.evaluationOptions)
    print("---------------------------")
    background_tasks.add_task(agents[agent_id].learn, total_time_steps=learning_parameters.totalTimeSteps, self_play_parameters=learning_parameters.selfPlayParameters, evaluation_options=learning_parameters.evaluationOptions)
    return Response("training started", media_type="text/plain")


# starts training Agent for one episode only
@app.post("/trainAgentOneEpisode")
async def train_agent_one_episode(background_tasks: BackgroundTasks):
    background_tasks.add_task()


class PredictRequest(BaseModel):
    observation: list[int] = Field(description="Observation without any preprocessing.")
    availableActions: list[int] = Field(description="Available actions for Action Mask: List of actions (ints) that are available")
    deterministic: bool = Field(description="If ture takes get the Action with the maximum value, if false get a action with the probepility of the value.", default=True)


@app.post("/agents/{agent_id}/predict")
async def predict(agent_id: UUID, predict_request: PredictRequest):
    global agents
    agent = agents[agent_id]

    action_with_values = agents[agent_id].predict(np.array(predict_request.observation), np.array(predict_request.availableActions), predict_request.deterministic)
    response = {"action": int(action_with_values[0]), #TODo: base object
                "actionValues": action_with_values[1]}
    print(f"predict {action_with_values}")
    return JSONResponse(response)


DATE_FORMAT: str = "%d_%m_%Y,%H_%M_%S"

def get_date_and_time() -> str:
    now = datetime.now()
    dt_string = now.strftime(DATE_FORMAT)
    return dt_string

def get_latest_file(files: str):
    return max(files, key=lambda x: datetime.strptime(x, f"{DATE_FORMAT}.zip"))


@app.post("/agents/{agent_id}/save")
async def save(agent_id: UUID): # TODO: put agetntype GaemName in creat
    agents[agent_id].save()


class LoadRequest(BaseModel):
    agentType: str = Field(description="Name of the agent. e.g: PPO, DQN.")
    gameName: str = Field(None, description="Name of the game. e.g: TTT, C4.")
    path: str = Field(None, description="Load a specific model. If None load latest.")


@app.post("/agents/{agent_id}/load")
async def load(agent_id: UUID, load_request: LoadRequest):
    global agents
    print(load_request.agentType)
    if load_request.path is None:
        agent_path = f"{get_save_dir()}/{load_request.gameName}/SB3Agent/{load_request.agentType}/{str(agent_id)}"
        files = os.listdir(f"{agent_path}/")
        latest = get_latest_file(files)
        path = f"{agent_path}/{latest}"
    else:
        path = load_request.path

    print(path)

    match load_request.agentType:
        case "DQN":
            agents[agent_id] = Agent(agent_id, DQN.load(path), load_request.agentType, load_request.gameName)
        case "PPO":
            agents[agent_id] = Agent(agent_id, PPO.load(path), load_request.agentType, load_request.gameName)

    print(f"Agent Loaded: {path}")
    print(agent_id)
    print(agents)


@app.post("/agents/{agent_id}/selfPlay")
async def selfPlay(agent_id: UUID, predict_request: PredictRequest):
    global agents
    action = agents[agent_id].predict(np.array(predict_request.observation), np.array(predict_request.availableActions), predict_request.deterministic, True)[0]
    print(f"self play: {action}")
    return Response(f"{action}", media_type="text/plain")


def exit_app():
    print("Killing process.")
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=PORT)
