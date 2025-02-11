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
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
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
    policyWindowSize: int = Field("How many old policy's/ models should be saved.", gt=0)
    addPolicyEveryXSteps: int = Field("Add current the policy every X steps. Determines how old policy should be are.",
                                      gt=0)
    useLatestPolicy: float = Field("How often should the latest policy be used for self play.", ge=0, le=1)


class Agent:
    agent_id: UUID
    baseAlgorithm: BaseAlgorithm
    self_play_polices: deque[BasePolicy] = None
    latest_policy: BasePolicy = None
    use_self_play: bool = False
    callback: BaseCallback = None
    use_latest_policy = 0.2

    use_model_lock = threading.Lock()

    def __init__(self, agent_id: UUID, model: BaseAlgorithm, self_play_parameters: [SelfPlayParameters | None] = None):
        self.agent_id = agent_id
        self.baseAlgorithm = model

        if self_play_parameters is not None:
            self.use_self_play = True
            self.latest_policy = copy.deepcopy(self.baseAlgorithm.policy)
            self.self_play_polices = deque(maxlen=self_play_parameters.policyWindowSize - 1)
            for _ in range(self_play_parameters.policyWindowSize - 1):
                self.self_play_polices.append(self.latest_policy)
            add_to_self_play = AddToSelfPlay(self)
            self.callback = EveryNTimesteps(n_steps=self_play_parameters.addPolicyEveryXSteps,
                                            callback=add_to_self_play)
            self.use_latest_policy = self_play_parameters.useLatestPolicy

    @classmethod
    def form_parameters(cls, agent_id, env: gym.Env, agent: str, base_parameters: dict, agent_parameters: dict,
                        network_parameters: dict, self_play_parameters: [SelfPlayParameters | None] = None):

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

        match agent:
            case "DQN":
                model = DQN("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **agent_parameters)
            case "PPO":
                model = PPO("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **agent_parameters)

        print(model.policy)
        return cls(agent_id, model, self_play_parameters)

    def learn(self, total_time_steps: int):
        print(total_time_steps)
        self.baseAlgorithm.learn(total_time_steps, callback=self.callback)
        requests.post(f"http://127.0.0.1:{JAVA_PORT}/trainingFinished", "Training finished")
        print("Training complete.")

    def self_play(self, observation: np.ndarray):
        if random.random() < self.use_latest_policy:
            policy = self.latest_policy
        else:
            policy = random.choice(self.self_play_polices)
        action = policy.predict(observation)

        print(action)

        return action

    def predict_values(self, observation: np.ndarray) -> np.ndarray:
        if type(self.baseAlgorithm) is DQN:
            with th.no_grad():
                tensor = self.baseAlgorithm.policy.obs_to_tensor(observation)[0]
                values = self.baseAlgorithm.policy.q_net(tensor)
                values = values.detach().to('cpu').numpy()
        elif type(self.baseAlgorithm) is PPO:
            with th.no_grad():
                tensor = self.baseAlgorithm.policy.obs_to_tensor(observation)[0]
                distribution = self.baseAlgorithm.policy.get_distribution(tensor)
                probs = distribution.distribution.probs
                values = probs.detach().to('cpu').numpy()

        print(f"action values: {values[0]}")
        return values[0]

    def predict(self, observation: np.ndarray, available_actions: np.ndarray) -> tuple[int, list[float]]:
        values = self.predict_values(observation)
        available_values: list[float] = []
        best_action: int = available_actions[0]

        for available_action in available_actions:
            if values[available_action] > values[best_action]:
                best_action = available_action
            available_values.append(float(values[available_action]))

        print(f"availbae actions {available_actions}")
        print(values)
        print(available_values)
        print(best_action)

        return int(best_action), available_values


class AddToSelfPlay(BaseCallback):
    def _on_step(self) -> bool:
        self.agent.self_play_polices.append(self.agent.latest_policy)
        with self.agent.use_model_lock:
            self.agent.latest_policy = copy.deepcopy(self.model.policy)
        return True

    def __init__(self, agent: Agent, verbose: int = 0):
        super().__init__(verbose)
        self.agent = agent


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
        terminated = response_body["terminated"]
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

    baseParameters: Dict[str, Union[str, int, float, bool, None]] = Field(
        description="Parameters every sb3 agent has in common.")
    agentParameters: Dict[str, Union[str, int, float, bool, None]] = Field(
        description="Special parameters specific to agentType.")
    networkParameters: Dict[str, Union[str, int, float, bool, list, None]] = Field(
        description="Parameter to creat the neural Netowrk inside the agent.")  # TODO: link sb3 webside with furthe information

    selfPlayParameters: Optional[SelfPlayParameters] = Field(
        "Parameters to use for self play. Only required if you want to use self play.")
    environmentParameters: EnvironmentParameters = Field("Parameters needed to creat the gym.Env")


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
    print(sb3_parameters.selfPlayParameters)
    env = HttpEnv(sb3_parameters.environmentParameters)
    agents[sb3_parameters.agent_id] = Agent.form_parameters(sb3_parameters.agent_id, env, sb3_parameters.agentType,
                                                            sb3_parameters.baseParameters,
                                                            sb3_parameters.agentParameters,
                                                            sb3_parameters.networkParameters,
                                                            sb3_parameters.selfPlayParameters)
    return Response("Env created", media_type="text/plain")


class LearningParameters(BaseModel):
    totalTimeSteps: int = Field(gt=0)


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
    print(learning_parameters.totalTimeSteps)
    background_tasks.add_task(agents[agent_id].learn, total_time_steps=learning_parameters.totalTimeSteps)
    return Response("training started", media_type="text/plain")


# starts training Agent for one episode only
@app.post("/trainAgentOneEpisode")
async def train_agent_one_episode(background_tasks: BackgroundTasks):
    background_tasks.add_task()


class PredictRequest(BaseModel):
    observation: list[int] = Field("Observation without any preprocessing.")
    availableActions: list[int] = Field("Available actions for Action Mask: List of actions (ints) that are available")


@app.post("/agents/{agent_id}/predict")
async def predict(agent_id: UUID, predictRequest: PredictRequest):
    global agents
    agent = agents[agent_id]

    action_with_values = agents[agent_id].predict(np.array(predictRequest.observation), np.array(predictRequest.availableActions))
    response = {"action": int(action_with_values[0]), #TODo: base object
                "actionValues": action_with_values[1]}
    print(f"predict {action_with_values}")
    return JSONResponse(response)


@app.post("/agents/{agent_id}/save")
async def save(agent_id: UUID):
    global agents
    path = os.path.dirname(os.path.abspath(__file__))
    agents[agent_id].baseAlgorithm.save(path + f"/saved_models/{str(agent_id)}")


class LoadParameter(BaseModel):
    agentType: str


@app.post("/agents/{agent_id}/load")
async def load(agent_id: UUID, load_parameter: LoadParameter):
    global agents
    print(load_parameter.agentType)
    path = os.path.dirname(os.path.abspath(__file__))
    match load_parameter.agentType:
        case "DQN":
            agents[agent_id] = Agent(agent_id, DQN.load(path + f"/saved_models/{str(agent_id)}"))
        case "PPO":
            agents[agent_id] = Agent(agent_id, PPO.load(path + f"/saved_models/{str(agent_id)}"))
    print(agent_id)
    print(agents)


@app.post("/agents/{agent_id}/selfPlay")
async def selfPlay(agent_id: UUID, observation: list[int]):
    global agents
    action = agents[agent_id].self_play(np.array(observation))[0]
    print(f"self play: {action}")
    return Response(f"{action}", media_type="text/plain")


def exit_app():
    print("Killing process.")
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=PORT)
