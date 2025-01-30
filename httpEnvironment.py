import os
import signal
from uuid import UUID
from typing import Dict, Union

import numpy as np
import gymnasium as gym
import requests
from pydantic import BaseModel, Field
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
import torch as th
from fastapi import BackgroundTasks, FastAPI, Response

# start with: uvicorn httpEnvironment:app --host 0.0.0.0 --port 8095

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

class Agent:
    agent_id: UUID
    model: BaseAlgorithm

    def __init__(self, agent_id: UUID, model: BaseAlgorithm):
        self.agent_id = agent_id
        self.model = model

    @classmethod
    def form_parameters(cls, agent_id, env: gym.Env, agent: str, base_parameters: dict, policy_parameters: dict,
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

        match agent:
            case "DQN":
                model = DQN("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **policy_parameters)
            case "PPO":
                model = PPO("MlpPolicy", env, policy_kwargs=network_parameters, **base_parameters, **policy_parameters)

        print(model.policy)
        return cls(agent_id, model)

    def learn(self, total_time_steps: int):
        print(total_time_steps)
        self.model.learn(total_time_steps)
        requests.post(f"http://127.0.0.1:{JAVA_PORT}/trainingFinished", "Training finished")
        print("Training complete.")


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
        self.observation_space = gym.spaces.MultiDiscrete(environment_parameters.observationRangeSizes, start=environment_parameters.observationRangeStarts)
        self.action_space = gym.spaces.Discrete(environment_parameters.actionSpaceSize)

    def reset(self, *, seed: int | None = None, options=None, ):
        super().reset(seed=seed)

        response = requests.post(f"http://127.0.0.1:{JAVA_PORT}/reset")
        response_body = response.json()

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
    agent_id: UUID = Field(description="The agents Id used for loading and saving and also impotent when several agents are used.")
    agentType: str = Field(description="Type os sb3 agent that should be created.")  # TODO: enums
    baseParameters: Dict[str, Union[str, int, float, bool, None]] = Field(description="Parameters every sb3 agent has in common.")
    policyParameters: Dict[str, Union[str, int, float, bool, None]] = Field(description="Special parameters specific to agentType.")
    networkParameters: Dict[str, Union[str, int, float, bool, list, None]] = Field(description="Parameter to creat the neural Netowrk inside the agent.") # TODO: link sb3 webside with furthe information

    environmentParameters: EnvironmentParameters


@app.post("/agents")
async def create_agent(sb3_parameters: SB3Parameters):
    global agents
    print(sb3_parameters.agent_id)
    print(sb3_parameters.environmentParameters.actionSpaceSize, sb3_parameters.agentType)
    print(sb3_parameters.baseParameters)
    print(sb3_parameters.policyParameters)
    print(sb3_parameters.networkParameters)
    env = HttpEnv(sb3_parameters.environmentParameters)
    agents[sb3_parameters.agent_id] = Agent.form_parameters(sb3_parameters.agent_id, env, sb3_parameters.agentType,
                                                            sb3_parameters.baseParameters,
                                                            sb3_parameters.policyParameters,
                                                            sb3_parameters.networkParameters)
    return Response("Env created", media_type="text/plain")


class LearningParameters(BaseModel):
    totalTimeSteps: int = Field(gt=0)


# starts training Agent for total number of time steps
@app.post("/agents/{agent_id}/learn")
async def learn(agent_id: UUID, learning_parameters: LearningParameters, background_tasks: BackgroundTasks):
    global agents
    print(learning_parameters.totalTimeSteps)
    background_tasks.add_task(agents[agent_id].learn, total_time_steps=learning_parameters.totalTimeSteps)
    return Response("training started", media_type="text/plain")


# starts training Agent for one episode only
@app.post("/trainAgentOneEpisode")
async def train_agent_one_episode(background_tasks: BackgroundTasks):
    background_tasks.add_task()


@app.post("/agents/{agent_id}/predict")
async def predict(agent_id: UUID, observation: list[int]):
    global agents
    action = agents[agent_id].model.predict(np.array(observation))[0]
    print(f"predict {action}")
    return Response(f"{action}", media_type="text/plain")


@app.post("/agents/{agent_id}/save")
async def save(agent_id: UUID):
    global agents
    path = os.path.dirname(os.path.abspath(__file__))
    agents[agent_id].model.save(path + f"/saved_models/{str(agent_id)}")


class LoadParameter(BaseModel):
    agentType: str


@app.post("/agents/{agent_id}/load")
async def load(agent_id: UUID, load_parameter: LoadParameter):
    global agents
    print(load_parameter.agentType)
    path = os.path.dirname(os.path.abspath(__file__))
    match load_parameter.agentType:
        case "DQN": agents[agent_id] = Agent(agent_id, DQN.load(path + f"/saved_models/{str(agent_id)}"))
        case "PPO": agents[agent_id] = Agent(agent_id, PPO.load(path + f"/saved_models/{str(agent_id)}"))
    print(agent_id)
    print(agents)


def exit_app():
    print("Killing process.")
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=PORT)
