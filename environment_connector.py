from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from sb3_agent_service import EnvironmentParameters
import utils
from sb3_contrib.common.envs import InvalidActionEnvDiscrete

import numpy as np
import gymnasium as gym
import requests
from pydantic import BaseModel


class ResetResponse(BaseModel):
    observationVector: list[float]
    info: dict[str, str]


class StepResponse(BaseModel):
    observationVector: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, str]


class EnvironmentConnector(gym.Env):
    def __init__(self, environment_parameters: EnvironmentParameters):
        self.observation_space = gym.spaces.MultiDiscrete(environment_parameters.observationRangeSizes,
                                                          start=environment_parameters.observationRangeStarts)
        self.action_space = gym.spaces.Discrete(environment_parameters.actionSpaceSize)

    def reset(self, *, seed: int | None = None, options=None, ):
        super().reset(seed=seed)
        response = requests.post(f"http://127.0.0.1:{utils.get_gbg_port()}/reset")
        reset_response = ResetResponse(**response.json())

        observation_vector = np.array(reset_response.observationVector)
        return observation_vector, reset_response.info

    def step(self, action: int):
        response = requests.post(f"http://127.0.0.1:{utils.get_gbg_port()}/step", json={"action": int(action)})
        step_response = StepResponse(**response.json())

        observation_vector = np.array(step_response.observationVector)
        print(f"step_response: {step_response}")
        return observation_vector, step_response.reward, step_response.terminated, step_response.truncated, step_response.info

    def action_masks(self) -> np.ndarray[bool]:
        """
        only used by MaskablePPO
        :return: action masks witch true if valid and false if invalid
        """
        response = requests.get(f"http://127.0.0.1:{utils.get_gbg_port()}/availableActions")
        action_mask = np.zeros(9, dtype=bool)
        for a in response.json():
            action_mask[a] = True

        print(f"action_mask_json: {response.json()}")
        print(f"action mask: {action_mask}")
        return action_mask
