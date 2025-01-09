import os
import signal

import numpy as np
import gymnasium as gym
import requests
from pydantic import BaseModel, Field
from stable_baselines3 import DQN
from fastapi import BackgroundTasks, FastAPI, Response

# start with: uvicorn httpEnvironment:app --host 0.0.0.0 --port 8095

JAVA_PORT = 8094
PORT = 8095
env = None


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

class Trainer:
    def __init__(self, env: gym.Env):
        self.model = DQN("MlpPolicy", env, verbose=2)

    def learn(self, total_time_steps: int):
        self.model.learn(total_time_steps)
        requests.post(f"http://127.0.0.1:{JAVA_PORT}/trainingFinished", "Training finished")
        print("Training complete.")

# TODO: WHere put env and trainer?
trainer: Trainer


class HttpEnv(gym.Env):

    def __init__(self, observation_vector_size: int, action_space_size):
        self.observation_vector_size = observation_vector_size

        self.observation_space = gym.spaces.Box(low=-100_000, high=100_000, dtype=np.float32, shape=(self.observation_vector_size,))
        self.action_space = gym.spaces.Discrete(action_space_size)

    def reset(self, *, seed: int | None = None, options = None,):
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
    observationVectorSize: int = Field(gt=0)
    actionSpaceSize: int = Field(gt=0)


@app.post("/createEnv")
async def create_env(sb3_parameters: SB3Parameters):
    global env
    global trainer
    print(sb3_parameters.actionSpaceSize, sb3_parameters.observationVectorSize)
    env = HttpEnv(sb3_parameters.observationVectorSize, sb3_parameters.actionSpaceSize)
    trainer = Trainer(env)
    return "Env created"


class LearningParameters(BaseModel):
    totalTimeSteps: int = Field(gt=0)


# starts training Agent for total number of time steps
@app.post("/learn")
async def learn(learning_parameters: LearningParameters, background_tasks: BackgroundTasks):
    global trainer
    background_tasks.add_task(trainer.learn, learning_parameters.totalTimeSteps)
    return "training started"


# starts training Agent for one episode only
@app.post("/trainAgentOneEpisode")
async def train_agent_one_episode(background_tasks: BackgroundTasks):
    background_tasks.add_task()


@app.post("/predict")
async def predict(observation: list[float]):
    global trainer
    action = trainer.model.predict(np.array(observation))[0]
    print(f"predict {action}")
    return Response(f"{action}", media_type="text/plain")


def exit_app():
    print("Killing process.")
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=PORT)
