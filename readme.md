build docker:
docker build -t sb3 .

run docker container:
docker run --gpus all -it --rm -v <working directory>:<directory to mount to> -p 8095:8095 --name <name> sb3 uvicorn main:app --host 0.0.0.0 --port 8095
docker run -it --gpus all --ipc=host --rm -v "/c/Users/Leon Püschel/PycharmProjects/SB3/app:/home/mambauser/stable-baselines3/app" -p 8095:8095 --name sb3 sb3pytorch uvicorn sb3_agent_service:app --host 0.0.0.0 --port 8095 --log-level warning
docker run -it --gpus all --rm -v "/c/Users/Leon Püschel/PycharmProjects/SB3/app:/home/mambauser/stable-baselines3/app" -p 8095:8095 --name sb3 sb3pytorch bash
--ipc=host
