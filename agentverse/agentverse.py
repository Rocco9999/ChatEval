import asyncio
import logging
from typing import List

# from agentverse.agents import Agent
from agentverse.agents.conversation_agent import BaseAgent
from agentverse.environments import BaseEnvironment
from agentverse.initialization import load_agent, load_environment, prepare_task_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


class AgentVerse:
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment):
        self.agents = agents
        self.environment = environment

    @classmethod
    def from_task(cls, task: str):
        """Build an AgentVerse from a task name.
        The task name should correspond to a directory in `tasks` directory.
        Then this method will load the configuration from the yaml file in that directory.
        """
        # Prepare the config of the task

        task_config = prepare_task_config(task)

        # Build the agents
        agents = []
        for agent_configs in task_config["agents"]:
            agent = load_agent(agent_configs)
            agents.append(agent)

        # Build the environment
        env_config = task_config["environment"]
        env_config["agents"] = agents
        environment = load_environment(env_config)

        # Set input_path and output_path
        input_path = task_config["data_path"]
        output_path = task_config["output_dir"]

        return cls(agents, environment), input_path, output_path

    async def run(self):
        """Run the environment from scratch until it is done."""
        self.environment.reset()
        while not self.environment.is_done():
            await self.environment.step()

    def reset(self):
        self.environment.reset()
        for agent in self.agents:
            agent.reset()

    async def next(self, *args, **kwargs):
        """Run the environment for one step and return the return message."""
        return await asyncio.run(self.environment.step(*args, **kwargs))
