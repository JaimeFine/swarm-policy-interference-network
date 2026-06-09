from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from src.pettingzoo_env import _make_env


def build_spin_rllib_env_class():
    try:
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
    except ImportError as exc:
        raise RuntimeError(
            "MAPPO baseline requires ray[rllib] to be installed in the active environment."
        ) from exc

    class SpinRllibEnv(MultiAgentEnv):
        def __init__(self, env_config: dict[str, Any]):
            self.mode = str(env_config["mode"])
            self.n_agents = int(env_config["n_agents"])
            self.steps = int(env_config["max_cycles"])
            self.seed = int(env_config.get("seed", 7))
            self.env = _make_env(mode=self.mode, n_agents=self.n_agents, steps=self.steps)
            self.env.reset(seed=self.seed)
            self.agents = [f"agent_{idx}" for idx in range(self.n_agents)]
            base_obs_space = self.env.observation_space(self.agents[0])
            base_action_space = self.env.action_space(self.agents[0])
            self.action_space = base_action_space
            self.observation_space = spaces.Dict({"obs": base_obs_space})

        def reset(self, *, seed: int | None = None, options=None):
            obs, _ = self.env.reset(seed=self.seed if seed is None else seed, options=options)
            return {agent: {"obs": obs[agent]} for agent in self.agents}

        def step(self, action_dict):
            obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
            dones = {"__all__": bool(all(terminations.values()) or all(truncations.values()))}
            wrapped_obs = {agent: {"obs": obs[agent]} for agent in obs}
            return wrapped_obs, rewards, dones, infos

        def close(self):
            self.env.close()

        def get_env_info(self):
            return {
                "space_obs": self.observation_space,
                "space_act": self.action_space,
                "num_agents": self.n_agents,
                "episode_limit": self.steps,
                "policy_mapping_info": {
                    "all_scenarios": {
                        "description": "one team cooperate",
                        "team_prefix": ("agent_",),
                        "all_agents_one_policy": True,
                        "one_agent_one_policy": True,
                    }
                },
            }

    return SpinRllibEnv
