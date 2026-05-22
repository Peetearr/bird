import sys
from pathlib import Path
import argparse
import functools

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import jax
import mujoco
import mujoco.viewer
from brax import envs
import jax.numpy as jp
from mujoco import mjx
from scripts.model import Bird
import time
from brax.training.agents.sac import checkpoint as sac_checkpoint  
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import brax.training.acme.specs


parser = argparse.ArgumentParser(description='Алгоритм')
parser.add_argument('--path', type=str, default='/home/user/bird/logs/ppo_3/best_policy')
args = parser.parse_args()

ckpt_path = args.path
try:
    inference_fn = sac_checkpoint.load_policy(ckpt_path, deterministic=True)
except:
    inference_fn = ppo_checkpoint.load_policy(ckpt_path)  

env_name = 'bird'
envs.register_environment('bird', Bird)
eval_env = envs.get_environment(env_name)

jit_inference_fn = jax.jit(inference_fn)

mj_model = eval_env.sys.mj_model
mj_data = mujoco.MjData(mj_model)

ctrl = jp.zeros(mj_model.nu)
rng = jax.random.PRNGKey(0)

h_target = 1
byas_x = 0#.1
byas_y = 1
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    with mujoco.Renderer(mj_model, 400, 600) as renderer:
        while True:
            t_0 = time.time()
            act_rng, rng = jax.random.split(rng)
            obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
            obs = obs.at[1].set(obs[1] + h_target)
            obs = obs.at[0].set(obs[0] + byas_x)
            obs = obs.at[2].set(obs[2] + byas_y)
            # print(obs[1])
            action, _ = jit_inference_fn(obs, act_rng)
            if any(act > 1 or act < -1 for act in action):
                print("Одно или несколько действий выходят за пределы [-1, 1]")

            # mj_data.ctrl = [action[0], action[1], action[2], action[0], action[1], -action[2], action[3], action[4]]
            mj_data.ctrl = [action[0] * .7, action[1] * .7, action[2] * 1.5, action[0] * .7, action[1] * .7, -action[2] * 1.5, action[3] * 1.5, action[4]* 1.]
            for _ in range(eval_env._n_frames):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                print(mj_data.time)