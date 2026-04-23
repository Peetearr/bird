import sys
from pathlib import Path
import argparse

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import jax
import mujoco
from brax import envs
import jax.numpy as jp
from mujoco import mjx
from scripts.model import Bird
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import checkpoint as sac_checkpoint  
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import functools
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Алгоритм')
parser.add_argument('--path', type=str, default='/home/user/bird/logs/ppo_vel/000070287360')
args = parser.parse_args()

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
    ppo.train, num_timesteps=1, num_evals=1, reward_scaling=0.1,
    episode_length=1, normalize_observations=True, action_repeat=1,
    unroll_length=1, num_minibatches=1, num_updates_per_batch=1,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1,
    batch_size=1, network_factory=make_networks_factory, seed=0)

ckpt_path = args.path
try:
    inference_fn = sac_checkpoint.load_policy(ckpt_path, deterministic=True)  
except:
    inference_fn = ppo_checkpoint.load_policy(ckpt_path, deterministic=True)  

env_name = 'bird'
envs.register_environment('bird', Bird)
eval_env = envs.get_environment(env_name)

jit_inference_fn = jax.jit(inference_fn)

mj_model = eval_env.sys.mj_model
mj_data = mujoco.MjData(mj_model)

ctrl = jp.zeros(mj_model.nu)
rng = jax.random.PRNGKey(0)

h_target = .6
byas = .17
z_pos = []
t = []
ydataerr = []
for _ in range(1000):
    if _%10==0:
        print(_)
    act_rng, rng = jax.random.split(rng)
    obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl, jp.array([7, 0]))
    # obs = obs.at[1].set(obs[1] + h_target + byas)
    action, _ = jit_inference_fn(obs, act_rng)

    mj_data.ctrl = [action[0] * .7, action[1] * .7, action[2] * 1.5, action[0] * .7, action[1] * .7, -action[2] * 1.5, action[3] * 1.5, action[4]* 1.]
    for k in range(eval_env._n_frames):
        mujoco.mj_step(mj_model, mj_data)  # Physics step using MuJoCo mj_step.
        z_pos.append(mj_data.qvel[0])
        t.append(mj_data.time)
        ydataerr.append(0.0)

plt.xlabel('environment steps')
plt.ylabel('velocity')
plt.title(f'y={z_pos[-1]:.3f}')

plt.errorbar(
    t, z_pos, yerr=ydataerr)
plt.axhline(y=7, color='red', linewidth=1, label='reference')

# plt.autoscale(enable=True, axis='both', tight=True)
plt.margins(y=0.1)
plt.show()