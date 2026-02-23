import jax
import mujoco
import mujoco.viewer
from brax.io import model
from brax import envs
import jax.numpy as jp
from mujoco import mjx
from model import Bird
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
import functools
import matplotlib.pyplot as plt

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
    ppo.train, num_timesteps=1, num_evals=1, reward_scaling=0.1,
    episode_length=1, normalize_observations=True, action_repeat=1,
    unroll_length=1, num_minibatches=1, num_updates_per_batch=1,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1,
    batch_size=1, network_factory=make_networks_factory, seed=0)

env_name = 'bird'
envs.register_environment('bird', Bird)
eval_env = envs.get_environment(env_name)
make_inference_fn, _, _= train_fn(environment=eval_env)

model_path = 'tmp/mjx_brax_policy_just_+.pkl'
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

mj_model = eval_env.sys.mj_model
mj_data = mujoco.MjData(mj_model)

ctrl = jp.zeros(mj_model.nu)
rng = jax.random.PRNGKey(0)

z_pos = []
t = []
ydataerr = []
for _ in range(200):
    act_rng, rng = jax.random.split(rng)
    obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
    ctrl, _ = jit_inference_fn(obs, act_rng)

    mj_data.ctrl = ctrl
    for _ in range(eval_env._n_frames):
        mujoco.mj_step(mj_model, mj_data)  # Physics step using MuJoCo mj_step.
        z_pos.append(mj_data.qpos[2])
        t.append(mj_data.time)
        ydataerr.append(0.0)

print(z_pos)
plt.xlabel('# environment steps')
plt.ylabel('reward per episode')
plt.title(f'y={z_pos[-1]:.3f}')

plt.errorbar(
    t, z_pos, yerr=ydataerr)

plt.autoscale(enable=True, axis='both', tight=True)
plt.show()