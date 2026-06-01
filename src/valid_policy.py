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
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import CubicSpline
import numpy as np


def controller(ff: np.array, calc_pos: np.array, 
                real_pos: np.array, prev_error: np.array,
                kp=np.array([2.0, 2.0]), kd=np.array([0.2, 0.2]), dt = .01):
    error_pos = calc_pos - real_pos
    derror_pos = (error_pos - prev_error)/dt
    return jp.array(ff + error_pos*kp + derror_pos*kd), error_pos

points = np.array([
    [0, 0],
    [40, 10],
    [50, 60],
    [0, 90],
    [-40, 70],
    [-60, 40],
    [-70, -20]
])

N = len(points)

# Параметризация по длине хорды
t = np.zeros(N)
for i in range(1, N):
    dist = np.linalg.norm(points[i] - points[i-1])
    t[i] = t[i-1] + dist

cs_x = CubicSpline(t, points[:, 0], bc_type='natural')
cs_y = CubicSpline(t, points[:, 1], bc_type='natural')
cs_dx = cs_x.derivative()  # производная для скорости
cs_dy = cs_y.derivative()

t_fine = np.linspace(t[0], t[-1], 200)
x_sp = cs_x(t_fine)
y_sp = cs_y(t_fine)

dx_du = cs_dx(t_fine)
dy_du = cs_dy(t_fine)
ds_du = np.sqrt(dx_du**2 + dy_du**2)
L_u = cumtrapz(ds_du, t_fine, initial=0)
total_length = L_u[-1]

# Шаг 2: Обратная функция u(L)
L_to_u = CubicSpline(L_u, t_fine)

parser = argparse.ArgumentParser(description='Алгоритм')
parser.add_argument('--path', type=str, default='/home/user/bird/logs/ppo_vel_w_orientation_task_w_orient/000014008320')
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

h_target = 0.0
z_pos = []
t = []
ydataerr = []
V = 5.0
current_error = [0.0, 0.0]
for _ in range(100):
    rotmat = mj_data.body("base").xmat
    R = rotmat.reshape(3, 3)
    global_yaw = np.arctan2(R[1, 0], R[0, 0])
    print(f"Угол вокруг глобальной Z: {np.degrees(global_yaw):.1f}°")

    sim_time = mj_data.time
    dx = cs_dx(L_to_u(np.clip(sim_time*V, 0, total_length)))
    dy = cs_dy(L_to_u(np.clip(sim_time*V, 0, total_length)))
    norm = np.linalg.norm([dx, dy])
    feedforward = [dx*V/norm, dy*V/norm]
    x = cs_x(L_to_u(np.clip(sim_time*V, 0, total_length)))
    y = cs_y(L_to_u(np.clip(sim_time*V, 0, total_length)))
    velocity, current_error = controller(feedforward, np.array([x, y]), np.array([mj_data.qpos[0], mj_data.qpos[2]]), current_error)
    act_rng, rng = jax.random.split(rng)
    obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl, jp.array(velocity))
    obs = obs.at[1].set(obs[1] + h_target)
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