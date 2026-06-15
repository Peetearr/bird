import sys
from pathlib import Path
import argparse
from scipy.interpolate import CubicSpline
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
import jax
import mujoco
import mujoco.viewer
from brax import envs
import jax.numpy as jp
from scripts.model import Bird
import time
from brax.training.agents.sac import checkpoint as sac_checkpoint  
from brax.training.agents.ppo import checkpoint as ppo_checkpoint


def controller(ff: np.array, calc_pos: np.array, 
                real_pos: np.array, prev_error: np.array,
                kp=np.array([0.0, 0.0]), kd=np.array([0.0, 0.0]), dt = .01):
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

t = np.zeros(N)
for i in range(1, N):
    dist = np.linalg.norm(points[i] - points[i-1])
    t[i] = t[i-1] + dist

cs_x = CubicSpline(t, points[:, 0], bc_type='natural')
cs_y = CubicSpline(t, points[:, 1], bc_type='natural')
cs_dx = cs_x.derivative()
cs_dy = cs_y.derivative()

t_fine = np.linspace(t[0], t[-1], 200)
x_sp = cs_x(t_fine)
y_sp = cs_y(t_fine)

dx_du = cs_dx(t_fine)
dy_du = cs_dy(t_fine)
ds_du = np.sqrt(dx_du**2 + dy_du**2)
L_u = cumtrapz(ds_du, t_fine, initial=0)
total_length = L_u[-1]
L_to_u = CubicSpline(L_u, t_fine)

parser = argparse.ArgumentParser(description='Алгоритм')
parser.add_argument('--path', type=str, default='/home/user/bird/logs/ppo_vel2/000014008320')

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

h_target = 0
byas = .17
V = 7
current_error = [0.0, 0.0]
i = 0
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    with mujoco.Renderer(mj_model, 400, 600) as renderer:
        while True:
            rotmat = mj_data.xmat[2]
            R = rotmat.reshape(3, 3)
            global_yaw = np.arctan2(R[1, 0], R[0, 0]) + np.pi
            if i > 1000:
                body_id = 2
                vx = mj_data.cvel[body_id, 3]
                vy = mj_data.cvel[body_id, 4]
                rotmat = mj_data.xmat[2]
                R = rotmat.reshape(3, 3)
                global_yaw = jp.arctan2(R[1, 0], R[0, 0]) + jp.pi
                norm_v = jp.linalg.norm(jp.array([vx, vy]))
                error_orient = jp.linalg.norm(jp.array([jp.cos(global_yaw) - vx/norm_v, jp.sin(global_yaw) - vy/norm_v]))
                print(f"vx: {vx/norm_v:.2f}, vy: {vy/norm_v:.2f}")
                print(f"Ox: {jp.cos(global_yaw):.2f}, Oy: {jp.sin(global_yaw):.2f}")
                print(f"Ошибка ориентации: {error_orient:.2f}")
                i = 0
            i+=1

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()