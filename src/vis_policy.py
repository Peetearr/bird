import jax
import mujoco
import mujoco.viewer
from brax import envs
import jax.numpy as jp
from mujoco import mjx
from model import Bird
import time
from brax.training.agents.bc import networks as bc_networks
from brax.training.agents.sac import checkpoint as sac_checkpoint  
from brax.training.agents.bc import checkpoint as bc_checkpoint

ckpt_path = '/home/user/bird/logs/sac_2/000009150464'
try:
    inference_fn = sac_checkpoint.load_policy(ckpt_path, deterministic=True)  
except:
    inference_fn = bc_checkpoint.load_policy(ckpt_path, bc_networks.make_bc_networks, deterministic=True)  

env_name = 'bird'
envs.register_environment('bird', Bird)
eval_env = envs.get_environment(env_name)

jit_inference_fn = jax.jit(inference_fn)

mj_model = eval_env.sys.mj_model
mj_data = mujoco.MjData(mj_model)

ctrl = jp.zeros(mj_model.nu)
rng = jax.random.PRNGKey(0)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    with mujoco.Renderer(mj_model, 400, 600) as renderer:
        while True:
            act_rng, rng = jax.random.split(rng)
            obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
            action, _ = jit_inference_fn(obs, act_rng)
            # print(action)

            mj_data.ctrl = [action[0], action[1], action[2], action[0], action[1], -action[2], action[3], 0.0]#action[4]]
            for _ in range(eval_env._n_frames):
                mujoco.mj_step(mj_model, mj_data)  # Physics step using MuJoCo mj_step.
                viewer.sync()
                time.sleep(.1)
                print(mj_data.time)