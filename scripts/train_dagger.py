from brax.training.agents.bc import train as bc
from brax.training.agents.bc import networks as bc_networks
import functools
from datetime import datetime
import matplotlib.pyplot as plt
from scripts.model import Bird
from brax import envs
from brax.io import model
from jax import numpy as jp

from brax.training.agents.bc import train as bc

make_networks_factory = functools.partial(
    bc_networks.make_bc_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
    bc.train, 
    demo_length=100, 
    epochs=4, 
    normalize_observations=True, 
    reset=False,
    num_envs=50, 
    dagger_steps=20,
    batch_size=1000, 
    learning_rate=3e-3,
    network_factory=make_networks_factory, 
    num_evals=3, 
    eval_length=128,
    save_checkpoint_path='/home/user/bird/logs/bc',
    seed=0)

if __name__ == '__main__':
  envs.register_environment('bird', Bird)
  env_name = 'bird'
  env = envs.get_environment(env_name)
  env = envs.training.wrap(env, episode_length=1000, action_repeat=1)

  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]

  def teacher_inference(obs, _):
    return jp.array([1.0, 1.0, 0.0, 0.0, 0.0])

  def progress(num_steps, metrics):
    print(num_steps)

  make_inference_fn, params, m= train_fn(env=env, progress_fn=progress, teacher_inference_fn=teacher_inference)
  print(m.keys())

  model_path = 'tmp/mjx_brax_policy_dagger.pkl'
  model.save_params(model_path, params)

  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')


  print(x_data, y_data)
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.title(f'y={y_data[-1]:.3f}')
      
  plt.autoscale(enable=True, axis='both', tight=True)
  plt.show()
