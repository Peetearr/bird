from brax.training.agents.sac import train as sac
from brax.training.agents.sac import networks as sac_networks
import functools
from datetime import datetime
import matplotlib.pyplot as plt
from scripts.model import Bird
from brax import envs
from brax.io import model

from brax.training.agents.bc import train as bc
from brax.training.agents.sac import losses as sac_losses  
from brax.training.acme import running_statistics 
import os

make_networks_factory = functools.partial(
    sac_networks.make_sac_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))


train_fn = functools.partial(
    sac.train, num_timesteps=10_864_320, num_evals=20, reward_scaling=5, 
    episode_length=1000, normalize_observations=True, action_repeat=1, 
    discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128, 
    grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=1048576, 
    min_replay_size=8192, checkpoint_logdir='/home/user/bird/logs/sac_2', seed=1)

if __name__ == '__main__':
  envs.register_environment('bird', Bird)
  env_name = 'bird'
  env = envs.get_environment(env_name)

  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]

  def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])
    print(num_steps)
    print(f'distance_from_target {metrics['eval/episode_distance_from_target']}')

  make_inference_fn, params, m= train_fn(environment=env, progress_fn=progress)

  model_path = 'tmp/mjx_brax_sac.pkl'
  model.save_params(model_path, params)

  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')


  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.title(f'y={y_data[-1]:.3f}')

  plt.errorbar(
      x_data, y_data, yerr=ydataerr)
      
  plt.autoscale(enable=True, axis='both', tight=True)
  plt.show()

  print(x_data, y_data)