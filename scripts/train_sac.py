from brax.training.agents.sac import train as sac
from brax.training.agents.sac import networks as sac_networks
import functools
from datetime import datetime
import matplotlib.pyplot as plt
from scripts.model import Bird
from brax import envs
from brax.io import model
import yaml

config_path = 'config/config.yaml'
with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

make_networks_factory = functools.partial(
    sac_networks.make_sac_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))


train_fn = functools.partial(
    sac.train,
    num_timesteps = config['sac']['num_timesteps'],
    num_evals=config['sac']['num_evals'],
    reward_scaling=config['sac']['reward_scaling'],
    episode_length=config['sac']['episode_length'], 
    normalize_observations=config['sac']['normalize_observations'], 
    action_repeat=config['sac']['action_repeat'],
    discounting=config['sac']['discounting'],
    learning_rate=config['sac']['learning_rate'], 
    num_envs=config['sac']['num_envs'], 
    batch_size=config['sac']['batch_size'],
    grad_updates_per_step=config['sac']['grad_updates_per_step'], 
    max_devices_per_host=config['sac']['max_devices_per_host'], 
    max_replay_size=config['sac']['max_replay_size'], 
    min_replay_size=config['sac']['min_replay_size'], 
    checkpoint_logdir=config['sac']['checkpoint_logdir'], 
    seed=config['sac']['seed'])

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