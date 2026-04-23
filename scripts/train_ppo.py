from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
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
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
    ppo.train,  
    # restore_checkpoint_path='/home/user/bird/logs/ppo_2/best_policy',
    num_timesteps = config['ppo']['num_timesteps'],
    num_evals=config['ppo']['num_evals'],
    reward_scaling=config['ppo']['reward_scaling'],
    episode_length=config['ppo']['episode_length'], 
    normalize_observations=config['ppo']['normalize_observations'], 
    action_repeat=config['ppo']['action_repeat'],
    discounting=config['ppo']['discounting'],
    learning_rate=config['ppo']['learning_rate'], 
    num_envs=config['ppo']['num_envs'], 
    batch_size=config['ppo']['batch_size'],
    entropy_cost=config['ppo']['entropy_cost'],
    unroll_length=config['ppo']['unroll_length'],
    num_minibatches=config['ppo']['num_minibatches'],
    num_updates_per_batch=config['ppo']['num_updates_per_batch'],
    save_checkpoint_path = config['ppo']['save_checkpoint_path'], 
    seed=config['ppo']['seed'])

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
  print(m.keys())

  model_path = 'tmp/mjx_brax_policy_just_aha_aha3.pkl'
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