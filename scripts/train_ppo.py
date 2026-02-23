from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
import functools
from datetime import datetime
import matplotlib.pyplot as plt
from scripts.model import Bird
from brax import envs
from brax.io import model

from brax.training.agents.bc import train as bc
from brax.training.agents.ppo import losses as ppo_losses  
from brax.training.acme import running_statistics 

# model_path = 'bc_model.pkl'
# params = model.load_params(model_path)
make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
    ppo.train, num_timesteps=100_000_000, num_evals=20, reward_scaling=0.1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=40, num_minibatches=24, num_updates_per_batch=8,
    discounting=0.99, learning_rate=3e-5, entropy_cost=1e-3, num_envs=3072,
    batch_size=256, network_factory=make_networks_factory, save_checkpoint_path='/home/user/bird/logs/ppo_1', seed=0)

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