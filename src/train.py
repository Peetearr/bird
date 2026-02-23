import sys
import os
from pathlib import Path
import argparse
import yaml

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from scripts.train_sac import train_fn as train_fn_sac
from scripts.train_ppo import train_fn as train_fn_ppo
from scripts.model import Bird
from brax import envs
from datetime import datetime
import matplotlib.pyplot as plt


# Выбор политики алгоритма
parser = argparse.ArgumentParser(description='Алгоритм')
parser.add_argument('--alg', type=str, default='sac',
                        help='Название алгоритма (sac/ppo)')
parser.add_argument('--config_path', type=str, default='config/config.yaml')
args = parser.parse_args()

with open(args.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

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

if args.alg == 'sac':
    train_fn = train_fn_sac
    make_inference_fn, params, m= train_fn(environment=env, progress_fn=progress,
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
elif args.alg == 'ppo':
    train_fn = train_fn_ppo
    make_inference_fn, params, m= train_fn(environment=env, progress_fn=progress,
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
else:
    print('неверно указан алгоритм. По умолчанию sac')
    train_fn = train_fn_sac
    make_inference_fn, params, m= train_fn(environment=env, progress_fn=progress)

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