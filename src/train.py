import sys
from pathlib import Path
import argparse

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
    print(f'velocity {metrics['eval/episode_z_position']}')

if args.alg == 'sac':
    train_fn = train_fn_sac
elif args.alg == 'ppo':
    train_fn = train_fn_ppo
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