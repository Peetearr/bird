import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import time
import pickle
import matplotlib.pyplot as plt

def control(err, derr, kp = 1, kd = .01, A_0 = 1.25):
    return A_0 - (kp*err + kd*derr)

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)
total_mass = 0.0
for body_id in range(m.nbody):
    body_mass = m.body_mass[body_id]
    total_mass += body_mass
    print(f"Body {body_id} ({m.body(body_id).name}): mass = {body_mass:.4f} kg")
print(f'total mass: {total_mass}')

observations = []
actions = []
z_pos = []
t = []
ydataerr = []
h_target = -.2

t_0 = 0
file_name = 'demos'

viewer = mujoco.viewer.launch_passive(m,d)
renderer = mujoco.Renderer(m, 400, 600)
while True:
    err = -h_target-d.qpos[1].copy()
    d_err = d.qvel[1].copy()
    pitch_err = -d.qpos[2].copy() - .5*d.qpos[0].copy()
    pitch_d_err = -d.qvel[2].copy() - 2*d.qvel[0].copy()
    w = 32
    A = control(err, d_err)
    tail_pitch = control(pitch_err, pitch_d_err, A_0=0, kp=1, kd=.5)

    obs = np.concatenate([
        d.qpos[:6],
        d.qacc[3:6],
        d.qvel[:6],
        d.qfrc_actuator[3:]
    ])
    d.ctrl[0] = -1*np.sin(d.time*w)
    d.ctrl[1] = -0*np.sin(d.time*w)+.1
    d.ctrl[2] = A*np.sin(d.time*w+1.57)
    d.ctrl[3] = -1*np.sin(d.time*w)
    d.ctrl[4] = 0*np.sin(d.time*w)-.1
    d.ctrl[5] = -A*np.sin(d.time*w+1.57)

    d.ctrl[6] = tail_pitch

    # act = d.ctrl[:4]
    mujoco.mj_step(m, d)
    z_pos.append(-d.qpos[1])
    t.append(d.time)
    ydataerr.append(0.0)
    viewer.sync()
    time.sleep(.01)
    # actions.append(act)
    observations.append(obs)
    if d.time - t_0 > 5:
        t_0 = d.time
        print('saving')
        # demos = {
        #         'observation': np.array(observations),
        #         'action': np.array(actions),
        #         }
        # with open('expert_traj/' + file_name + '.pkl', 'wb') as f:
        #     pickle.dump(demos, f)

    if d.time > 30:
        plt.xlabel('environment steps')
        plt.ylabel('height')
        plt.title(f'y={z_pos[-1]:.3f}')
        plt.errorbar(
            t, z_pos, yerr=ydataerr)
        plt.axhline(y=h_target, color='red', linewidth=1, label='reference')
        plt.margins(y=0.1)
        plt.show()
        break