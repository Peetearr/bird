import jax
from jax import numpy as jp

import mujoco
from mujoco import mjx

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


class Bird(PipelineEnv):

  def __init__(
      self,
      forward_reward_weight=1,
      ctrl_cost_weight=0,#0.1,
      healthy_reward=0.0,
      terminate_when_unhealthy=False,
      healthy_z_range=(-1.0, 1.0),
      reset_noise_scale=1e-1,
      exclude_current_positions_from_observation=False,
      phys_weight = 1e-4,
      **kwargs,
  ):
#
    mj_model = mujoco.MjModel.from_xml_path(
        'model/scene.xml')
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 1
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)
    self.action_map = {
            0: [0, 3],
            1: [1, 4],
            2: [2, 5],  # action[0] управляет приводами 0 и 1
            6: [6],
            7: [7]
        }
    self._action_size = len(self.action_map)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._phys_weight = phys_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )
    
  @property
  def action_size(self) -> int:
    return 5

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )
    qpos = qpos.at[3:].set(0.0)
    qvel = qvel.at[3:].set(0.0)

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'z_position': zero,
        'distance_from_target': zero
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    action = jp.array([action[0], action[1], action[2], action[0], action[1], -action[2], action[3], action[4]])
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    com_after = data.qpos[:2]

    # flap_pos_l = data.qpos[4]
    # flap_pos_r = data.qpos[7]

    # flap_acc_l = data.qacc[4]
    # flap_acc_r = data.qacc[7]

    # N1 = -jp.square(flap_acc_l + (5*2*jp.pi)**2 * flap_pos_l)
    # N2 = -jp.square(flap_acc_r + (5*2*jp.pi)**2 * flap_pos_r)
    distance = -jp.linalg.norm(com_after)
    # velocities = -(jp.square(data.qvel[0]) + jp.square(data.qvel[1]))
    forward_reward = self._forward_reward_weight * distance

    # min_z, max_z = self._healthy_z_range
    # is_healthy = jp.where(data.qpos[2] < min_z, 0.0, 1.0)
    # is_healthy = jp.where(data.qpos[2] > max_z, 0.0, is_healthy)
    # if self._terminate_when_unhealthy:
    #   healthy_reward = self._healthy_reward
    # else:
    #   healthy_reward = self._healthy_reward * is_healthy

    # ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(data, action)
    reward = forward_reward# + healthy_reward - ctrl_cost
    # done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    done = 0.0
    state.metrics.update(
        forward_reward=forward_reward,
        reward_quadctrl=0.0,#-ctrl_cost,
        reward_alive=0.0,#healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        z_position=com_after[2],
        distance_from_target=distance
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray
  ) -> jp.ndarray:
    """Observes humanoid body position, velocities, and angles."""
    position = data.qpos
    if self._exclude_current_positions_from_observation:
      position = position[2:]

    # external_contact_forces are excluded
    return jp.concatenate([
        data.qpos[:6],
        data.qacc[3:6],
        data.qvel[:6],
        data.qfrc_actuator[3:],
    ])