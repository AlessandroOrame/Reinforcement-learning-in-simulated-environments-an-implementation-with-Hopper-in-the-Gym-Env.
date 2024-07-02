"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy
import numpy as np
import gym
from gymnasium import utils
from .mujoco_env import MujocoEnv

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        self.sim.model.body_mass[1] -= 1.0 
        

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.sim.model.body_mass[1:])
        return masses

    
    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        # Current state observation
        state = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

        # Position of the hopper's torso
        hopper_pos = self.sim.data.get_body_xpos('torso')

        # Position of the obstacles
        obstacle1_pos = self.sim.data.get_body_xpos('obstacle1')
        obstacle2_pos = self.sim.data.get_body_xpos('obstacle2')
        obstacle3_pos = self.sim.data.get_body_xpos('obstacle3')
        
        # Calculate distances from hopper to obstacles in the x-axis
        distance_to_obstacle1 = obstacle1_pos[0] - hopper_pos[0]
        distance_to_obstacle2 = obstacle2_pos[0] - hopper_pos[0]
        distance_to_obstacle3 = obstacle3_pos[0] - hopper_pos[0]

        if distance_to_obstacle1 > -0.2:
            distance = distance_to_obstacle1
        elif distance_to_obstacle2 > -0.2:
            distance = distance_to_obstacle2
        else:
            distance = distance_to_obstacle3
        
        # Append the distance to the state observation
        return np.concatenate([state, [distance]]) #[distance]])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        # set obstacles positions
        obstacle1_body_id = self.sim.model.body_name2id('obstacle1')
        obstacle1_pos = self.sim.model.body_pos[obstacle1_body_id].copy()
        obstacle1_pos[0] = 4
        self.sim.model.body_pos[obstacle1_body_id] = obstacle1_pos

        obstacle2_body_id = self.sim.model.body_name2id('obstacle2')
        obstacle2_pos = self.sim.model.body_pos[obstacle2_body_id].copy()
        obstacle2_pos[0] = 8.5
        self.sim.model.body_pos[obstacle2_body_id] = obstacle2_pos

        obstacle3_body_id = self.sim.model.body_name2id('obstacle3')
        obstacle3_pos = self.sim.model.body_pos[obstacle3_body_id].copy()
        obstacle3_pos[0] = 13
        self.sim.model.body_pos[obstacle3_body_id] = obstacle3_pos

        # Set the new state including the updated obstacle position
        self.set_state(qpos, qvel)
        
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())
        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]
        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()




"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id='CustomHopper-source-v0',
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=1000,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)
