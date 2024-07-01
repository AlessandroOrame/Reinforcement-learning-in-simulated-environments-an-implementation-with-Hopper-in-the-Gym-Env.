"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
import gym
import numpy as np
from gym import utils
from copy import deepcopy
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, random=False):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0
        
        self.random = random 
        if self.random:    
            # Define mass ranges for the thigh, leg, and foot links
            self.mass_ranges = {
                'thigh': (self.original_masses[1] - 0.5, self.original_masses[1] + 0.5),
                'leg': (self.original_masses[2] - 0.5, self.original_masses[2] + 0.5),
                'foot': (self.original_masses[3] - 0.5, self.original_masses[3] + 0.5),
            }


    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        
        #
        # TASK 6: implement domain randomization. Remember to sample new dynamics parameter
        #         at the start of each training episode.

        # Sample new masses from the defined ranges 
        new_masses = np.array([
            self.sim.model.body_mass[1],  # Keep the torso mass fixed
            np.random.uniform(*self.mass_ranges['thigh']),
            np.random.uniform(*self.mass_ranges['leg']),
            np.random.uniform(*self.mass_ranges['foot'])
        ])

        return new_masses


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


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
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


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


        self.set_state(qpos, qvel)
        if self.random == True:    
            self.set_random_parameters()  # Apply randomization at the start of each episode
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
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="CustomHopper-source-random-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "random": True}
)
