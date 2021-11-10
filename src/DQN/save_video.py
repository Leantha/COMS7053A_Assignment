import gym
import cv2
"""
Help from Simon in class who shared his wrapper openly to the group
"""

cv2.ocl.setUseOpenCL(False)


class RenderRGB(gym.Wrapper):
    def __init__(self, env, key_name="pixel"):
        super().__init__(env)
        self.last_pixels = None
        self.viewer = None
        self.key_name = key_name

        render_modes = env.metadata['render.modes']
        render_modes.append("rgb_array")
        env.metadata['render.modes'] = render_modes

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_pixels = obs[self.key_name]
        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        img = self.last_pixels

        # Hacky but works
        if mode != "human":
            return img
        else:
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def reset(self):
        obs = self.env.reset()
        self.last_pixels = obs[self.key_name]
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


pixel_obs = "pixel_crop" # Or could be be 'pixel_crop'
env = gym.make(env, observation_keys=("glyphs", "blstats", pixel_obs))
env.seed(seed)
env = RenderRGB(env, pixel_obs)
env = gym.wrappers.Monitor(env, "recordings", force=True)
