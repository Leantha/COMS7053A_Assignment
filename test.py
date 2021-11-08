# Importing helper visualisation functions
#MiniHack-Quest-Hard-v0
#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

# from minihack.tiles.rendering import get_des_file_rendering
#
# import IPython.display
# def render_des_file(des_file, **kwargs):
#     image = get_des_file_rendering(des_file, **kwargs)
#     IPython.display.display(image)

import gym
import minihack
env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=("pixel", "glyphs", "colors", "chars"),
max_episode_steps=100)
env.reset() # each reset generates a new environment instance
env.step(1)  # move agent '@' north
env.render()

from nle import nethack
MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (
nethack.Command.OPEN,
nethack.Command.KICK,
nethack.Command.SEARCH,
)
env = gym.make(
"MiniHack-Quest-Hard-v0",
actions=NAVIGATE_ACTIONS,
)


# from minihack import RewardManager
# reward_gen = RewardManager()
# reward_gen.add_eat_event("apple", reward=1)
# reward_gen.add_wield_event("dagger", reward=2)
# reward_gen.add_location_event("sink", reward=-1, terminal_required=False)
#
# env = gym.make("MiniHackSkill",
#     def_file=des_file.
#     reward_manager=reward_manager)
