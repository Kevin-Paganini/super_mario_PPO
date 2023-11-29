First Experiment:
500 episodes
Keeps running to the left 
reward function: reward = 0.5 * self.mario.score + 0.5 * self.mario.level_progress + 10 * self.mario.time_left


Second Experiment:
300 episodes
Just stands still
reward functio: self.mario.level_progress
maybe the learning rate is too high

maybe we can add a do nothing button for the bot

stitch five frames together I think

added a do nothing button

could we do a different game representation than game area?

if it stays in the same area punish it

test 7 shows being confused between getting upgrade or progressing right