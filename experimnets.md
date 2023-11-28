First Experiment:
500 episodes
Keeps running to the left 
reward function: reward = 0.5 * self.mario.score + 0.5 * self.mario.level_progress + 10 * self.mario.time_left


Second Experiment:
300 episodes
Just stands still
reward functio: self.mario.level_progress
maybe the learning rate is too high