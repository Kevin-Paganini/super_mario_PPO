#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#

import os
import sys
import numpy as np

# Makes us able to import PyBoy from the directory below
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")

from pyboy import PyBoy, WindowEvent # isort:skip

# Check if the ROM is given through argv
filename = 'super_mario.gb'

quiet = "--quiet" in sys.argv
pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=not quiet, game_wrapper=True)
pyboy.set_emulation_speed(0)
assert pyboy.cartridge_title() == "SUPER MARIOLAN"

mario = pyboy.game_wrapper()
mario.start_game() 


def generate_episode():
    

    assert mario.score == 0
    assert mario.lives_left == 2
    assert mario.time_left == 400
    assert mario.world == (1, 1)
    last_lives_left = mario.lives_left

    print('falalala', dir(mario), 'falalalala')
    print('Bot view: ', np.array(mario.game_area()))   
    # pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
    # for i in range(1000):
    #     print(mario.level_progress)
    #     if mario.lives_left < last_lives_left:
    #         print('Fitness', mario.fitness, 'Fitness last', last_lives_left)
    #         break
    #     else:
    #         last_lives_left = mario.lives_left
            
    #         if i % 100 == 60:
    #             pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #         # if i % 100 == 0:
    #         #     pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

    #     pyboy.tick()
        
    

    mario.reset_game()
    assert mario.lives_left == 2


 
generate_episode()

pyboy.stop()
