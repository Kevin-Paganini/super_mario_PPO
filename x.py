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
mario.set_world_level(1, 1)
mario.start_game() 


def generate_episode():
    

    assert mario.score == 0
    assert mario.lives_left == 2
    assert mario.time_left == 400
    last_lives_left = mario.lives_left
    print('falalala', dir(mario), 'falalalala')
    print('Bot view: ', np.array(mario.game_area()))   
    while mario.lives_left > 0:
        
        


        pyboy.tick()
        
    

    mario.reset_game()
    assert mario.lives_left == 2


 
generate_episode()

pyboy.stop()
