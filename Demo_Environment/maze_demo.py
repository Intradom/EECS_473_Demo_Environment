# Code taken and modified from https://pythonspot.com/en/maze-in-pygame/

from pygame.locals import *
import pygame
import time
import numpy as np
import win32api

# Parameters
WALL_SIZE = 64
PLAYER_SIZE = 32
GOAL_SIZE = 32
MOUSE_DEADZONE = 1 # In pixels

class Player:
    x = WALL_SIZE * 1.5
    y = WALL_SIZE * 1.5
    speed = 10
 
class Maze:
    def __init__(self):
       self.rows = 10
       self.cols = 10
       
       # Always leave upper left area 2nd row, 2nd col blank b/c player spawns there
       # Always leave lower right area 2nd last row, 2nd last col blank b/c goal resides there
       self.grid = [ 1,1,1,1,1,1,1,1,1,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,0,1,1,1,1,1,1,0,1,
                     1,0,1,0,0,0,0,0,0,1,
                     1,0,1,0,1,1,1,1,0,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,1,1,1,1,1,1,1,1,1,]
 
    def draw(self,display_surf,image_surf):
       bx = 0
       by = 0
       for i in range(0,self.rows*self.cols):
           if self.grid[ bx + (by*self.cols) ] == 1:
               display_surf.blit(image_surf,( bx * WALL_SIZE, by * WALL_SIZE))
 
           bx = bx + 1
           if bx > self.cols-1:
               bx = 0 
               by = by + 1
 
 
class App:
 
    windowWidth = 640
    windowHeight = 640
    player = 0
    goal_x = windowWidth - WALL_SIZE * 1.5
    goal_y = windowHeight - WALL_SIZE * 1.5
 
    def __init__(self):
        self._running = True
        self._display_surf = None
        self._player_surf = None
        self._block_surf = None
        self._goal_surf = None
        self.player = Player()
        self.maze = Maze()
        
        self.collision_count = 0
        self.start_time = time.time()
        
        self.in_collision = False
        self.goal_reached = False
 
    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
 
        pygame.display.set_caption('Headgear Demo Environment')
        self._running = True
        self._player_surf = pygame.image.load("Maze_Demo_Images/player.png").convert_alpha()
        self._goal_surf = pygame.image.load("Maze_Demo_Images/goal.png").convert_alpha()
        self._block_surf = pygame.image.load("Maze_Demo_Images/wall.png").convert()
 
    def on_render(self):
        if (not self.goal_reached):
            self._display_surf.fill((0,0,0))
            self._display_surf.blit(self._player_surf,(self.player.x - PLAYER_SIZE / 2.0,self.player.y - PLAYER_SIZE / 2.0))
            self._display_surf.blit(self._goal_surf,(self.goal_x - GOAL_SIZE / 2.0,self.goal_y - GOAL_SIZE / 2.0))
            self.maze.draw(self._display_surf, self._block_surf)
        else:
            self._display_surf.fill((255, 255, 255))

        pygame.display.flip()
 
    def on_cleanup(self):
        pygame.quit()
    
    """
    # From https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
    def circle_rect_collision(self, cir_x, cir_y, cir_rad, rect_x, rect_y, rect_w, rect_h):
        cir_dist_x = abs(cir_x - rect_x);
        cir_dist_y = abs(cir_y - rect_y);

        if (cir_dist_x > (rect_w + circle.r)):
            return False
        if (cir_dist_y > (rect_h + circle.r)):
            return False

        if (cir_dist_x <= (rect_w / 2.0)):
            return True 
        if (cir_dist_y <= (rect_h / 2.0)):
            return True

        corner_dist_sq = (cir_dist_x - rect_w / 2.0)^2 + (cir_dist_y - rect_h / 2.0)^2;

        return (corner_dist_sq <= (circle.r^2))
    """

    def scale_move(self, mx, my):
       
        move_dist = np.linalg.norm((self.player.x - mx, self.player.y - my))
        if (move_dist > self.player.speed):
            move_dist = self.player.speed

        return move_dist

    def move_intent(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        x_offset = 0
        y_offset = 0
        if (self.player.x > mouse_x + MOUSE_DEADZONE):
            x_offset = -self.scale_move(mouse_x, self.player.y)
        elif (self.player.x < mouse_x - MOUSE_DEADZONE):
            x_offset = self.scale_move(mouse_x, self.player.y)
        elif (self.player.y > mouse_y + MOUSE_DEADZONE):
            y_offset = -self.scale_move(self.player.x, mouse_y)
        elif (self.player.y < mouse_y - MOUSE_DEADZONE):
            y_offset = self.scale_move(self.player.x, mouse_y)

        #print('----')
        #print(x_offset)
        #print(y_offset)
        return x_offset, y_offset
    
    def wall_block(self, x, y):
        # Based on coordinates, find what index the wall would be in
        x_ind = int(x / WALL_SIZE)
        y_ind = int(y / WALL_SIZE)
        
        if (self.maze.grid[y_ind * self.maze.rows + x_ind] == 1):
            return True
            
        return False
 
    def collision_in_dir(self, x_offset, y_offset, boundary_check):    
        """
        angle_rad = 0
        if (x_offset != 0):
            angle_rad = np.arctan(y_offset / x_offset)
        else:
            if (y_offset > 0):
                angle_rad = np.pi / 4.0
            else:
                angle_rad = np.pi * 3 / 4.0
            
        if (x_offset < 0):
            angle_rad += np.pi / 2.0
        """
        
        x_extension = 0
        if (x_offset != 0):
            if (y_offset != 0):
                # 45 degrees
                x_extension = PLAYER_SIZE / 2 * np.cos(45) * np.sign(x_offset)
            else:
                # 90 degrees
                x_extension = PLAYER_SIZE / 2 * np.sign(x_offset)
        
        y_extension = 0
        if (y_offset != 0):
            if (x_offset != 0):
                # 45 degrees
                y_extension = PLAYER_SIZE / 2 * np.cos(45) * np.sign(y_offset)
            else:
                # 90 degrees
                y_extension = PLAYER_SIZE / 2 * np.sign(y_offset)
        
        new_x_extended = self.player.x + x_extension
        new_y_extended = self.player.y + y_extension
        if (not boundary_check):
            new_x_extended += x_offset
            new_y_extended += y_offset
        else:
            new_x_extended += np.sign(x_offset)
            new_y_extended += np.sign(y_offset)
    
         # Check if new x and y of furthest player point are inside a wall
        collided = self.wall_block(new_x_extended, new_y_extended)
            
        #print(x_offset)
        #print(y_offset)
        #print(x_extension)
        #print(y_extension)
        #print(new_x_extended)
        #print(new_y_extended)
        #print(collided)
        return collided
        
    def goal_collided(self):
        # Circle circle collision
        if (np.linalg.norm(np.array((self.player.x - self.goal_x, self.goal_y - self.player.y))) <= (PLAYER_SIZE / 2.0 + GOAL_SIZE / 2.0)):
            return True
        
        return False
 
    def wait_seconds(self, num_secs):
        if (num_secs == 0):
            return
        
        print(num_secs)
        time.sleep(1)
        self.wait_seconds(num_secs - 1)

    def on_execute(self):
        # Init
        if self.on_init() == False:
            self._running = False
 
        self.wait_seconds(3)
        # Set the mouse cursor to player x and y, TODO
        
        print("--------------------------------")

        # Game loop
        while( self._running ):
            for event in pygame.event.get():
                if event.type == QUIT:
                    self._running = False
                 
            if (not self.goal_reached):
                # Move player if possible
                x_offset, y_offset = self.move_intent()
                if (not self.collision_in_dir(x_offset, y_offset, False)):
                    self.player.x += x_offset
                    self.player.y += y_offset
                    
                # Check for collision in all directions
                any_collision = self.collision_in_dir(1, 0, True) or self.collision_in_dir(-1, 0, True) or self.collision_in_dir(0, 1, True) or self.collision_in_dir(0, -1, True)
                #print(any_collision)
                if (any_collision and not self.in_collision):
                    self.collision_count += 1
                    #print(self.collision_count)
                self.in_collision = any_collision 
              
                # Check if on goal
                if ((self.goal_collided())):
                    print("Elapsed time: {0:.0f}".format(time.time() - self.start_time))
                    print("Total collisions: {0}".format(self.collision_count))
                    self.end_timer = time.time()
                    self.goal_reached = True
    
            if (self.goal_reached):
                if (time.time() - self.end_timer > 3):
                    # Reset everything
                    self.player.x = WALL_SIZE * 1.5
                    self.player.y = WALL_SIZE * 1.5
                    # Set the mouse cursor to player x and y, TODO
                    
                    self.start_time = time.time()
                    self.collision_count = 0
                    self.goal_reached = False
                    print("--------------------------------")
    
            self.on_render()
            
        # End
        self.on_cleanup()
 
if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()
