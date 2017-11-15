import win32api
import win32con
import pygame
import time

# Parameters
CURSOR_MOVE_SPEED = 1
DELAY = 0.01 # in seconds, otherwise cursor increments too fast

def main():
    SCREEN_WIDTH = win32api.GetSystemMetrics(0)
    SCREEN_HEIGHT = win32api.GetSystemMetrics(1)
    
    mouse_x = (int) (SCREEN_WIDTH / 2)
    mouse_y = (int) (SCREEN_HEIGHT / 2)
    
    pygame.init()
    screen = pygame.display.set_mode((64, 32))

    while True:
        pressed = pygame.key.get_pressed()
        
        change_pos = False;
        if pressed[pygame.K_w]:
            mouse_y -= CURSOR_MOVE_SPEED
            change_pos = True
        if pressed[pygame.K_s]:
            mouse_y += CURSOR_MOVE_SPEED
            change_pos = True
        if pressed[pygame.K_a]:
            mouse_x -= CURSOR_MOVE_SPEED
            change_pos = True
        if pressed[pygame.K_d]:
            mouse_x += CURSOR_MOVE_SPEED
            change_pos = True
        
        if (mouse_x < 0):
            mouse_x = 0
        if (mouse_y < 0):
            mouse_y = 0
        if (mouse_x > SCREEN_WIDTH):
            mouse_x = SCREEN_WIDTH
        if (mouse_y > SCREEN_HEIGHT):
            mouse_y = SCREEN_HEIGHT
        
        win32api.SetCursorPos((mouse_x,mouse_y))
        
        for event in pygame.event.get():
            
            # determin if X was clicked, or Ctrl+W or Alt+F4 was used 
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, mouse_x, mouse_y, 0, 0)
                    
        time.sleep(DELAY)

if __name__ == "__main__":
    main()