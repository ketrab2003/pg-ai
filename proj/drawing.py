import pygame
import threading
from PIL import Image
from recognizer import recognize_digit
import time

RECOGNITION_DELAY = 0.1

screen = pygame.display.set_mode((500, 500))
drawing_changed = False
last_recognition_time = 0

def recognize():
    global drawing_changed

    string_img = pygame.image.tostring(screen, 'RGB')
    img = Image.frombytes('RGB', (500, 500), string_img)
    digit = recognize_digit(img)
    pygame.display.set_caption(f'The digit in the image is: {digit}')
    drawing_changed = False

def main():
    pygame.init()
    pygame.display.set_caption('Drawing')
    screen.fill((255, 255, 255))
    clock = pygame.time.Clock()
    running = True
    while running:
        running = main_loop(screen, clock)
    pygame.quit()

mouse_left_down = False
mouse_right_down = False

def main_loop(screen: pygame.Surface, clock: pygame.time.Clock):
    global mouse_left_down, mouse_right_down, drawing_changed, last_recognition_time
    running = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_left_down = True
            if event.button == 3:
                mouse_right_down = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_left_down = False
            if event.button == 3:
                mouse_right_down = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill((255, 255, 255))
                pygame.display.set_caption('Drawing')
            if event.key == pygame.K_r:
                threading.Thread(target=recognize).start()
            if event.key == pygame.K_q:
                running = False

    if mouse_left_down:
        pygame.draw.circle(screen, (0, 0, 0), pygame.mouse.get_pos(), 15)
        drawing_changed = True
    if mouse_right_down:
        pygame.draw.circle(screen, (255, 255, 255), pygame.mouse.get_pos(), 15)
        drawing_changed = True

    if drawing_changed and time.time() - last_recognition_time > RECOGNITION_DELAY:
        last_recognition_time = time.time()
        threading.Thread(target=recognize).start()

    pygame.display.flip()
    clock.tick(60)

    return running

if __name__ == '__main__':
    main()