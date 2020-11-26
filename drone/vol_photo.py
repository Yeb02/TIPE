import sys
sys.path.append(r'/home/edern/opencv_build/opencv/build/python_loader')
import cv2, pygame, time

def vol_photo(bebopVision, args):
    bebop = args[0]
    bebop.smart_sleep(2)
    # bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)

    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("help me")
    pygame.event.get()
    userVision = UserVision(bebopVision)
    run = True
    a = 50
    while run:
        pygame.event.get()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: #z
            run = False
        if keys[pygame.K_s]: #s
            img = userVision.vision.get_latest_valid_picture()[:511]
            filename = r'/home/edern/Documents/TIPE/traitement/mesures/test_image_%06d.jpg' % a
            cv2.imwrite(filename, img)
            a += 1
            print(a)
        pygame.time.delay(150)
    pygame.quit()
    bebop.smart_sleep(3)
    bebop.disconnect()


