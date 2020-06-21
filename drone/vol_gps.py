import cv2, pygame, time

def vol_gps(bebopVision, args):
    bebop = args[0]
    bebop.smart_sleep(5)
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("help me")
    pygame.event.get()
    positions = np.zeros([4])
    run = True
    while run:
        pygame.event.get()
        dicti = bebop.sensors.sensors_dict
        positions += np.array([dicti['GpsLocationChanged_latitude'], dicti['GpsLocationChanged_longitude'], dicti['GpsLocationChanged_altitude'], 1])
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: #z
            run = False
        pygame.time.delay(250)
    pygame.quit()
    print(positions[:-1] * (1/positions[3]), positions[3])
    bebop.smart_sleep(3)
    bebop.disconnect()