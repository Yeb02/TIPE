def vol_suivi(bebopVision, args):
    bebop = args[0]
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("help me")
    pygame.event.get()
    bebop.safe_takeoff(10)
    bebop.smart_sleep(1)
    run = True
    v = 50
    suivi = False
    bebop.set_max_tilt(15) #entre 5 et 30
    # userVision = UserVision(bebopVision)
    positions = np.zeros([4])
    nb = 0
    lower = np.array([0 ,190 ,90])
    upper = np.array([10 ,220 ,115])

    while run:
        dicti = bebop.sensors.sensors_dict
        positions += np.array([dicti['GpsLocationChanged_latitude'], dicti['GpsLocationChanged_longitude'], dicti['GpsLocationChanged_altitude'], 1])
        pygame.event.get()
        controls = np.zeros([5])

        if suivi:
            #[:511] normalement, mais en cas d' artefacts.
            t = time.time()
            i = cv2.cvtColor(cv2.resize(userVision.vision.get_latest_valid_picture()[:500], (60, 100)), cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(i, lower, upper)
            c = np.zeros([3])
            for a in range(100):
                for b in range(60):
                    if mask[a, b] == 255:
                        c += np.array([a, b, 1])
            if c[2] < 20:
                suivi = False
            n = c[2]
            c *= 1 / c[-1]
            co = [int(np.floor(c[0])), int(np.floor(c[1]))]
            if  not 20 < co[0] < 40:
                controls += np.array(0, 0, 0, [3 * (30 - co[0]), 0])
            if  not 40 < co[0] < 60:
                controls += np.array([0, 0, 0, 3 * (50 - co[0]), 0])
            if not nb - 20 < n:
                controls += np.array([0, 30, 0, 0, 0])
            if not n < nb + 20:
                controls += np.array([0, -30, 0, 0, 0])

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]: #a
            controls += np.array([0, 0, 0, -v, 0])    # fly_direct(roll, pitch, yaw, vertical_movement, duration)
        if keys[pygame.K_w]: #z
            controls += np.array([0, v, 0, 0, 0])
        if keys[pygame.K_e]: #e
            controls += np.array([0, 0, 0, v, 0])
        if keys[pygame.K_a]: #q
            controls += np.array([-v, 0, 0, 0, 0])
        if keys[pygame.K_s]: #s
            controls += np.array([0, -v, 0, 0, 0])
        if keys[pygame.K_d]: #d
            controls += np.array([v, 0, 0, 0, 0])
        # if keys[pygame.K_UP]: #fleche haute
        #     suivi = not suivi
        if keys[pygame.K_DOWN]: #fleche basse
            run = False

        i = []
        for k in range(4):
            if not -100 < controls[k] < 100:
                i.append(int(np.sign(controls[k]) * 100))
            else:
                i.append(int(controls[k]))
        i.append(0.06)
        bebop.fly_direct(i[0], i[1], i[2], i[3], i[4])
        pygame.time.delay(50)

    pygame.quit()
    bebop.smart_sleep(3)
    #img1 = userVision.vision.get_latest_valid_picture()
    # cv2.imwrite(r"C:\Users\alpha\Desktop\Informatique\TIPE\Images\jpg\test4.jpg", userVision.vision.get_latest_valid_picture())
    #cv2.imshow('miracle', img1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    bebop.safe_land(5)
    return positions
    error_trigger
    bebopVision.close_video()
    bebop.disconnect()
