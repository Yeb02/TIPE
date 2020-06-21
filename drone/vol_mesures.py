# sift = cv2.xfeatures2d.SIFT_create()  #mettre nb keypoints dans les parentheses
#a besoin de 'import concurrent.futures'
def vol_mesures(bebopVision, args):
    global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob, auto_mode_glob
    run_glob = True
    auto_mode_glob = False
    mesures_glob = False
    vitesse_glob = 50  #entre 0 (immobile) et 100.
    tilt_glob = 5   #entre 5 et 30.  30 = rapide, 5 = lent. Mieux vaut 5 pour voler près d'obstacles.
    bebop.set_max_tilt(tilt_glob)

    def controles():
        global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob
        positions = np.zeros([4])
        while run_glob:
            pygame.event.get()
            ctrl = np.zeros([4])   #controles. 0 = haut, 1 = avant, 2 = tourner, 3 = droite, 4 = temps d'éxécution
            keys = pygame.key.get_pressed()

            if mesures_glob:
                dicti = bebop.sensors.sensors_dict
                positions += np.array([dicti['GpsLocationChanged_latitude'],
                dicti['GpsLocationChanged_longitude'], dicti['GpsLocationChanged_altitude'], 1])
                # trouver le sonar.
                # pas besoin d'aussi rapide, et le bebop 2 refresh à 10 Hz
            v = vitesse_glob
            if keys[pygame.K_a]:
                ctrl[3] = - v
            if keys[pygame.K_z]:
                ctrl[1] = v
            if keys[pygame.K_e]:
                ctrl[3] = v
            if keys[pygame.K_q]:
                ctrl[0] = - v
            if keys[pygame.K_s]:
                ctrl[1] = - v
            if keys[pygame.K_d]:
                ctrl[0] = v
            if keys[pygame.K_w]:
                ctrl[2] = v * np.pi / 100
            if keys[pygame.K_x]:
                ctrl[2] = - v * np.pi / 100
            if keys[pygame.K_SPACE]:
                bebop.safe_land(1)
                bebop.safe_takeoff(1)
                bebop.smart_sleep(2)
                ctrl = np.zeros([4])
            if keys[pygame.K_UP]:
                mesures_glob = not mesures_glob
            if keys[pygame.K_DOWN]:
                run_glob = not run_glob
            if keys[pygame.K_ESCAPE]:
                auto_mode_glob = not auto_mode_glob

            bebop.fly_direct(ctrl[0], ctrl[1], ctrl[2], ctrl[3], .1)
            pygame.time.delay(100)

        return positions

    def auto_mode():
        global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob, auto_mode_glob
        sift = cv2.xfeatures2d.SIFT_create()
        color_pics = []

        while run_glob:
            if auto_mode_glob:
                pass


# reference d' enregistrement dans vol photo.

        time.sleep(.5)

    bebop = args[0]
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("control window")
    pygame.event.get()
    bebop.smart_sleep(1)


    with concurrent.futures.ThreadPoolExecutor() as executor:   #ne sort pas du contexte manager tant que tout les programmes n' ont pas fini.
        thread_1 = executor.submit(controles)
        thread_2 = executor.submit(auto_mode)
        position = thread_1.result()

    bebop.smart_sleep(3)
    if bebop.safe_land(5):   #n' enleve le controle à l'utilisateur que si le drone a bien atterit.
        pygame.quit()
        bebop.disconnect()

