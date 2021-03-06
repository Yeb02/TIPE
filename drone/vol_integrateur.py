def vol_integrateur(bebopVision, args):   #trouver le probleme !!!
    global run_glob, tilt_glob, mesures_glob_gps, mesures_glob_accel, bebop, tilt_glob, vitesse_glob, auto_mode_glob, acceleration_glob, refresh_glob
    run_glob = True
    auto_mode_glob = False
    mesures_glob_gps = False
    mesures_glob_accel = False
    acceleration_glob = np.zeros([3]) #X, Y, Z pour l'instant.
    vitesse_glob = 50  #entre 0 (immobile) et 100.
    tilt_glob = 5   #entre 5 et 30.  30 = rapide, 5 = lent. Mieux vaut 5 pour voler près d'obstacles.
    refresh_glob = .1  #la vitesse de la boucle capteurs + commande, sachant que le bebop 2 refresh à 10 Hz.
    bebop.set_max_tilt(tilt_glob)

    def controles():
        global run_glob, tilt_glob, mesures_glob_gps, mesures_glob_accel, bebop, tilt_glob, vitesse_glob, auto_mode_glob, acceleration_glob, refresh_glob
        positions = np.zeros([4])
        while run_glob:
            pygame.event.get()
            ctrl = np.zeros([4])   #controles. 0 = haut, 1 = avant, 2 = tourner, 3 = droite, 4 = temps d'éxécution
            keys = pygame.key.get_pressed()

            if mesures_glob_gps:
                dicti = bebop.sensors.sensors_dict
                positions += np.array([dicti['GpsLocationChanged_latitude'],
                dicti['GpsLocationChanged_longitude'], dicti['GpsLocationChanged_altitude'], 1])  # trouver le sonar.
            if mesures_glob_accel:
                dicti = bebop.sensors.sensors_dict
                acceleration_glob += np.array([dicti['SpeedChanged_speedX'], dicti['SpeedChanged_speedY'], dicti['SpeedChanged_speedZ'], 1])
            global vitesse_glob
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
                mesures_glob_accel = not mesures_glob_accel
                bebop.smart_sleep(2)
                ctrl = np.zeros([4])
            if keys[pygame.K_UP]:
                mesures_glob_gps = not mesures_glob_gps
            if keys[pygame.K_DOWN]:
                run_glob = not run_glob
            if keys[pygame.K_ESCAPE]:
                auto_mode_glob = not auto_mode_glob

            bebop.fly_direct(ctrl[0], ctrl[1], ctrl[2], ctrl[3], .1)
            pygame.time.delay(refresh_glob)

        return positions

    def auto_mode():   #à faire. Utiliserai le ML pour éviter les obstacles, sur un parcours prédéfini
                        #par des injonctions de mouvement ou des cos GPS.
        global run_glob, tilt_glob, mesures_glob_gps, bebop, tilt_glob, vitesse_glob, auto_mode_glob
        sift = cv2.xfeatures2d.SIFT_create()
        color_pics = []
        while run_glob:
            if auto_mode_glob:
                pass
        time.sleep(.5)

    bebop = args[0]
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("control window")
    pygame.event.get()
    bebop.smart_sleep(1)

    def integrateur():
        global run_glob, refresh_glob, acceleration_glob
        vitesse = [np.array([0, 0, 0])]
        position = np.zeros([3])
        while run_glob:
            if mesures_glob_accel:
                vitesse.append(vitesse[-1] + refresh_glob * acceleration_glob)
                position += vitesse[-1] * refresh_glob
            time.sleep(refresh_glob)
        return position

    with concurrent.futures.ThreadPoolExecutor() as executor:   #ne sort pas du contexte manager tant que tout les programmes n' ont pas fini.
        thread_1 = executor.submit(controles)
        # thread_2 = executor.submit(auto_mode)
        thread_3 = executor.submit(integrateur)
        position_integree = thread_3.result()
        position_gps = thread_1.result()

    print('position intégrée =', position_integree)
    def end_flight():
        global bebop
        bebop.smart_sleep(3)
        if bebop.safe_land(5):   #n' enleve le controle à l'utilisateur que si le drone a bien atteri.
            pygame.quit()
            bebop.disconnect()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        thread_1 = executor.submit(end_flight)
        # thread_2 = executor.submit(post_traitement, position)  #afficher l' aller et le retour
