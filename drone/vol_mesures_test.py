def vol_mesures_test(bebopVision, args):
    bebop = args[0]
    bebop.safe_takeoff(10)
    bebop.smart_sleep(1)
    compteur = 50
    bebop.set_max_tilt(15) #entre 5 et 30
    bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-5, duration=4)
    img = userVision.vision.get_latest_valid_picture()   #[:512]
    bebop.fly_direct(0, 0, 0, 50, 2)
    filename = r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\traitement\mesures\test_image_%06d.jpg' % compteur
    cv2.imwrite(filename, img)
    compteur += 1
    bebop.smart_sleep(2)
    bebop.fly_direct(50, 0, 0, 0, 1)
    dico = []
    t = time.time()
    while time.time() - t < 2:
        dico.append(bebop.sensors.sensors_dict)

    img = userVision.vision.get_latest_valid_picture()  #[:512]
    filename = r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\traitement\mesures\test_image_%06d.jpg' % compteur
    cv2.imwrite(filename, img)
    compteur += 1
    bebop.fly_direct(50, 0, 0, 0, 1)
    bebop.smart_sleep(2)
    img = userVision.vision.get_latest_valid_picture()   #[:512]
    filename = r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\traitement\mesures\test_image_%06d.jpg' % compteur
    cv2.imwrite(filename, img)
    bebop.smart_sleep(2)
    bebop.fly_direct(0, 0, 0, -50, 2)
    bebop.smart_sleep(2)
    bebop.safe_land(5)
    bebop.disconnect()

