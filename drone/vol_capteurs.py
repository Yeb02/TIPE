
def vol_capteurs(bebopVision, args):
    bebop = args[0]
    bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-5, duration=4)
    dico = []
    t = time.time()
    print(bebop.sensors.sensors_dict)
    time.sleep(1)
    print(bebop.sensors.sensors_dict)
    bebop.smart_sleep(5)
    bebop.disconnect()
