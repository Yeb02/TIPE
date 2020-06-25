import numpy as np
import threading
from time import sleep
import concurrent.futures

global run_glob
run_glob = True


def wait(x):
    print('starting loop')
    while run_glob:
        print(x)
        sleep(.5)
    print('done looping')
    sleep(2)
    print('done sleeping')
    return('data')

def change():
    print('starting changes')
    sleep(3)
    global run_glob
    run_glob = not run_glob
    print('done changing')


with concurrent.futures.ThreadPoolExecutor() as executor:   #sort quand tout est fini.
    t1 = executor.submit(wait, 'waiting')
    t2 = executor.submit(change)
    print(t1.result())

print('bypassed')

# t1 = threading.Thread(target=wait, args = ['waiting'])
# t2 = threading.Thread(target=change, args = [3])
#
# t1.start()
# t2.start()
#
# t1.join()
# sleep(2)
# run_glob = False
