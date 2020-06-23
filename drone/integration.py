import numpy as np

def intégrateur(acceleration_array, v0, co0, deltaT):  #arrays internes de dimension 3
    for k in range(acceleration_array):
        co0 += deltaT * v0
        v0 += deltaT * acceleration_array[k]
    return(co0, v0)
