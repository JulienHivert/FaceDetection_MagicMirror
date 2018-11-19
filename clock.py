#! /usr/bin/env python3

import time

class Clock:

    def __init__(self, delta):
        self.__t0 = time.time()
        self.__delta = delta

    def getElapsedTime(self):
        return time.time() - self.__t0

    def timeElapsed(self):
        if self.getElapsedTime() >= self.__delta:
            return True
        return False

    # GETTERS
    def getTime(self):
        return self.__t0

    def getDelta(self):
        return self.__delta

    # SETTERS    
    def setTime(self, t0):
        self.__t0 = t0
    
    def setDelta(self, delta):
        self.__delta = delta