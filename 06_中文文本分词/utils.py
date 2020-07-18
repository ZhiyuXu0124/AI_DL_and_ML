import os
import pickle
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

def _get_label(word):
    length = len(word)
    if length==1:
        return "s" 
    elif length==2:
        return "be" 
    else:
        return "b" + "m"*(length-2)+"e"
    
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

