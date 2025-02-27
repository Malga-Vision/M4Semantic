'''
Note: This script requires two specific libraries:
    * h5py for opening Mid-Air data records
    * pyquaternion for quaternion operations
Both can be installed with pip:
$ pip install pyquaternion h5py
'''

import os
import argparse
import h5py
#from pyquaternion import Quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","Ninja2"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "ninja2"]), help="path to folder to store csv files")
a = parser.parse_args()

FRAME_SKIP = 1 # Downsample framerate
#FRAME_SKIP = [1,2,3,1,2,3,1]

if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)

    #data = ["seq0000", "seq0001", "seq0002", "seq0003", "seq0004", "seq0005", "seq0006"]
    data = ["img"]
    #sensors = [["images", ".png"], ["depth", ".png"]]
    

    for nset, set in enumerate(data):
        print("Processing %s" % (set))
        
        out_dir = a.output_dir
        file_name = os.path.join(out_dir, "traj_%s.csv" % str(nset).zfill(4))
        trans = [0,0,0]
        quat = [0,0,0,1]
        # Create csv file
        with open(file_name, 'w') as file:
            file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
            
            
            
            ##
            
            
            imgs = os.listdir("/media/DATA_4TB/Yara/Ninja2/img")
            traj_len = len(imgs)
            
            print("Trajectory Length: ")
            print(traj_len)
            

            # Iterate over sequence samples
            for index in range(traj_len):
                # Compute frame-to-frame camera motion
                
                camera_l = os.path.join("img", imgs[index])
                
                # Write sample to file
                file.write("%i\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, quat[3], quat[0], quat[1], quat[2], trans[0], trans[1], trans[2]))

