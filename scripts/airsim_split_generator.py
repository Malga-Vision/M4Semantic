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
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","AirsimTopView"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "AirsimTopViewTest"]), help="path to folder to store csv files")
a = parser.parse_args()

#FRAME_SKIP = 1 # Downsample framerate
FRAME_SKIP = [1,2,3,1,2,3,1]

if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)

    data = ["seq0000", "seq0001", "seq0002", "seq0003", "seq0004", "seq0005", "seq0006"]
    #data = ["seq0001"]
    sensors = [["images", ".png"], ["depth", ".png"]]
    

    for nset, set in enumerate(data):
        print("Processing %s" % (set))
        
        out_dir = a.output_dir
        file_name = os.path.join(out_dir, "traj_%s.csv" % str(nset).zfill(4))
        trans = [0,0,0]
        # Create csv file
        with open(file_name, 'w') as file:
            file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "depth", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
            def get_path(sensor, index, ext):
                im_name = str(index).zfill(6) + "." + ext
                path = os.path.join(*[set, sensor, im_name])
                return path
            
            
            ##
            x = pd.read_csv("/media/DATA_4TB/Yara/AirsimTopView/"+set+"/airsim_rec.txt",sep = "\t")
            #traj_len = len(x)
            traj_len = len(x)//FRAME_SKIP[nset]
            
            
            print("Trajectory Length: ")
            print(traj_len)
            

            # Iterate over sequence samples
            for index in range(traj_len-1):
                # Compute frame-to-frame camera motion
                i = index*FRAME_SKIP[nset]
                ##
                #f1 = open(os.path.join(*[a.db_path, set, 'metadata',str(i).zfill(6)+".json"]))
                #data1 = json.load(f1)
                #f2 = open(os.path.join(*[a.db_path, set, 'metadata',str(i+FRAME_SKIP).zfill(6)+".json"]))
                #data2 = json.load(f2)
                ##
                # get rotation and translation between two consecutive frames
                q1 = np.array([x.loc[i, 'Q_X'], x.loc[i, 'Q_Y'], x.loc[i, 'Q_Z'], x.loc[i, 'Q_W']])
                q2 = np.array([x.loc[i+FRAME_SKIP[nset], 'Q_X'], x.loc[i+FRAME_SKIP[nset], 'Q_Y'], x.loc[i+FRAME_SKIP[nset], 'Q_Z'], x.loc[i+FRAME_SKIP[nset], 'Q_W']])
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)
                r1 = r1.as_matrix()
                r2 = r2.as_matrix()
                
                
                
                p1 = np.array([x.loc[i, 'POS_X'], x.loc[i, 'POS_Y'], x.loc[i, 'POS_Z']])
                p2 = np.array([x.loc[i+FRAME_SKIP[nset], 'POS_X'], x.loc[i+FRAME_SKIP[nset], 'POS_Y'], x.loc[i+FRAME_SKIP[nset], 'POS_Z']])
                
                # rotation of camera frame
                r = R.from_euler('yzx',[1.57, 0, 1.57])
                r = r.as_matrix()
                
                r11 = np.matmul(r.transpose(), r1.transpose())
                r22 = np.matmul(r2, r)
                
                ##
                
                #trans = np.matmul(r1.transpose(),p2-p1)
                #rot_mat = np.matmul(r1.transpose(),r2)
                trans_b = trans
                trans = np.matmul(r11,p2-p1)
                rot_mat = np.matmul(r11,r22)
                rot = R.from_matrix(rot_mat)
                quat = rot.as_quat()
                
                if trans[0] == 0 and trans[1] == 0 and trans[2] == 0 and index > 0:
                    trans[0] = 0.0001
                    trans[1] = 0.0001
                    trans[2] = 0.0001
                    #trans[0] = trans_b[0]
                    #trans[1] = trans_b[1]
                    #trans[2] = trans_b[2]
                
                camera_l = get_path("images", i+FRAME_SKIP[nset], "png")
                depth = get_path("depth", i+FRAME_SKIP[nset], "png")
                
                # Write sample to file
                file.write("%i\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, depth, quat[3], quat[0], quat[1], quat[2], trans[0], trans[1], trans[2]))

