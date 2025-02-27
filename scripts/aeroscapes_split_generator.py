import os
import argparse
#from pyquaternion import Quaternion
import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","Aeroscapes"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "aeroscapes"]), help="path to folder to store csv files")
a = parser.parse_args()

train_file = "/media/DATA_4TB/Yara/aeroscapes/ImageSets/trn.txt"
test_file = "/media/DATA_4TB/Yara/aeroscapes/ImageSets/val.txt" 
if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)
    file1 = open(test_file)
    imgs = file1.readlines()
    file1.close()
    
    out_dir = a.output_dir
    file_name = os.path.join(out_dir, "test.csv")
    trans = [0,0,0]
    quat = [0,0,0,1]
    # Create csv file
    index = 0
    with open(file_name, 'w') as file:
        file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "semantic", "depth", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
        
        for img in imgs:
            # Compute frame-to-frame camera motion
                
            camera_l = os.path.join("JPEGImages", img[:-1]+".jpg")
            semantic = os.path.join("SegmentationClass", img[:-1]+".png")
            depth = os.path.join("DepthTest", "depth_"+str(index)+".npy")
                
            # Write sample to file
            file.write("%i\t%s\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, semantic, depth, quat[3], quat[0], quat[1], quat[2], trans[0], trans[1], trans[2]))
            index= index + 1
