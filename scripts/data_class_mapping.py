import numpy as np
import os
import cv2

midair_class = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 6,
    6: 8,
}

tartanair_class = {
    75: 5,
    108: 4,
    112: 0,
    133: 5,
    145: 5,
    151: 4,
    152: 4,
    205: 3,
    218: 5, 
    219: 4,
    232: 4,
    234: 5,
    240: 4,
    241: 5,
    250: 4
}

ninja2_class = {
    0: 6,
    1: 7,
    2: 7,
    3: 1,
    4: 4,
    5: 3,
    6: 2,
    7: 7,
    8: 7,
    9: 8,
    10: 7,
    11: 7,
    12: 8,
    13: 8,
    14: 6,
    15: 3,
    16: 8,
    17: 8,
    18: 6,
    19: 3,
    20: 7,
    21: 2,
    22: 1,
    23: 1,
    24: 8
}

def map_udd5(a):
    #print(a)
    if (a == [35,142,107]).all():
        return 2
    elif (a == [156,102,102]).all():
        return 7
    elif (a == [128,64,128]).all():
        return 6
    elif (a == [142,0,0]).all():
        return 4
    elif (a == [0,0,0]).all():
        return 8
        
def map_icg(a):
    #print(a)
    if (a == [128,128,0]).all():
        return 0
    elif (a == [0,0,0]).all():
        return 3
    elif (a == [128,128,192]).all():
        return 8
    elif (a == [0,128,0]).all():
        return 8
    elif (a == [128,128,128]).all():
        return 4
    elif (a == [0,0,128]).all():
        return 8
    elif (a == [128,0,0]).all():
        return 8
    elif (a == [128,0,192]).all():
        return 8
    elif (a == [0,0,192]).all():
        return 8
    elif (a == [0,128,192]).all():
        return 7
    elif (a == [0,64,0]).all():
        return 2
    elif (a == [0,128,128]).all():
        return 6
        
def map_aeroscapes(a):
    #print(a)
    if (a == [128,128,0]).all():
        return 0
    elif (a == [0,0,0]).all():
        return 3
    elif (a == [128,128,192]).all():
        return 8
    elif (a == [0,128,0]).all():
        return 8
    elif (a == [128,128,128]).all():
        return 4
    elif (a == [0,0,128]).all():
        return 8
    elif (a == [128,0,0]).all():
        return 8
    elif (a == [128,0,192]).all():
        return 8
    elif (a == [0,0,192]).all():
        return 8
    elif (a == [0,128,192]).all():
        return 7
    elif (a == [0,64,0]).all():
        return 2
    elif (a == [0,128,128]).all():
        return 6
        
        
def map_ruralscapes(a):
    #print(a)
    if (a == [0,255,0]).all():
        return 3
    elif (a == [0,127,0]).all():
        return 2
    elif (a == [0,255,255]).all():
        return 7
    elif (a == [255,255,0]).all():
        return 0
    elif (a == [0,127,255]).all():
        return 8
    elif (a == [255,255,255]).all():
        return 6
    elif (a == [255,0,255]).all():
        return 7
    elif (a == [127,127,127]).all():
        return 4
    elif (a == [255,0,0]).all():
        return 1
    elif (a == [63,127,127]).all():
        return 3
    elif (a == [0,0,255]).all():
        return 8
    elif (a == [0,127,127]).all():
        return 8
    
    

def map_wuav(a):
    #print(a)
    if (a == [255,255,0]).all():
        return 0
    elif (a == [0,127,0]).all():
        return 2
    elif (a == [69,132,19]).all():
        return 2
    elif (a == [65,53,0]).all():
        return 2
    elif (a == [0,76,130]).all():
        return 3
    elif (a == [152,251,152]).all():
        return 3
    elif (a == [171,126,151]).all():
        return 5
    elif (a == [255,0,0]).all():
        return 1
    elif (a == [0,150,250]).all():
        return 7
    elif (a == [195,176,115]).all():
        return 8
    elif (a == [128,64,128]).all():
        return 6
    elif (a == [228,77,255]).all():
        return 6
    elif (a == [123,123,123]).all():
        return 4
    elif (a == [255,255,255]).all():
        return 4
    elif (a == [0,0,200]).all():
        return 8
    elif (a == [0,0,0]).all():
        return 8
    else:
        return 8
        
data = ["Vis"]
#data = ["Seq002","Seq003"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","Ninja2"])

for traj in data:
    print("Processing %s" % (traj))
    imgs = os.listdir(os.path.join(*[db_path, 'segmentation']))
    os.makedirs(os.path.join(*[db_path, "seg_id"]), exist_ok = True)
    
    for img in imgs:
        print(img)
        inp_path = os.path.join(*[db_path, 'segmentation', img])
        out_path = os.path.join(*[db_path, 'seg_id', img])
        im = cv2.imread(inp_path)
        im_upd = np.vectorize(ninja2_class.get)(im)
        cv2.imwrite(out_path,im_upd)
        
'''
data = ["Vis"]
#data = ["Seq002","Seq003"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","aeroscapes"])

for traj in data:
    print("Processing %s" % (traj))
    imgs = os.listdir(os.path.join(*[db_path, 'Visualizations']))
    os.makedirs(os.path.join(*[db_path, "seg_id"]), exist_ok = True)
    
    for img in imgs:
        print(img)
        inp_path = os.path.join(*[db_path, 'Visualizations', img])
        out_path = os.path.join(*[db_path, 'seg_id', img])
        im = cv2.imread(inp_path)
        im_upd = np.apply_along_axis(map_aeroscapes, -1,im)
        cv2.imwrite(out_path,im_upd)
'''    
'''
data = ["Seq000","Seq001","Seq002","Seq003"]
#data = ["Seq002","Seq003"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","wildUAV"])

for traj in data:
    print("Processing %s" % (traj))
    imgs = os.listdir(os.path.join(*[db_path, traj, 'segmentation']))
    for img in imgs:
        print(img)
        inp_path = os.path.join(*[db_path, traj, 'segmentation', img])
        out_path = os.path.join(*[db_path, traj, 'seg_id', img])
        im = cv2.imread(inp_path)
        im_upd = np.apply_along_axis(map_wuav, -1,im)
        cv2.imwrite(out_path,im_upd)
'''
'''
data = ["UDD5"]
#data = ["Seq002","Seq003"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","UDD"])

for traj in data:
    print("Processing %s" % (traj))
    imgs = os.listdir(os.path.join(*[db_path, traj, 'gt']))
    for img in imgs:
        print(img)
        inp_path = os.path.join(*[db_path, traj, 'gt', img])
        out_path = os.path.join(*[db_path, traj, 'seg_id', img])
        im = cv2.imread(inp_path)
        im_upd = np.apply_along_axis(map_udd5, -1,im)
        cv2.imwrite(out_path,im_upd)
'''
'''
data = ["DJI_0043","DJI_0044", "DJI_0045", "DJI_0046", "DJI_0047", "DJI_0050", "DJI_0051", "DJI_0053", "DJI_0056", "DJI_0061", "DJI_0085", "DJI_0086",
           "DJI_0088", "DJI_0089", "DJI_0093", "DJI_0097", "DJI_0101", "DJI_0114", "DJI_0116", "DJI_0118"]
#data = ["Seq002","Seq003"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","ruralscapes"])

for traj in data:
    print("Processing %s" % (traj))
    imgs = os.listdir(os.path.join(*[db_path, 'labels', traj]))
    os.makedirs(os.path.join(*[db_path, "seg_id", traj]), exist_ok = True)
    
    for img in imgs:
        print(img)
        inp_path = os.path.join(*[db_path, 'labels', traj, img])
        out_path = os.path.join(*[db_path, 'seg_id', traj, img])
        im = cv2.imread(inp_path)
        im_upd = np.apply_along_axis(map_ruralscapes, -1,im)
        cv2.imwrite(out_path,im_upd)
'''
'''
def map_sem(a):
    #print(a)
    if (a == [246,1,142]).all():
        return 6
    elif (a == [80,250,232]).all():
        return 4
    elif (a == [214,254,86]).all():
        return 2
    elif (a == [63,42,47]).all():
        return 7
    elif (a == [248,201,34]).all():
        return 8



class_index = {
        2: [(214, 254, 86)],
        4: [(80, 250, 232)],
        6: [(246, 1, 142)],
        7: [(63, 42, 47)],
        8: [(248, 201, 34)]}



'''

'''
## For Gascola
data = ["Easy","Hard"]

dir_path = os.path.dirname(os.path.realpath(__file__))
db_path = os.path.join(*[dir_path,"..", "datasets","TartanAir","gascola","gascola"])
s2 = set(list(tartanair_class.keys()))
for level in data:
    trajectories = os.listdir(os.path.join(db_path,level))
    for traj in trajectories:
        print("Processing %s %s" % (level, traj))
        #out_dir = os.path.join(*[db_path, level, traj, 'seg_upd'])
        #os.makedirs(out_dir, exist_ok=True)
        imgs = os.listdir(os.path.join(*[db_path, level, traj, 'seg_upd']))
        traj_len = len(imgs)
        for img in imgs:
            #print(img)
            inp_path = os.path.join(*[db_path, level, traj, 'seg_upd', img])
            out_path = os.path.join(*[db_path, level, traj, 'seg_upd', img])
            img_np = np.load(inp_path)
            values = np.unique(img_np)
            s1 = set(values.tolist())
            missing = s1 - s2
            
            if len(missing) > 0:
                print(img)
                print(missing)
            else:
                img_upd = np.vectorize(tartanair_class.get)(img_np)
                img_upd = img_upd.astype('uint8')
                with open(out_path, 'wb') as f:
                    np.save(out_path, img_upd)
           
'''
'''
data = ["Kite_training","PLE_training"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","midair"])

##For MidAir
for set in data:
    climates = os.listdir(os.path.join(db_path,set))
    for climate in climates:
        print("Processing %s %s" % (set, climate))
        out_dir = os.path.join(*[db_path, set, climate, 'segmentation'])
        os.makedirs(out_dir, exist_ok=True)
        trajectories = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation_v2']))
        for traj_num, (traj) in enumerate(trajectories):
            print(traj)
            out_dir = os.path.join(*[db_path, set, climate, 'segmentation', traj])
            os.makedirs(out_dir, exist_ok=True)
            imgs = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation_v2', traj]))
            traj_len = len(imgs)
            for img in imgs:
                inp_path = os.path.join(*[db_path, set, climate, 'segmentation_v2', traj, img])
                out_path = os.path.join(*[db_path, set, climate, 'segmentation', traj, img])
                #print("Img: ", inp_path)
                img_cv = cv2.imread(inp_path)
                img_upd = np.vectorize(midair_class.get)(img_cv)
                cv2.imwrite(out_path, img_upd)
'''       
                
                
                


