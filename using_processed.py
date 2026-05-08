import cv2
import numpy as np
from approxes import Marker2D
from utils import remove_background, check_bottom_half, check_straight, voxel_position_to_index
from tqdm import tqdm


import yaml
from yaml.loader import SafeLoader

input = "chat_data/calibration.yaml"

# Open the file and load the file
with open(input) as f:
    data = yaml.load(f, Loader=SafeLoader)

print(f"camera_matrix ======= {data['camera_matrix']}")
print(f"dist_coefs ======= {data['dist_coefs']}")
mtx = np.array([data['camera_matrix']]).reshape(3,3)
dist = np.array([data['dist_coefs']]).reshape(5,1)


obj_name = "obj03"

obj_dicts = {
    "obj01": {"size":(100,100,120), "fill_holes":1},
    "obj02": {"size":(150,150,70), "fill_holes":0},
    "obj03": {"size":(170,170,90), "fill_holes":1},
    "obj04": {"size":(90,90,90),"fill_holes":0}
}





cap = cv2.VideoCapture(f"data/{obj_name}.mp4") 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

center = False


d1,d2,d3 = obj_dicts[obj_name]["size"]
initials = [-d1/2,-d2/2,80]
V = np.ones((d1,d2,d3,1))

x = np.linspace(initials[0],initials[0] + d1 - 1,d1)
y = np.linspace(initials[1],initials[1] + d2 - 1,d2)
z = np.linspace(initials[2],initials[2] + d3 - 1,d3)


xx,yy,zz = np.meshgrid(x,y,z)
positions = np.vstack([xx.ravel(), yy.ravel(),zz.ravel()])


ellipse_ranges = [1550,1531,1516,1500,1485,1470]


angle_scanned = [-1]

pose_name = obj_name.replace("0","")
poses = np.load(f"processed/{pose_name}/poses.npz")
rmats = poses['Rmats']
tvecs = poses['Tvecs']
length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))

assert length == len(rmats), "not enough projection data"

frame_number = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    rmat = rmats[frame_number]
    tvec = tvecs[frame_number]

    frame_number += 1

    rvec = cv2.Rodrigues(rmat)[0]


    h,  w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    if not ret:
        break

    frame = dst

    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    THRESHOLD_VALUE = 160 # This value was better
    ret, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    contours, hierarchies = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0


    ellipses = []
    binary_angle = [1,1,1,1,1]
    angle = -1
    markers =[]

    width = frame.shape[1]
    height = frame.shape[0]

    for contour in contours:
  
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
    
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        area = cv2.contourArea(contour)
        

        #Get fiducal markers
        if len(approx) == 5:
            
            # filter small pentagons to be sure it's fiducal marker
            if area >= 1000 :
                
                # filter fiducal marker exist in center of image
                if check_straight(approx,width):
                    marker = Marker2D(approx,width,[width/2.,1293])
                    # print("markers: \n",marker)
                    markers.append(marker.get_points())
                    # print("marker approx:",approx, "\n", approx.shape)
                    center = True
                    cv2.drawContours(frame, [approx], -1, (255,0,12), 3)
        #Get ellipses in fiducal to calculate angle
        elif len(approx) > 5 and area > 200  and area < 400 :
            
            # filter ellipses and get ones who is in horizentally center and bottom of image
            if check_straight(approx, width) and center and check_bottom_half(approx, height, 0.2):
                # print("am i here?")
                ellipse = cv2.fitEllipse(contour)
                
                ellipses.append(ellipse)
                cv2.ellipse(frame, ellipse, (0,0,12), 2)

                # calculate angle, if ellipse exist -> 0 else remains -> 1
                for i in range(5):
                    if int(ellipse[0][1]) in range(int(ellipse_ranges[i+1]),int(ellipse_ranges[i])):
                        # print("Im here")
                        binary_angle[i] = 0


    if binary_angle != [1,1,1,1,1]:
        angle = binary_angle[0] * 16 + binary_angle[1] * 8 + binary_angle[2]*4 + binary_angle[3]*2 + binary_angle[4] * 1
    
    # show "unknown" before detecting any marker, else show angle
    if angle == -1 or angle == 32:
        angle_text = "unknown"
    else:
        angle_text = str(angle*15)
    
    # print("angles", angle, angle_scanned)
    
    if angle in angle_scanned or angle > 23:
            continue
    
    if len(angle_scanned) > 2:
        if abs(angle_scanned[-1] - angle)  > 3 and angle != 23:
            continue

    if len(ellipses) != 0 and angle != -1 and len(markers) == 1: 
        print("scanning angle:", angle," angles scanned so far:", angle_scanned)

        
        imp, j = cv2.projectPoints(positions, rvec, tvec, newcameramtx,np.zeros((5,1)))
        # changing x and y in projection (because of ratate 90)
        imp[:,0, 0], imp[:, 0,1] = imp[:,0, 1], imp[:,0, 0].copy()
        imp = imp.astype('int')

        mask = remove_background(frame, obj_dicts[obj_name]["fill_holes"])
        cv2.drawContours(frame, imp, -1, (255,0,12), 3)
    
        for i,projected in enumerate(tqdm(imp)):
            # clip x and y for avoiding index error if cube is large
            y = projected[0][1]
            x = projected[0][0]
            y = y if y<height else height -1
            y = y if y>0 else 0
            x = x if x < width else width - 1
            x = x if x> 0 else 0
            # carving space(voxels)
            if mask[y,x] < 160:
                x = positions[0,i]
                y = positions[1,i]
                z = positions[2,i]
                index = voxel_position_to_index([x,y,z],initials)
                V[index] = 0
        

        print("number of points exist in voxels:",np.sum(V))

        
        angle_scanned.append(angle)
    
    # display angle
    cv2.putText(frame, angle_text, (int(width/2),100),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    

    center =False


    # save voxels in npy format
    np.save("voxels.npy",V)

    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # else:
    #     cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()


##### Algorithm
"""
For the space-carving technique being effective, two sub-problems must be solved at each
frame i:
1. Estimate the position of the camera with respect to the object (ie. the camera pose
Ri, Ti)
2. Find the “silhouette” of the object by clustering the pixels as part of the background or
foreground.
The complete algorithm works as follows:
• Let V be an initial set of voxels distributed in a N x N x N cube
• For each image:
• Compute the camera Projection matrix P = K [Ri Ti]
• Project all the voxels onto the image
• Check if the projection of the voxel is inside or outside the silhouette. In the latter
case, remove the voxel from V
"""


# Marker.svg
"""
marker.svg the marker used in the turntable. You can open this file with Inkscape
to inspect its dimensions
"""

# obtain camera pose from the markers
"""
To recover the camera pose from the marker, you can either factorize the
Homography matrix mapping points from the marker reference frame to the camera
or use the OpenCV function solvePnP
(https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a5
8bc931c033d). This latter method tends to be more stable in case of noise,
especially if using the method SOLVEPNP_IPPE.

Once the camera pose has been recovered, try to project back the marker points
onto the image plane to check the error between the projected position and the
expected marker point location. You should aim for RMS error below 1px for good
results when carving the voxel grid

"""

