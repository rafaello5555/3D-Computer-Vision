
import cv2
import numpy as np
from approxes import Marker2D
from utils import rotate, check_bottom_half, check_straight,  sort_markers_by_side
import os

import yaml
from yaml.loader import SafeLoader


class FrameAngle:
    def __init__(self,frame,angle,markers):
        self.frame = frame
        self.angle = angle
        self.markers = markers


def get_best_frame_angle(candidates):
    candidates = [c for c in candidates if len(c.markers) == 9]
    angles = [f.angle for f in candidates]
    # print(angles)
    best_angle = max(set(angles), key=angles.count)
    choices = [f for f in candidates if f.angle == best_angle]
    best = choices[int(len(choices)/2)]
    return best




input = "chat_data/calibration.yaml"

# Open the file and load the file
with open(input) as f:
    data = yaml.load(f, Loader=SafeLoader)

print(f"camera_matrix ======= {data['camera_matrix']}")
print(f"dist_coefs ======= {data['dist_coefs']}")
mtx = np.array([data['camera_matrix']]).reshape(3,3)
dist = np.array([data['dist_coefs']]).reshape(5,1)



obj_name = "obj04"

if not os.path.isdir(f"res_{obj_name}"):
    os.mkdir(f"res_{obj_name}")

cap = cv2.VideoCapture(f"data/{obj_name}.mp4") 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

center = False

ellipse_ranges = [1550,1531,1516,1500,1485,1470]


frame_angle_candidates = []
final_frame_angle = []

angle_scanned = [-1]


newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (frame_width,frame_height), 1, (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    new_dist = np.zeros((5,1))




    dst = cv2.rotate(dst, cv2.cv2.ROTATE_90_CLOCKWISE)  

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    THRESHOLD_VALUE = 160 # This value was better
    ret, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    contours, hierarchies = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    ellipses = []
    binary_angle = [1,1,1,1,1]
    angle = -1
    markers =[]
    dots = []
    
    
    width = dst.shape[1]
    height = dst.shape[0]

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
            if area >= 300  and check_bottom_half(approx, height, 0.195):
                
                # filter fiducal markers exist in bottom and center of image
                # if check_straight(approx,width):
                    # print("approx", approx)
                marker = Marker2D(approx,width,[width/2.,1293])
                # print(marker)
                cv2.drawContours(dst, [approx], -1, (0,0,255), 2)
    
                markers.append(marker)
                center = True

        # compute angle based on centered fiducal marker (this calculation maybe result wrong angles but it will filter using FrameAngle class)
        elif len(approx) > 5 and area > 200  and area < 400 :
            
            # filter ellipses and get ones who is in horizentally center and bottom of image
            if check_straight(approx, width) and center and check_bottom_half(approx, height, 0.195):
                # print("am i here?")
                ellipse = cv2.fitEllipse(contour)
                
                ellipses.append(ellipse)
                cv2.ellipse(dst, ellipse, (0,0,12), 2)

                # calculate angle, if ellipse exist -> 0 else remains -> 1
                for i in range(5):
                    if int(ellipse[0][1]) in range(int(ellipse_ranges[i+1]),int(ellipse_ranges[i])):
                        # print("Im here")
                        binary_angle[i] = 0





    #Calculate angle based on binary_angle 
    if binary_angle != [1,1,1,1,1]:
        angle = binary_angle[0] * 16 + binary_angle[1] * 8 + binary_angle[2]*4 + binary_angle[3]*2 + binary_angle[4] * 1
    
    

    # show "unknown" before detecting any marker, else show angle
    if angle == -1 or angle == 32:
        angle_text = "unknown"
    else:
        angle_text = str(angle*15)
    
    # filter false angles
    if angle > 23:
        continue

    
    # avoid checking checked angles
    angle_appended = [f.angle for f in final_frame_angle]
    if angle in angle_appended:
        continue


    #filter false angles, by using mode angle in frequent angles
    if angle != -1:
        frame_angle_candidates.append(FrameAngle(dst,angle,markers))
    if angle == -1 and len(frame_angle_candidates) != 0:
        best_frame_angle = get_best_frame_angle(frame_angle_candidates)
        final_frame_angle.append(best_frame_angle)
        frame_angle_candidates = []

    # if all frames processed, do not process again
    if len(final_frame_angle) == 24:
        break

    # print("best len", len(final_frame_angle))

    cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
    cv2.imshow('dst', dst)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# result.release()
cv2.destroyAllWindows()

# process extracted frames


rmats = []
tvecs = []

for frame_number, frame_angle in enumerate(final_frame_angle):


    point2d = np.array([])
    points_3d = np.array([
                    # (70, 0, 0), # point A
                    (65, 5, 0), # point B
                    (98, 5, 0), # point C
                    (98, -5, 0), # point D
                    (65, -5, 0), # point E
                    ], dtype=np.float32)

    frame = frame_angle.frame
    markers = frame_angle.markers
    angle = frame_angle.angle

    lmarkers = sort_markers_by_side(markers,"left")
    rmarkers = sort_markers_by_side(markers,"right")
    cmarker = [marker for marker in markers if marker.side=="center"][0]


    # calculate 2d camera points based on markers
    for i,marker in enumerate(lmarkers):   
        point2d  = np.append(
            point2d,
            marker.get_points().reshape(-1,2),
        ).reshape(-1,2,1).astype('float32')

        
    point2d  = np.append(
            point2d,
            cmarker.get_points().reshape(-1,2),
        ).reshape(-1,2,1).astype('float32')

        
    for i,marker in enumerate(rmarkers):
        point2d  = np.append(
            point2d,
            marker.get_points().reshape(-1,2),
        ).reshape(-1,2,1).astype('float32')

            
    # show markers in image to validate calculation
    for i,marker in enumerate(point2d):

        cv2.putText(frame, f"BE {str(i)}", (int(marker[0]) , int(marker[1])),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)  
        cv2.circle(frame,(int(marker[0]) , int(marker[1])),5,(255,255,255),-1)



    # calculate 3d points. we use 9 markers
    left4 = rotate(points_3d,-60)
    left3 = rotate(points_3d,-45)
    left2 = rotate(points_3d,-30)
    left1 = rotate(points_3d,-15)
    right1 = rotate(points_3d,15)
    right2 = rotate(points_3d,30)
    right3 = rotate(points_3d,45)
    right4 = rotate(points_3d,60)

    new_points_3d = np.array([])
    final_points_3d = np.array([])

    new_points_3d = np.vstack((left1,left2,left3,left4,points_3d,right1, right2,right3, right4 ))


    # calculate 3d points current coordinates based on angles
    for point in new_points_3d:
        new_point = rotate(point,-angle*15)
        final_points_3d = np.append(final_points_3d,new_point)
    
    
    final_points_3d = final_points_3d.reshape((-1,3))

    # print to check shape of 3d points and 2d points before solving projection
    print("shapes:", final_points_3d.shape, point2d.shape)

    ret, rvecs, tvec = cv2.solvePnP(final_points_3d, point2d, newcameramtx, np.zeros((5,1)), flags=cv2.SOLVEPNP_IPPE )
    # refine projection 
    rvecs, tvec = 	cv2.solvePnPRefineVVS(	final_points_3d, point2d, newcameramtx, np.zeros((5,1)), rvecs, tvec	) 

    # test 3 arbitary 3d points. result will show as blue dots in images
    point_for_test = np.array([
            (0,0,0),
            (0,0,80),
            (70, 0, 0),  # point A
    ], dtype=np.float32)
    
    # project 3 3d points 
    imp, j1 = cv2.projectPoints(point_for_test, rvecs, tvec, newcameramtx,np.zeros((5,1)))
    imp = imp.astype('int')
    
    # draw 3 projected points in image
    cv2.drawContours(frame, imp, -1, (255,0,12), 10)

    #save visual results 
    cv2.imwrite(f"res_{obj_name}/frame{frame_number}.png",frame)

    # convert rvec to rmat
    rmat = cv2.Rodrigues(rvecs)[0]

    rmats.append(rmat)
    tvecs.append(tvec)

np.savez(f"{obj_name}.npz",Rmats=rmats, Tvecs=tvecs)

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

