#With reference to Mathetmatica example provided by Professor M.Lavery

#import relevent modules
import itertools
import math
import os, time
import cv2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

plt.ion() #prevents plt.show() from blocking rest of code

#Mask parameters
H_Res = 512
V_Res = 512
length = 1024
pixelSize = 5.4e-6
x = np.linspace(0, length, H_Res)
y = np.linspace(0, length, V_Res)

wavelength = 0.633e-6
dist = 300e-3 #in mm
aper = 5e-3
pixelSize = 5.4e-6

#Mask parameters
xz = 10*np.pi #to be controlled by slider
yz = 10*np.pi #to be controlled by slider

x_gratGrid = 2
y_gratGrid = 2

# Works!
def multipleGratings(x_gratGrid, y_gratGrid, xz_list, yz_list):
    multiGrat = [] #array to append all gratings to
    
    #first make the gratings across y and then combine
    k = 0 #for going through the x*y values in the two lists
    m = 0
    for i in range(0, x_gratGrid): # x-grating size
        gratList = []
        for j in range(0, y_gratGrid): #y-grating size
            temp = update_grating(xz_list[k], yz_list[k])
            gratList.append(temp)
            k = k+1
        # print(gratList[0].shape)
        # print(gratList[1].shape)
        
        # Combine first two gratings
        combGrat = []    
        temp1 = gratList[0]
        temp2 = gratList[1]
        combGrat = np.vstack((temp1, temp2))
        
        #check if more than two rows of gratings are required
        if y_gratGrid > 2:
            for o in range(3, y_gratGrid):
                temp = gratList[o]
                combGrat = np.vstack((combGrat, temp))
        
        #Combine each vertical column of gratings
        if i == 0:
            multiGrat = combGrat
        else:
            multiGrat = np.hstack((multiGrat, combGrat))    
                   
    return multiGrat

#create meshgrid/table
def meshgridFunc(X, Y):
    x_mesh, y_mesh = np.meshgrid(X,Y)
    return x_mesh, y_mesh

# https://opg.optica.org/ao/fulltext.cfm?uri=ao-56-16-4779&id=367129
def shifttip():
    m = 0
    angle = (math.asin(((0.5-m)*wavelength)/pixelSize))/2
    # phi = (((res*2*np.pi)/(1920*1080)))
    # angle = math.atan(phi/aper) # in radians
    return angle

def shifttilt(): 
    m = 0
    angle = (math.asin(((0.5-m)*wavelength)/pixelSize))/2
    # phi = (((res*2*np.pi)/(1920*1080)))
    # angle = math.atan(phi/aper) # in radians
    return angle

def update_grating(xz, yz):
    #calculate tilt and tip
    phasetilt = yz #shifttilt()
    phasetip = xz #shifttip() 
    
    grad = np.empty((0, V_Res-1), int)
    for j in range(1, V_Res):
        x_grad = []
        for i in range(1, H_Res):
            temp = i*(phasetip)/H_Res+j*(phasetilt)/V_Res
            x_grad.append(temp)
        
        grad = np.append(grad, np.array([x_grad]), axis = 0)
       
    gratmod = np.empty((0, grad.shape[1]), int)
    
    for j in range(0,grad.shape[1]):
        x_axis = []
        for i in range(0,grad.shape[0]):
            x_axis.append(np.remainder(grad[i][j]*15, 2*np.pi)) 
        gratmod = np.append(gratmod, np.array([x_axis]), axis = 0)
    
    gratbin = np.empty((0, grad.shape[1]), int)
    max_val = np.amax(gratmod)
    comp_val = max_val/2
    
    for l in range(0, grad.shape[1]):
        x_ax = []
        for k in range(0, grad.shape[0]):
            if gratmod[k][l] > comp_val:
                x_ax.append(1)
            elif gratmod[k][l]<= comp_val:
                x_ax.append(0)
        gratbin = np.append(gratbin, np.array([x_ax]), axis = 0)
        
    return gratbin

#single grating
gratbin = update_grating(xz, yz)

#create some random xz_list and yz_list values for testing
xz_list = [-3, 10, 15, 20]
yz_list = [20, 5, 20, 25]
#gratbin = multipleGratings(x_gratGrid, y_gratGrid, xz_list, yz_list)

#show plot
plt.figure(1)
fig, ax = plt.subplots()

axy = plt.axes([0.25, 0.1, 0.65, 0.03])
x_slider = Slider(
    ax=axy,
    label="x",
    valmin=-30*math.pi, #-30*math.pi,
    valmax=30*math.pi,
    valinit=xz,
)

axX = plt.axes([0.1, 0.25, 0.0225, 0.63])
y_slider = Slider(
    ax=axX,
    label="y",
    valmin= -30*math.pi, #-30*math.pi, #30*math.pi,
    valmax=30*math.pi, #30*math.pi,
    valinit=yz,
    orientation="vertical",
)

# The function to be called anytime a slider's value changes
def update(var):
    xz = x_slider.val
    yz = y_slider.val
    gratbin = update_grating(xz, yz)
    im = ax.imshow(gratbin, cmap='Greys', aspect='auto')
    #fig.canvas.draw_idle()
    im2 = ax2.imshow(gratbin, cmap='Greys', aspect='auto')
    #fig2.canvas.draw_idle()
    
    
# register the update function with each slider
y_slider.on_changed(update)
x_slider.on_changed(update) 

Baxe = plt.axes([0.25, 0.02, 0.1, 0.03])
Balign = Button(Baxe, "align")

def align(var):
    sweep_align()
    
Balign.on_clicked(align)

BintensityAxe = plt.axes([0.4, 0.02, 0.1, 0.03])
Bintensity = Button(BintensityAxe, "intensity align")

def intensityAlign(var):
    sweep_align_with_intensity()

Bintensity.on_clicked(intensityAlign)
    
Bpreciseaxes = plt.axes([0.55, 0.02, 0.1, 0.03])
Bprecise = Button(Bpreciseaxes, "precise align")

def precise(var):
    precise_align()

Bprecise.on_clicked(precise)

BmultiAxes = plt.axes([0.7, 0.02, 0.1, 0.03])
Bmulti = Button(BmultiAxes, "multiple")

#def multi(var):

#Bmulti.on_clicked(multi)

im = ax.imshow(gratbin, cmap='Greys', aspect='auto')

plt.figure(20)
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(gratbin, cmap='Greys', aspect='auto')
plt.show()

############################################################################################################################################################################################
################################################################################ Alignment code starts here ################################################################################

def captureImage():
     # Take an image from the video feed
    check, frame = video.read()
    params = cv2.SimpleBlobDetector_Params()

    # params.minThreshold = 0;
    # params.maxThreshold = 200;
    params.filterByArea = True
    params.minArea = 100
    params.filterByInertia = False
    params.filterByConvexity = False
    # convert colour image to grayscale
    grey = cv2.bitwise_not(frame)
    #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #new_image = np.zeros(grey.shape, grey.dtype)
    
    # change image exposure
    # alpha = 0.5 # Simple contrast control
    # beta = 0    # Simple brightness control
    # for y in range(grey.shape[0]):
    #     for x in range(grey.shape[1]):
    #             new_image[y,x] = np.clip(alpha*grey[y,x] + beta, 0, 255)
    
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(grey)
    pts = cv2.KeyPoint_convert(keypoints)
    print(pts)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_detected = cv2.drawKeypoints(grey, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show keypoints
    cv2.imshow("Keypoints", im_detected)
    return pts

def captureImage_with_intensity():
     # Take an image from the video feed
    intensity = 0
    check, frame = video.read()
    params = cv2.SimpleBlobDetector_Params()

    # params.minThreshold = 0;
    # params.maxThreshold = 200;
    params.filterByArea = True
    params.minArea = 100
    params.filterByInertia = False
    params.filterByConvexity = False
    # convert colour image to grayscale
    grey = cv2.bitwise_not(frame)
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(grey)
    pts = cv2.KeyPoint_convert(keypoints)
    print(pts)
    if len(pts)>0: 
        radius = 50 #cv2.KeyPoint(keypoints).size /2 # Obtain radius of blob (keypoint.size returns diameter)
        # Section out the detected blob section to obtain average intensity of blob
        grey_section = grey[pts[0][0]-radius:pts[0][0]+radius, pts[0][1]-radius:pts[0][1]+radius]
        intensity = cv2.mean(grey_section)
    
    # Draw detected blobs as red circles.
    im_detected = cv2.drawKeypoints(grey, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show keypoints
    cv2.imshow("Keypoints", im_detected)
    return pts, intensity

target_coordinates = [[0,0], [1,1]]

########## Simple sweep-based rough alignment #############
def sweep_align():
    start_time = time.time()
    pts = []
    broken = False
    for i in range(-4,4): # Values to be adjusted
        for j in range(-4,4): # values to be adjusted
            xz = j*np.pi*7
            yz = i*np.pi*7
            print(j,i)
            y_slider.eventson = False
            y_slider.set_val(yz)
            y_slider.eventson = True
            x_slider.eventson = False
            x_slider.set_val(xz)
            x_slider.eventson = True
            update(0)
            #Take capture from webcam, check if any spots are visible, if not continue to iterate through. 
            pts = captureImage()
            if np.any(pts):
                print("aligned with:")
                print(j, i)
                print("time taken:")
                print(time.time() - start_time)
                broken = True
                break
            plt.pause(0.2)
        if broken:
            break
                    
def sweep_align_with_intensity():
    start_time = time.time()
    highest_intensity = [0,0,0]
    num_of_spots = 0
    for i in range(-6, 6): # Values to be adjusted
        for j in range(-6,6): # values to be adjusted
            xz = j*np.pi*5
            yz = i*np.pi*5
            print(xz, yz)
            y_slider.eventson = False
            y_slider.set_val(yz)
            y_slider.eventson = True
            x_slider.eventson = False
            x_slider.set_val(xz)
            x_slider.eventson = True
            update(0)
            
            pts, intensity = captureImage_with_intensity() #obtain centre-point of blob+intensity
            if len(pts)>0:
                num_of_spots = num_of_spots + 1
                if highest_intensity[2] < intensity: #compare and save highest intensity spot
                    highest_intensity = [xz, yz, intensity]
            plt.pause(0.2)
                           
    update_grating(highest_intensity[0], highest_intensity[1])
    print("brightest spot found at:")
    print(highest_intensity)
    print("number of spots found:")
    print(num_of_spots)
    print("runtime:")
    print(time.time()-start_time)
    
########## Image position-based alignment #############
# Image-based alignment if the spot is already pointing on the CCD sensor.
def simple_alignment(targ_coord, pt, cur_xz, cur_yz):
    x = (targ_coord[0] - pt[0][0]) * (20 / 512)
    y = (targ_coord[1] - pt[0][1]) * (8 / 512)
    xz = cur_xz + x
    yz = cur_yz + y
    return xz,yz
    
def precise_align():
    target = [420, 200]
    pts = captureImage
    current_yz = y_slider.val
    current_xz = x_slider.val
    x, y = simple_alignment(target, pts, current_xz, current_yz)
    xz = current_xz + x * np.pi
    yz = current_yz + y * np.pi
    y_slider.eventson = False
    y_slider.set_val(yz)
    y_slider.eventson = True
    x_slider.eventson = False
    x_slider.set_val(xz)
    x_slider.eventson = True
    update(0)
    print("Spot aligned to target:")
    print(target)
    
        
# Allows multiple spots to be aligned
def simple_align_multi(targ_coords, pts, num_of_spots): #expects a 2D-Array for both target coordinates and pts 
    new_align = [] 
    # assign each blob found with nearest target_coordinate
    for i in range(0, num_of_spots):
        # find index for nearest targ_coordinate and detected point
        idx = targ_coords[np.abs(targ_coords-pts[i]).nanargmin()]
        # calculate grating adjustment
        new_align.append([simple_alignment(targ_coords[idx], pts[i])])
    return new_align #Function returns a 2D array consisting of [[xz1, yz1], [xz2, yz2]] for alignment of each spot. 
  
# Capturing video
video = cv2.VideoCapture(1) #0 for builtin webcam, 1 for external 
  
# Infinte loop for video feed
while True:
    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        break
  
video.release()
  
# close all windows
cv2.destroyAllWindows()