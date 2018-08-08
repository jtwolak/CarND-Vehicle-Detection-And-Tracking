import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
import glob
import time
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split

#===============================
# CONFIGURATION
#===============================
TRAIN_MODEL = "NO"
TRAIN_MODEL_SMALLSET = "NO"
TRAIN_MODEL_LARGESET = "YES"
SAVE_MODEL = "YES"
SHOW_RESULTS = "YES"
SHOW_CAR_NOT_CAR = "NO"
SHOW_HOG_FEATURES = "NO"
SHOW_HOG_FEATURES_HLS = "NO"
SHOW_ALL_WINDOWS = "NO" #Warning "YES" this breaks the find_cars() function
SHOW_VIDEO_PROCESSING = "NO"
PROCESS_TEST_IMAGES = "NO"
PROCESS_TEST_VIDEO = "NO"
PROCESS_PROJECT_VIDEO = "YES"
PROCESS_VIDEO_BYPASS = "NO"
BBoX_HISTORY = 5

#===============================
# FUNCTIONS
#===============================

#----------------------------------------------------------------------------------------------
# Function that converts image from RGB to another format
#---------------------------------------------------------------------------------------------
def convert_color(image_in, color_space='YCrCb'):
    if color_space != 'RGB':
        if color_space == 'HSV':    image_out = cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':  image_out = cv2.cvtColor(image_in, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':  image_out = cv2.cvtColor(image_in, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':  image_out = cv2.cvtColor(image_in, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':image_out = cv2.cvtColor(image_in, cv2.COLOR_RGB2YCrCb)
    else:
        image_out = np.copy(image_in)

    return image_out

#----------------------------------------------------------------------------------------------
# Function that computes and returns HOG features and visualization
#---------------------------------------------------------------------------------------------
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

#--------------------------------------------------------------------------
# Function that compute binned color features
#--------------------------------------------------------------------------
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()

    #color1 = cv2.resize(img[:, :, 0], size).ravel()
    #color2 = cv2.resize(img[:, :, 1], size).ravel()
    #color3 = cv2.resize(img[:, :, 2], size).ravel()
    #features1 = np.hstack((color1, color2, color3))

    # Return the feature vector
    return features

#---------------------------------------------------------------------------
# Function that computes color histogram features
# --------------------------------------------------------------------------
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#--------------------------------------------------------------------
# Function that extract features from a single image window
#--------------------------------------------------------------------
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                        orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                        orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)

#---------------------------------------------------------------
# Function that extracts features from a list of images
#--------------------------------------------------------------
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        img_features = []
        # 1. Read in each one by one
        image = ndimage.imread(file, mode='RGB')
        # 2. Extract features for the current image
        img_features = single_img_features(image, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                           cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        # 3. Append the features to the list of features
        features.append( img_features )

    # Return list of feature vectors
    return features

#-----------------------------------------------------------------------------
# Function that takes an image, start and stop positions in both x and y,
# window size (x and y dimensions), and overlap fraction (for both x and y)
# and returns a list of windows
#----------------------------------------------------------------------------
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None: x_start_stop[0] = 0
    if x_start_stop[1] == None: x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None: y_start_stop[0] = 0
    if y_start_stop[1] == None: y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

#---------------------------------------------------------------
# Function that draws bounding boxes
#---------------------------------------------------------------
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_boxes_nocopy(img, bboxes, color=(0, 0, 255), thick=6):
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return

#---------------------------------------------------------------------------
# Function that pass an image and the list of windows to be searched
# (output of slide_windows()) and returns windows with positive detections
#---------------------------------------------------------------------------
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows

#-------------------------------------------------------------------------------------------------------------
# Function that can extract features using hog sub-sampling and makes predictions
#-------------------------------------------------------------------------------------------------------------
def find_cars(img, bbox_list, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel='ALL' ):
    draw_img = np.copy(img)
    #img = img.astype(np.float32) / 255

    # reduce the input image size only to the region we are interested in searching for cars
    img_tosearch = img[ystart:ystop, :, :]

    # apply color conversion the RGB input image
    ctrans_tosearch = convert_color(img_tosearch, color_space)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_feat == True:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    hog_features = []
    spatial_features = []
    hist_features = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            if hog_feat == True:
                if hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                elif hog_channel == 0:
                    hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                elif hog_channel == 1:
                    hog_features = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                elif hog_channel == 2:
                    hog_features = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                else:
                    hog_features = []

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get spacial features
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)

            # Get color histogram features
            if hist_feat == True:
                hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            # Predict car or not car?
            test_prediction = svc.predict(test_features)
            if SHOW_ALL_WINDOWS == "YES": test_prediction = 1
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox_list.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img

#--------------------------------------------------------------------
# Function used to generate heatmap
#---------------------------------------------------------------------
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

#----------------------------------------------------------------------
# Function that applies threshold to heatmap to remove false positives
#----------------------------------------------------------------------
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

#----------------------------------------------------------------------
# Function that draws bounding boxes
#----------------------------------------------------------------------
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img

#----------------------------------------------------------------------
# Function that defines image processing pipeline
#----------------------------------------------------------------------
def image_processing_pipeline( image, bbs=None ):
    # Detect cars and generate bounding boxes around them
    box_list = []
    ystart = 400 #np.int32(h/2)
    ystop = 528 # np.int32((3*h)/4)
    scale = 1.0
    out_img1 = find_cars(image, box_list, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, hog_channel)
    ystart = 408 # np.int32(h/2)
    ystop = 600 # np.int32(h)
    scale = 1.5
    out_img2 = find_cars(image, box_list, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, hog_channel)
    ystart = 400 # np.int32(h/2)
    ystop = 756 # np.int32(h)
    scale = 2.0
    out_img3 = find_cars(image, box_list, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, hog_channel)

    if SHOW_ALL_WINDOWS == "YES":
        populate_plot(3, 1, 1, out_img1, "64x64", color="rgb",axis='on')
        populate_plot(3, 1, 2, out_img2, "96x96", color="rgb",axis='on')
        populate_plot(3, 1, 3, out_img3, "128x128", color="rgb",axis='on')
        plt.show()

    if SHOW_VIDEO_PROCESSING == "YES":
        tmp_image = np.copy(image)

    # Preapre heatmap image
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    if bbs:
        # Keep boundig boxes from the last few frames
        bbs.img_cnt += 1
        bbs.bbox_lists.append(box_list)  # add the box list for the current image
        bbs.n_curr += 1

        if bbs.n_curr > bbs.n_max:
            bbs.bbox_lists.pop(0)  # remove the last element from the list
            bbs.n_curr -= 1

        for list in bbs.bbox_lists:
           heat = add_heat(heat, list)
           if SHOW_VIDEO_PROCESSING == "YES":
               draw_boxes_nocopy(tmp_image, list)

        if SHOW_VIDEO_PROCESSING == "YES":
            bbs.last_imgs.append(tmp_image)
            bbs.last_heatmaps.append(heat)
            bbs.m_curr += 1
            if bbs.m_curr > bbs.n_max:
                bbs.last_imgs.pop(0)
                bbs.last_heatmaps.pop(0)
                bbs.m_curr -= 1
            idx = 1
            id = bbs.img_cnt
            for img in bbs.last_imgs:
                title = "img_" + str(id) + "_bboxes"
                populate_plot(bbs.n_max, 2, idx, img, title, color="rgb", axis='off')
                idx += 2
                id -= 1
            idx = 2
            id = bbs.img_cnt
            for img in bbs.last_heatmaps:
                title = "img_" + str(id) + "_heatmap"
                populate_plot(bbs.n_max, 2, idx, img, title, color="rgb", axis='off')
                idx += 2
                id -= 1
            plt.show()
    else:
        # Add heat to each box in box list
        heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 3)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    print(labels[1], 'cars found')
    if SHOW_VIDEO_PROCESSING == "YES":
        if bbs: id = bbs.img_cnt
        else: id = 0
        title = "img_"+str(id)+"_labels_image"
        plt.title(title)
        plt.imshow(labels[0], cmap='gray')
        plt.show()

    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    if SHOW_VIDEO_PROCESSING == "YES":
        if bbs: id = bbs.img_cnt
        else: id = 0
        title = "img_"+str(id)+"labeled_bboxes"
        plt.title(title)
        plt.imshow(draw_img)
        plt.show()

    return draw_img, heatmap

#----------------------------------------------------------------------
# Helper function that is used to populate plots in multi-plot graphs
#----------------------------------------------------------------------
def populate_plot(rows, cols, subplot_id, img, title, color="rgb", axis='off'):
    plt.subplot(rows,cols,subplot_id)
    plt.axis(axis)
    if color == "gray":
        plt.imshow(img,cmap='gray')
    else :
        plt.imshow(img)
    plt.title(title)

#===============================
# MAIN START
#===============================
if TRAIN_MODEL == "YES":
    # Read in cars and notcars
    cars = []
    notcars = []
    if TRAIN_MODEL_SMALLSET == "YES":
        # not-cars
        images = []
        images.extend( glob.glob('../smallset/non-vehicles_smallset/notcars1/*.jpeg') )
        images.extend( glob.glob('../smallset/non-vehicles_smallset/notcars2/*.jpeg') )
        images.extend( glob.glob('../smallset/non-vehicles_smallset/notcars3/*.jpeg') )
        for image in images:
            notcars.append(image)
        # cars
        images = []
        images.extend( glob.glob('../smallset/vehicles_smallset/cars1/*.jpeg') )
        images.extend( glob.glob('../smallset/vehicles_smallset/cars2/*.jpeg') )
        images.extend( glob.glob('../smallset/vehicles_smallset/cars3/*.jpeg') )
        for image in images:
            cars.append(image)

    if TRAIN_MODEL_LARGESET == "YES":
        # not-cars
        images = []
        images.extend( glob.glob('../largeset/non-vehicles/Extras/*.png') )
        images.extend(glob.glob('../largeset/non-vehicles/GTI/*.png'))
        for image in images:
            notcars.append(image)
        # cars
        images = []
        images.extend( glob.glob('../largeset/vehicles/KITTI_extracted/*.png') )
        images.extend(glob.glob('../largeset/vehicles/GTI_Far/*.png'))
        images.extend(glob.glob('../largeset/vehicles/GTI_Left/*.png'))
        images.extend(glob.glob('../largeset/vehicles/GTI_Right/*.png'))
        images.extend(glob.glob('../largeset/vehicles/GTI_MiddleClose/*.png'))
        for image in images:
            cars.append(image)

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    #sample_size = 500
    #cars = cars[0:sample_size]
    #notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'HLS'# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL" # 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 24#16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
        'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    if SAVE_MODEL == "YES":
        with open('svc_pickle.p', 'wb') as pickle_out:
            pickle.dump(svc, pickle_out)
            pickle.dump(X_scaler, pickle_out)
            pickle.dump(color_space, pickle_out)
            pickle.dump(spatial_size, pickle_out)
            pickle.dump(hist_bins, pickle_out)
            pickle.dump(orient, pickle_out)
            pickle.dump(pix_per_cell, pickle_out)
            pickle.dump(cell_per_block, pickle_out)
            pickle.dump(hog_channel, pickle_out)
            pickle.dump(spatial_feat, pickle_out)
            pickle.dump(hist_feat, pickle_out)
            pickle.dump(hog_feat, pickle_out)
            pickle_out.close()
else:
    # Load the previously trained model
    with open('svc_pickle.p', 'rb') as pickle_in:
        svc = pickle.load(pickle_in)
        X_scaler = pickle.load(pickle_in)
        color_space = pickle.load(pickle_in)
        spatial_size = pickle.load(pickle_in)
        hist_bins = pickle.load(pickle_in)
        orient = pickle.load(pickle_in)
        pix_per_cell = pickle.load(pickle_in)
        cell_per_block = pickle.load(pickle_in)
        hog_channel = pickle.load(pickle_in)
        spatial_feat = pickle.load(pickle_in)
        hist_feat = pickle.load(pickle_in)
        hog_feat = pickle.load(pickle_in)
        pickle_in.close()

#===================================================================
# Generate Report
#===================================================================
if SHOW_CAR_NOT_CAR == "YES":
    car_img_rgb = ndimage.imread("./output_images/car.png", mode='RGB')
    ncar_img_rgb = ndimage.imread("./output_images/not_car.png", mode='RGB')
    plt.subplot(121)
    plt.imshow(car_img_rgb)
    plt.title("car")
    plt.subplot(122)
    plt.imshow(ncar_img_rgb)
    plt.title("not car")
    plt.show()

if SHOW_HOG_FEATURES == "YES":
    car_img_hls = convert_color(car_img_rgb, 'HLS')
    ncar_img_hls = convert_color(ncar_img_rgb, 'HLS')
    car_img_yuv = convert_color(car_img_rgb, 'YUV')
    ncar_img_yuv = convert_color(ncar_img_rgb, 'YUV')
    car_img_hsv = convert_color(car_img_rgb, 'HSV')
    ncar_img_hsv = convert_color(ncar_img_rgb, 'HSV')
    #RGB
    f, car_img_hog_r = get_hog_features(car_img_rgb[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
    f, car_img_hog_g = get_hog_features(car_img_rgb[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
    f, car_img_hog_b = get_hog_features(car_img_rgb[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
    f, ncar_img_hog_r = get_hog_features(ncar_img_rgb[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
    f, ncar_img_hog_g = get_hog_features(ncar_img_rgb[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
    f, ncar_img_hog_b = get_hog_features(ncar_img_rgb[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
    # HSV
    f, car_img_hsv_hog_h = get_hog_features(car_img_hsv[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
    f, car_img_hsv_hog_s = get_hog_features(car_img_hsv[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
    f, car_img_hsv_hog_v = get_hog_features(car_img_hsv[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
    f, ncar_img_hsv_hog_h = get_hog_features(ncar_img_hsv[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
    f, ncar_img_hsv_hog_s = get_hog_features(ncar_img_hsv[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
    f, ncar_img_hsv_hog_v = get_hog_features(ncar_img_hsv[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
    # RGB
    populate_plot(6, 7, 1, car_img_rgb, "car - RGB", "rgb")
    populate_plot(6, 7, 2, car_img_rgb[:, :, 0], "car - R", "gray")
    populate_plot(6, 7, 3, car_img_rgb[:, :, 1], "car - G", "gray")
    populate_plot(6, 7, 4, car_img_rgb[:, :, 2], "car - B", "gray")
    populate_plot(6, 7, 5, car_img_hog_r, "car - HOG-R", "gray")
    populate_plot(6, 7, 6, car_img_hog_g, "car - HOG-G", "gray")
    populate_plot(6, 7, 7, car_img_hog_b, "car - HOG-B", "gray")
    populate_plot(6, 7, 8, ncar_img_rgb, "not car - RGB", "rgb")
    populate_plot(6, 7, 9, ncar_img_rgb[:, :, 0], "not car - R", "gray")
    populate_plot(6, 7, 10, ncar_img_rgb[:, :, 1], "not car - G", "gray")
    populate_plot(6, 7, 11, ncar_img_rgb[:, :, 2], "not car - B", "gray")
    populate_plot(6, 7, 12, ncar_img_hog_r, "not car - HOG-R", "gray")
    populate_plot(6, 7, 13, ncar_img_hog_g, "not car - HOG-G", "gray")
    populate_plot(6, 7, 14, ncar_img_hog_b, "not car - HOG-B", "gray")
    if SHOW_HOG_FEATURES_HLS == "YES":
        # HLS
        f, car_img_hog_h = get_hog_features(car_img_hls[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
        f, car_img_hog_l = get_hog_features(car_img_hls[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
        f, car_img_hog_s = get_hog_features(car_img_hls[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
        f, ncar_img_hog_h = get_hog_features(ncar_img_hls[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
        f, ncar_img_hog_l = get_hog_features(ncar_img_hls[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
        f, ncar_img_hog_s = get_hog_features(ncar_img_hls[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
        populate_plot(6, 7, 15, car_img_rgb, "car - RGB", "rgb")
        populate_plot(6, 7, 16, car_img_hls[:, :, 0], "car - H", "gray")
        populate_plot(6, 7, 17, car_img_hls[:, :, 1], "car - L", "gray")
        populate_plot(6, 7, 18, car_img_hls[:, :, 2], "car - S", "gray")
        populate_plot(6, 7, 19, car_img_hog_h, "car - HOG-H", "gray")
        populate_plot(6, 7, 20, car_img_hog_l, "car - HOG-L", "gray")
        populate_plot(6, 7, 21, car_img_hog_s, "car - HOG-S", "gray")
        populate_plot(6, 7, 22, ncar_img_rgb, "not car - RGB", "rgb")
        populate_plot(6, 7, 23, ncar_img_hls[:, :, 0], "not car - H", "gray")
        populate_plot(6, 7, 24, ncar_img_hls[:, :, 1], "not car - L", "gray")
        populate_plot(6, 7, 25, ncar_img_hls[:, :, 2], "not car - S", "gray")
        populate_plot(6, 7, 26, ncar_img_hog_h, "not car - HOG-H", "gray")
        populate_plot(6, 7, 27, ncar_img_hog_l, "not car - HOG-L", "gray")
        populate_plot(6, 7, 28, ncar_img_hog_s, "not car - HOG-S", "gray")
    else:
        # YUV
        f, car_img_hog_y = get_hog_features(car_img_yuv[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
        f, car_img_hog_u = get_hog_features(car_img_yuv[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
        f, car_img_hog_v = get_hog_features(car_img_yuv[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
        f, ncar_img_hog_y = get_hog_features(ncar_img_yuv[:, :, 0], orient, pix_per_cell, cell_per_block, True, True)
        f, ncar_img_hog_u = get_hog_features(ncar_img_yuv[:, :, 1], orient, pix_per_cell, cell_per_block, True, True)
        f, ncar_img_hog_v = get_hog_features(ncar_img_yuv[:, :, 2], orient, pix_per_cell, cell_per_block, True, True)
        populate_plot(6, 7, 15, car_img_rgb, "car - YUV", "yuv")
        populate_plot(6, 7, 16, car_img_yuv[:, :, 0], "car - Y", "gray")
        populate_plot(6, 7, 17, car_img_yuv[:, :, 1], "car - U", "gray")
        populate_plot(6, 7, 18, car_img_yuv[:, :, 2], "car - V", "gray")
        populate_plot(6, 7, 19, car_img_hog_y, "car - HOG-Y", "gray")
        populate_plot(6, 7, 20, car_img_hog_u, "car - HOG-U", "gray")
        populate_plot(6, 7, 21, car_img_hog_v, "car - HOG-V", "gray")
        populate_plot(6, 7, 22, ncar_img_rgb, "not car - YUV", "yuv")
        populate_plot(6, 7, 23, ncar_img_yuv[:, :, 0], "not car - Y", "gray")
        populate_plot(6, 7, 24, ncar_img_yuv[:, :, 1], "not car - U", "gray")
        populate_plot(6, 7, 25, ncar_img_yuv[:, :, 2], "not car - V", "gray")
        populate_plot(6, 7, 26, ncar_img_hog_y, "not car - HOG-Y", "gray")
        populate_plot(6, 7, 27, ncar_img_hog_u, "not car - HOG-U", "gray")
        populate_plot(6, 7, 28, ncar_img_hog_v, "not car - HOG-V", "gray")
    # HSV
    populate_plot(6, 7, 29, car_img_rgb, "car - HSV", "rgb")
    populate_plot(6, 7, 30, car_img_hsv[:, :, 0], "car - H", "gray")
    populate_plot(6, 7, 31, car_img_hsv[:, :, 1], "car - S", "gray")
    populate_plot(6, 7, 32, car_img_hsv[:, :, 2], "car - V", "gray")
    populate_plot(6, 7, 33, car_img_hsv_hog_h, "car - HOG-H", "gray")
    populate_plot(6, 7, 34, car_img_hsv_hog_s, "car - HOG-S", "gray")
    populate_plot(6, 7, 35, car_img_hsv_hog_v, "car - HOG-V", "gray")
    populate_plot(6, 7, 36, ncar_img_rgb, "not car - HSV", "rgb")
    populate_plot(6, 7, 37, ncar_img_hsv[:, :, 0], "not car - H", "gray")
    populate_plot(6, 7, 38, ncar_img_hsv[:, :, 1], "not car - S", "gray")
    populate_plot(6, 7, 39, ncar_img_hsv[:, :, 2], "not car - V", "gray")
    populate_plot(6, 7, 40, ncar_img_hsv_hog_h, "not car - HOG-H", "gray")
    populate_plot(6, 7, 41, ncar_img_hsv_hog_s, "not car - HOG-S", "gray")
    populate_plot(6, 7, 42, ncar_img_hsv_hog_v, "not car - HOG-V", "gray")
    plt.show()

#===================================================================
# Process the test images
#===================================================================
if PROCESS_TEST_IMAGES == "YES":
    images_fnames = glob.glob('./test_images/*.jpg')
    num_images = images_fnames.__len__()
    idx = 0

    if SHOW_RESULTS == "YES":
        f, ax = plt.subplots(num_images, 2)
        bx = np.reshape(ax, num_images*2)

    for image_fname in images_fnames:
        # Read in an input image
        image = mpimg.imread(image_fname)
        # Process the input image
        draw_img, heatmap, box_list = image_processing_pipeline( image )
        # Show result
        if SHOW_RESULTS =="YES":
            bx[idx].imshow(draw_img)
            bx[idx].set_title(image_fname)
            bx[idx].axis('off')
            bx[idx+1].imshow(heatmap)
            bx[idx+1].set_title("Heat Map")
            bx[idx+1].axis('off')
            idx += 2

    if SHOW_RESULTS =="YES":
        plt.show()

# ===================================================================
# Video processing
# ===================================================================
# Define a class to keep history of bounding boxes
class bboxState():
    def __init__(self):
        self.img_cnt = 0
        self.bbox_lists = []
        self.last_imgs = []
        self.last_heatmaps = []
        self.m_curr = 0
        self.n_curr = 0
        self.n_max = BBoX_HISTORY

class videoPipeline():
    def __init__(self):
        self.img_cnt = 0
        self.bbs = bboxState()

    def image_processing_pipeline(self, img_in):

        if PROCESS_VIDEO_BYPASS == "YES":
            img_out = img_in
        else:
            img_out, heatmap = image_processing_pipeline( img_in, self.bbs )

        self.img_cnt += 1
        return img_out

#===================================================================
# Video processing pipeline
#===================================================================
from moviepy.editor import VideoFileClip
def video_pipeline( input_file, output_file ):

    vp = videoPipeline()

    in_clip = VideoFileClip(input_file)

    out_clip = in_clip.fl_image(vp.image_processing_pipeline)

    out_clip.write_videofile(output_file, audio=False)

#===================================================================
# Process the test video
#===================================================================
if PROCESS_TEST_VIDEO == "YES":
    video_pipeline('test_video.mp4', 'test_video_out.mp4')

#===================================================================
# Process the project video
#===================================================================
if PROCESS_PROJECT_VIDEO == "YES":
    video_pipeline('project_video.mp4', 'project_video_out.mp4')