import numpy as np
import cv2
from scipy.ndimage.measurements import label
from src import lesson_functions


class VehicleDetector:
    def __init__(self, svc, X_scaler):
        self.last_labels = False
        self.skip_frames = 3
        self.frames_skiped = 0

        self.ystart = 400
        self.ystop = 656
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.scales = [1, 1.5, 2.5, 3.5, 4.5, 5.5]

        self.svc = svc
        self.X_scaler = X_scaler

    def set_frames_skiped(self, frames_skiped):
        self.frames_skiped = frames_skiped

    def find(self, img):
        draw_img = np.copy(img)

        if self.last_labels is False or self.frames_skiped >= self.skip_frames:
            self.frames_skiped = 0

            # create tmp
            heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

            # find cars
            bounding_boxes = self.find_cars(
                img,
                self.ystart,
                self.ystop,
                self.scales,
                self.svc,
                self.X_scaler,
                self.orient,
                self.pix_per_cell,
                self.cell_per_block,
                self.spatial_size,
                self.hist_bins)

            # clean multiple detections and remove false positives
            heatmap = self.create_heatmap(heatmap, bounding_boxes)
            heatmap = self.apply_threshold(heatmap, 2)
            self.last_labels = label(heatmap)

        else:
            self.frames_skiped += 1

        return self.draw_labeled_bboxes(draw_img)


    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop, :, :]
        bounding_boxes = []

        for scale in scales:
            ctrans_tosearch = lesson_functions.convert_color(img_tosearch, conv='RGB2YCrCb')

            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // pix_per_cell)-1
            nyblocks = (ch1.shape[0] // pix_per_cell)-1 
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // pix_per_cell)-1 
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            # Compute individual channel HOG features for the entire image
            hog1 = lesson_functions.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = lesson_functions.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = lesson_functions.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                    # Get color features
                    spatial_features = lesson_functions.bin_spatial(subimg, size=spatial_size)
                    hist_features = lesson_functions.color_hist(subimg, nbins=hist_bins)

                    # Scale features and make a prediction
                    test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                    test_prediction = svc.predict(test_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        bounding_boxes.append( [[xbox_left, ytop_draw+ystart],[xbox_left+win_draw,ytop_draw+win_draw+ystart]] )


        return bounding_boxes


    def create_heatmap(self, heatmap, bbox_list):
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap


    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap


    def draw_labeled_bboxes(self, img):
        # Iterate through all detected cars
        for car_number in range(1, self.last_labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (self.last_labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
