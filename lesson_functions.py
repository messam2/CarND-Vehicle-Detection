import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler



def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(image)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = cv2.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

def get_cars_notcars(path, vis=False):
    cars = []
    notcars = []

    for image in path:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    data_info = data_look(cars, notcars)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])

    if vis:
        # Just for fun choose random car / not-car indices and plot example images
        car_ind = np.random.randint(0, len(cars))
        notcar_ind = np.random.randint(0, len(notcars))

        # Read in car / not-car images
        car_image = cv2.imread(cars[car_ind])
        notcar_image = cv2.imread(notcars[notcar_ind])

        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(car_image)
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(notcar_image)
        plt.title('Example Not-car Image')
        plt.show()

    return cars, notcars

def find_matches(image, template_list):
    # Define an empty list to take bbox coords
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for temp in template_list:
        # Read in templates one by one
        tmp = mpimg.imread(temp)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(image, tmp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes

    return bbox_list

def convert_cspace(image, cspace='BGR'):
    if cspace != 'BGR':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(image)

    return feature_image

def bin_spatial(image, cspace='BGR', size=(32, 32)):
    feature_image = convert_cspace(image, cspace=cspace)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

def color_hist(image, nbins=32, bins_range=(0, 256), vis=False):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if vis:
        return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features
    else:
        return hist_features

def get_hog_features(image, orient, pix_per_cell, cell_per_block, vis=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if vis == True:
        features, hog_image = hog(gray, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(gray, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=True)
        return features

def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        spatial_features = bin_spatial(image, size=spatial_size)
        hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)

        features.append(np.concatenate((spatial_features, hist_features)))

        # spatial_features = bin_spatial(image, size=spatial_size)
        # hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
        # hog_features = get_hog_features(image, orient=9,
        #                                 pix_per_cell=8,
        #                                 cell_per_block=2,
        #                                 vis=False)
        #
        # features.append(np.concatenate((spatial_features, hist_features, hog_features)))

    # Return list of feature vectors
    return features

if __name__ == "__main__":
    path = 'test_images/test1.jpg'
    image = cv2.imread(path)
    # rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))
    # if rh is not None:
    #     fig = plt.figure(figsize=(12, 3))
    #     plt.subplot(131)
    #     plt.bar(bincen, rh[0])
    #     plt.xlim(0, 256)
    #     plt.title('R Histogram')
    #     plt.subplot(132)
    #     plt.bar(bincen, gh[0])
    #     plt.xlim(0, 256)
    #     plt.title('G Histogram')
    #     plt.subplot(133)
    #     plt.bar(bincen, bh[0])
    #     plt.xlim(0, 256)
    #     plt.title('B Histogram')
    #     fig.tight_layout()
    #     plt.show()

    # bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)),
    #           ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
    # result = draw_boxes(image, bboxes)
    # plt.imshow(result)
    # plt.show()

    # # Select a small fraction of pixels to plot by subsampling it
    # scale = max(image.shape[0], image.shape[1], 64) / 64  # at most 64 rows and columns
    # img_small = cv2.resize(image, (np.int(image.shape[1] / scale), np.int(image.shape[0] / scale)),
    #                        interpolation=cv2.INTER_NEAREST)
    # # Convert subsampled image to desired color space(s)
    # img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
    # img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    # img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
    # # Plot and show
    # plot3d(img_small_RGB, img_small_rgb)
    # plt.show()
    #
    # plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    # plt.show()


    images_path = glob.glob('../images/**/*.png', recursive=True)
    cars, notcars = get_cars_notcars(images_path)

    # # Call our function with vis=True to see an image output
    # car_ind = np.random.randint(0, len(cars))
    # car_image = cv2.imread(cars[car_ind])
    # features, hog_image = get_hog_features(car_image, orient=9,
    #                                        pix_per_cell=8,
    #                                        cell_per_block=2,
    #                                        vis=True)
    # # Plot the examples
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(car_image, cmap='gray')
    # plt.title('Example Car Image')
    # plt.subplot(122)
    # plt.imshow(hog_image, cmap='gray')
    # plt.title('HOG Visualization')
    # plt.show()

    car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256))
    notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                                       hist_bins=32, hist_range=(0, 256))

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(cv2.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()
    else:
        print('Your function only returns empty feature vectors...')
