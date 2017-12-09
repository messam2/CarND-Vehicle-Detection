import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def color_hist(image, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
    ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
    bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector

    return rhist, ghist, bhist, bin_centers, hist_features

def process_image(image):
    rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))
    if rh is not None:
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()

        plt.show()

if __name__ == "__main__":
    video = False

    if video:
        input_video = 'project_video.mp4'
        # input_video = 'challenge_video.mp4'
        # input_video = 'harder_challenge_video.mp4'

        white_output = 'output_videos/' + input_video.split('.mp4')[0] + '_output' + '.mp4'

        clip = VideoFileClip(input_video).subclip(0,1)
        # clip = VideoFileClip(input_video)

        white_clip = clip.fl_image(process_image)

        white_clip.write_videofile(white_output, audio=False)

    else:
        path = 'test_images/*.jpg'
        path = 'test_images/test1.jpg'
        for path in glob.glob(path):
            img = cv2.imread(path)
            # output = process_image(img)
            #
            # output_image_name = path.split('\\')[-1].split('.jpg')[0] + '_output.jpg'
            # cv2.imwrite('output_images/' + output_image_name, output)
            #
            # fig = plt.figure(figsize=(6, 6))
            # plt.imshow(output)
            # plt.title('Ouput Image')
            # plt.show()
            #
            process_image(img)

