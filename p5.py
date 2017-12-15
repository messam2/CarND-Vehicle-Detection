from lesson_functions import *
import pickle

def process_image(image):
    out_img = find_cars(image, ystart=400, ystop=650, clf=clf, X_scaler=X_scaler, parameters=parameters)

    return out_img

if __name__ == "__main__":
    video = False

    pickle_file = 'classifier.p'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        X_scaler = pickle_data['X_scaler']
        parameters = pickle_data['parameters']
        clf = pickle_data['clf']

        del pickle_data  # Free up memory

    print('Data and modules loaded.')

    for k in parameters:
        print(k, ":", parameters[k])

    if video:
        from moviepy.editor import VideoFileClip

        # input_video = 'project_video.mp4'
        input_video = 'test_video.mp4'

        white_output = 'output_videos/' + input_video.split('.mp4')[0] + '_output' + '.mp4'

        # clip = VideoFileClip(input_video).subclip(10,11)
        clip = VideoFileClip(input_video)

        white_clip = clip.fl_image(process_image)

        white_clip.write_videofile(white_output, audio=False)
    elif video is False:

        color_space = parameters['color_space']
        orient = parameters['orient']
        pix_per_cell = parameters['pix_per_cell']
        cell_per_block = parameters['cell_per_block']
        hog_channel = parameters['hog_channel']
        spatial_size = parameters['spatial_size']
        hist_bins = parameters['hist_bins']
        spatial_feat = parameters['spatial_feat']
        hist_feat = parameters['hist_feat']
        hog_feat = parameters['hog_feat']

        images_path = glob.glob('../images/**/*.png', recursive=True)

        cars, notcars = get_cars_notcars(images_path)
        car_ind = np.random.randint(0, len(cars))
        car_image = mpimg.imread(cars[car_ind])

        feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
        hog_features, hog_image = get_hog_features(feature_image, orient=8,
                                                   pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=True)
        # fig = plt.figure(figsize=(12,12))
        # plt.subplot(121)
        # plt.imshow(car_image, cmap='gray')
        # plt.title('Example Car Image')
        # plt.subplot(122)
        # plt.imshow(hog_image, cmap='gray')
        # plt.title('HOG Visualization')
        # plt.savefig('md_images/hog_image.jpg')
        # plt.show()

        # path, split_str = 'test_images/*.jpg', '\\'
        path, split_str = 'test_images/test1.jpg', '/'

        for i, path in enumerate(glob.glob(path)):
            image = mpimg.imread(path)
            # output = process_image(image)

            ###for md images
            draw_image = np.copy(image)
            image = image.astype(np.float32) / 255
            y_start_stop = [450, 650]  # Min and max in y to search in slide_window()
            windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                                   xy_window=(64, 64), xy_overlap=(0.8, 0.8))

            windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                    xy_window=(96, 96), xy_overlap=(0.8, 0.8))

            windows += slide_window(image, x_start_stop=[1000, None], y_start_stop=y_start_stop,
                                    xy_window=(128, 128), xy_overlap=(0.8, 0.8))

            hot_windows = search_windows(image, windows, clf, X_scaler, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)

            window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

            # fig = plt.figure(figsize=(12, 12))
            # plt.imshow(window_img)
            # plt.title('windows Image')
            # # plt.savefig('md_images/window_image.png')
            # plt.savefig('md_images/sliding_windows' + str(i + 1) + '.jpg')
            # plt.show()

            heat = np.zeros_like(image[:, :, 0]).astype(np.float)
            heat = add_heat(heat, hot_windows)

            # Apply threshold to help remove false positives
            heat = apply_threshold(heat, 1)

            # Visualize the heatmap when displaying
            heatmap = np.clip(heat, 0, 255)

            # fig = plt.figure(figsize=(12, 12))
            # plt.subplot(121)
            # plt.imshow(window_img)
            # plt.title('window Image')
            # plt.subplot(122)
            # plt.imshow(heatmap)
            # plt.title('Heat map Image')
            # plt.savefig('md_images/bboxes_and_heat.png')
            # plt.show()

            # Find final boxes from heatmap using label function
            labels = label(heatmap)

            # fig = plt.figure(figsize=(12, 12))
            # plt.imshow(labels[0], cmap='gray')
            # plt.title('labels map')
            # plt.savefig('md_images/labels_map.png')
            # plt.show()

            draw_image = draw_labeled_bboxes(np.copy(draw_image), labels)

            fig = plt.figure(figsize=(12, 12))
            plt.imshow(draw_image)
            plt.title('output bboxes')
            plt.savefig('md_images/output_bboxes.png')
            plt.show()



