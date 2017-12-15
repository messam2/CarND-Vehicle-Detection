from lesson_functions import *
from moviepy.editor import VideoFileClip
import pickle

def process_image(image):
    out_img = find_cars(image, ystart=400, ystop=650, clf=clf, X_scaler=X_scaler, parameters=parameters)

    return out_img

if __name__ == "__main__":
    video = True

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
        input_video = 'project_video.mp4'
        # input_video = 'test_video.mp4'

        white_output = 'output_videos/' + input_video.split('.mp4')[0] + '_output' + '.mp4'

        # clip = VideoFileClip(input_video).subclip(10,11)
        clip = VideoFileClip(input_video)

        white_clip = clip.fl_image(process_image)

        white_clip.write_videofile(white_output, audio=False)
    elif video is False:
        path = 'test_images/*.jpg'
        # path = 'test_images/test1.jpg'
        for path in glob.glob(path):
            image = mpimg.imread(path)
            output = process_image(image)

            output_image_name = path.split('/')[-1].split('.jpg')[0] + '_output.jpg'
            mpimg.imsave('output_images/' + output_image_name, output)

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(output)
            plt.title('Ouput Image')
            plt.show()



