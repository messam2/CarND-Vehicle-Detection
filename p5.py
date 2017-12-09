from lesson_functions import *
from moviepy.editor import VideoFileClip


def process_image(image):
    pass

if __name__ == "__main__":
    video = None #False

    if video:
        input_video = 'project_video.mp4'
        # input_video = 'challenge_video.mp4'
        # input_video = 'harder_challenge_video.mp4'

        white_output = 'output_videos/' + input_video.split('.mp4')[0] + '_output' + '.mp4'

        clip = VideoFileClip(input_video).subclip(0,1)
        # clip = VideoFileClip(input_video)

        white_clip = clip.fl_image(process_image)

        white_clip.write_videofile(white_output, audio=False)
    elif video is False:
        path = 'test_images/*.jpg'
        path = 'test_images/test1.jpg'
        for path in glob.glob(path):
            image = cv2.imread(path)
            output = process_image(image)

            output_image_name = path.split('\\')[-1].split('.jpg')[0] + '_output.jpg'
            cv2.imwrite('output_images/' + output_image_name, output)

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(output)
            plt.title('Ouput Image')
            plt.show()



