import argparse as ap
import animal_detection_test as animal_detection
import cartoonize_test as cartoonize
import image_combiation
import frame_process
import createVideo

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default='shapes20220723T1219')
    parser.add_argument('--model_name', type=str, default='mask_rcnn_shapes_0010.h5')
    parser.add_argument('--image_folder', type=str, default='images')

    parser.add_argument('--input_video', type=str, default='')
    parser.add_argument('--output_video', type=str, default='output.mp4')
    return parser.parse_args()

def main(args):
    fps = 30
    
    model_folder = args.model_folder
    model_name = args.model_name
    background_model_path = "./model/animeGAN/cartoonize_Hayao_weight/"
    animal_model_path = "./model/animeGAN/cartoonize_Shinkai_weight/"

    input_video_path = args.input_video
    real_image_folder = "./images/real_frames/"
    real_animal_folder = "./images/real_animal_image/"
    real_background_folder = "./images/real_background_image/"
    anime_animal_folder = "./images/anime_animal_image/"
    anime_background_folder = "./images/anime_background_image/"
    anime_image_folder = "./images/anime_frames/"
    output_video_path = args.output_video

    frame_process.get_frames(output_folder=real_image_folder, fps=fps, source_file=input_video_path)
    animal_detection.detect_animal(model_folder = model_folder, model_name = model_name)
    cartoonize.test(checkpoint_dir=background_model_path, result_dir=anime_background_folder, test_dir=real_background_folder , if_adjust_brightness=True)
    cartoonize.test(checkpoint_dir=animal_model_path, result_dir=anime_animal_folder, test_dir=real_animal_folder, if_adjust_brightness=True)
    image_combiation.combine(animal_folder=anime_animal_folder, background_folder=anime_background_folder, output_folder=anime_image_folder)
    createVideo.frames2video(image_folder=anime_image_folder,output_file=output_video_path, fps=fps)
    
if __name__=='__main__':
    args = parse_args()
    main(args)