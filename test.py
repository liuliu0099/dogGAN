import argparse as ap
import animal_detection_test as animal_detection
import cartoonize_test as cartoonize
import image_combiation

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default='shapes20220723T1219')
    parser.add_argument('--model_name', type=str, default='mask_rcnn_shapes_0010.h5')
    return parser.parse_args()

def main(args):
    animal_detection.detect_animal(model_folder = args.model_folder, model_name = args.model_name)
    cartoonize.test("./model/animeGAN/cartoonize_Hayao_weight", "./images/anime_background_image", "./images/real_background_image", True)
    cartoonize.test("./model/animeGAN/cartoonize_Shinkai_weight", "./images/anime_animal_image", "./images/real_animal_image", True)
    image_combiation.combine('./images/anime_animal_image', './images/anime_background_image', './images/anime_frames')
    
if __name__=='__main__':
    args = parse_args()
    main(args)