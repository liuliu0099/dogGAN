import argparse as ap
import animal_detection_train as animal_detection

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default='shapes20220723T1219')
    parser.add_argument('--model_name', type=str, default='mask_rcnn_shapes_0010.h5')
    return parser.parse_args()

def main(args):
    animal_detection.train()
    
    
if __name__=='__main__':
    args = parse_args()
    main(args)