import argparse
from train import train
from test import test

def main(args):
    if args.test == False:
        train(args)
    else:
        test(args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Parmeters
    parser.add_argument('--n_epochs', type=float, default=1000,
                        help='max epochs')
    parser.add_argument('--batch_size', type=float, default=32,
                        help='each batch size')
    parser.add_argument('--g_lr', type=float, default=0.0025,
                        help='generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.001,
                        help='discriminator learning rate')
    parser.add_argument('--beta', type=tuple, default=(0.5, 0.5),
                        help='beta for adam')
    parser.add_argument('--d_thresh', type=float, default=0.8,
                        help='for balance dsicriminator and generator')
    parser.add_argument('--z_size', type=float, default=200,
                        help='latent space size')
    parser.add_argument('--z_dis', type=str, default="norm", choices=["norm", "uni"],
                        help='uniform: uni, normal: norm')
    parser.add_argument('--bias', type=str2bool, default=False,
                        help='using cnn bias')
    parser.add_argument('--leak_value', type=float, default=0.2,
                        help='leakeay relu')
    parser.add_argument('--cube_len', type=float, default=32,
                        help='cube length')
    parser.add_argument('--obj', type=str, default="chair",
                        help='tranining dataset object category')
    parser.add_argument('--soft_label', type=str2bool, default=True,
                        help='using soft_label')
    parser.add_argument('--lrsh', type=str2bool, default=True,
                        help='for learning rate shecduler')

    # dir parameters
    parser.add_argument('--output_dir', type=str, default="../output",
                        help='output path')
    parser.add_argument('--input_dir', type=str, default='../input',
                        help='input path')
    parser.add_argument('--pickle_dir', type=str, default='/pickle/',
                        help='input path')
    parser.add_argument('--log_dir', type=str, default='/log/',
                        help='for tensorboard log path save in output_dir + log_dir')
    parser.add_argument('--image_dir', type=str, default='/image/',
                        help='for output image path save in output_dir + image_dir')
    parser.add_argument('--data_dir', type=str, default='/chair/',
                        help='dataset load path')

    # step parameter
    parser.add_argument('--pickle_step', type=int, default=100,
                        help='pickle save at pickle_step epoch')
    parser.add_argument('--log_step', type=int, default=1,
                        help='tensorboard log save at log_step epoch')
    parser.add_argument('--image_save_step', type=int, default=10,
                        help='output image save at image_save_step epoch')

    # other parameters
    parser.add_argument('--model_name', type=str, default="V2",
                        help='this model name for save pickle, logs, output image path and if model_name contain V2 modelV2 excute')
    parser.add_argument('--use_tensorboard', type=str2bool, default=True,
                        help='using tensorboard logging')
    parser.add_argument('--test_iter', type=int, default=10,
                        help='test_epoch number')
    parser.add_argument('--test', type=str2bool, default=False,
                        help='for test')


    args = parser.parse_args()
    main(args)

