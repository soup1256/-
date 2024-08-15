import argparse
import template

parser = argparse.ArgumentParser(description='RIDNET')

parser.add_argument('--debug', action='store_true', help='Enables debug mode')
parser.add_argument('--template', default='.', help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default=r'D:\eecsproject\archive', help='数据集根目录')  # 更新为你的数据集路径
parser.add_argument('--dir_demo', type=str, default='../test', help='demo image directory')
parser.add_argument('--data_train', type=str, default='MyImage', help='训练集数据集名称')
parser.add_argument('--data_test', type=str, default='MyImage', help='测试集数据集名称')
parser.add_argument('--patch_size', type=int, default=192, help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--noise_g', type=str, default='25', help='Gaussian noise levels for training, separated by + (e.g., "10+20+30")')
parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')
parser.add_argument('--noise', type=str, default='.', help='Gaussian noise std.')

# Model specifications
parser.add_argument('--model', default='RIDNET', help='model name')
parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--pre_train', type=str, default='.', help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.', help='pre-trained model directory')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1, help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1, help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=300, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for feature channel reduction
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1', help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default=1e6, help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test', help='file name to save')
parser.add_argument('--load', type=str, default='.', help='file name to load')
parser.add_argument('--resume', type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true', help='print model')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', help='save output results')

# Options for test
parser.add_argument('--testpath', type=str, default='../dataset', help='dataset directory for testing')  # 更新为你的测试路径
parser.add_argument('--testset', type=str, default='MyImageDataset', help='dataset name for testing')  # 更新为你的测试集名称

parser.add_argument('--device', type=str, default='cuda:0',
                    help='device to use for training / testing')

# Additional arguments for noise level progression
parser.add_argument('--initial_noise_level', type=int, default=10, help='Initial noise level')
parser.add_argument('--max_noise_level', type=int, default=50, help='Maximum noise level')
parser.add_argument('--noise_increment_per_epoch', type=int, default=5, help='Noise level increment per epoch')

args = parser.parse_args()
template.set_template(args)

# Handle noise_g as a list of integers
args.noise_g = list(map(int, args.noise_g.split('+')))

if args.epochs == 0:
    args.epochs = int(1e8)

for arg in vars(args):
    value = getattr(args, arg)
    if isinstance(value, str) and value.lower() == 'true':
        setattr(args, arg, True)
    elif isinstance(value, str) and value.lower() == 'false':
        setattr(args, arg, False)
