import argparse
import shutil
import os

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--ngpu',  type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--gpuids', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
parser.add_argument('--num_epochs', type=int, default=20)

parser.add_argument('--world_dim',type=int, default=3)
parser.add_argument("--display_fre", type=int, default =10)

parser.add_argument('--crop_size', type=int, default = [64, 128])

parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument("--beta1", type=float, default = 0.5)
parser.add_argument('--gamma', type=float, default=0.9)

parser.add_argument('--group_dir', default='/data/tianchiSR/dataset/', type=str, metavar='PATH')
parser.add_argument('--model_path', default='result/', type=str, metavar='PATH')
parser.add_argument('--save_dir', default='result/', type=str, metavar='PATH', help='path to validation images')
parser.add_argument('--log_dir', default='', type=str, metavar='PATH', help='path to validation images')
parser.add_argument('--resume', action='store_true')

parser.add_argument('--phase', default='train',type=str)
parser.add_argument('--nframes', type=int, default = 7)

args = parser.parse_args()
gpuids = [int(_) for _ in range(args.ngpu)]
args.log_dir = os.path.join(args.save_dir,'log')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if os.path.exists(args.log_dir):
    shutil.rmtree(args.log_dir)