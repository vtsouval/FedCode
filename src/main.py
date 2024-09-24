import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
import shutil
import flwr as fl
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from utils.utils import (str2bool, none_or_str)
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['TORCH_DATA_DIR'] = f'{PROJECT_DIR}/data/torch_datasets/'
if os.path.isdir(f'{PROJECT_DIR}/tmp'): shutil.rmtree(f'{PROJECT_DIR}/tmp')
os.makedirs(f'{PROJECT_DIR}/tmp/', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/assets/', exist_ok=True)

parser = argparse.ArgumentParser(description='Federated Training with Transferring Codebooks')
parser.add_argument('--model', 						type=str, 			default='mobilenet', 						help='Model name (default: mobilenet)')
parser.add_argument("--init_model",					type=none_or_str,	default=None, 			nargs='?',			help="Pretrained Weights (default: None)")
parser.add_argument("--num_rounds",					type=int,			default=100,								help="Number of federated rounds (default: 100)")
parser.add_argument("--num_epochs",					type=int,			default=4,									help="Number of local train epochs (default: 4)")
parser.add_argument("--num_clients",				type=int,			default=10,									help="Number of clients (default: 10)")
parser.add_argument("--max_parallel_executions", 	type=int,			default=10,									help="Number of clients instances to run in parallel (default: 10)")
parser.add_argument("--participation",				type=float,			default=1.,									help="Participation rate (default: 1.0)")
parser.add_argument('--dataset', 					type=str, 			default='cifar10', 							help='Dataset name (default: cifar10)')
parser.add_argument('--split',						type=str,			default='iid', 								help='split type (default: iid)')
parser.add_argument('--batch_size',					type=int,			default=128, 								help='Batch size (default: 64)')
parser.add_argument('--lr',							type=float,			default=1e-3, 								help='Learning rate (default: 1e-3)')
parser.add_argument('--device', 					type=str,			default='cuda', 							help='Device to use (default: cuda)')
parser.add_argument('--seed', 						type=int,			default=42, 								help='Seed value (default: 42)')
parser.add_argument("--timeout",					type=int,			default=1000,								help="Timeout seconds (default: 1000)")
parser.add_argument('--verbose',					type=str2bool,		default=True,								help='Verbosability (default: True)')
parser.add_argument('--store_dir',					type=str,			default=f'{PROJECT_DIR}/assets',			help='Progress store directory (default: ../assets)')
# Extra parameters (Hyperparameters)
parser.add_argument('--num_clusters',				type=int,			default=64,									help='Number of clusters (default: 64)')
parser.add_argument('--start_rnd',					type=int,			default=2,									help='Minimum Round for Codebook Transfer (default: 2)')
parser.add_argument('--broadcast_rate',				type=int,			default=5,									help='Server Calibration Rate (default: 5)')
parser.add_argument('--request_rate',				type=int,			default=2,									help='Clients Calibration Rate (default: 2)')
args = parser.parse_args()

def create_client(cid):
	import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from utils.utils import (grab_gpu, set_seed)
	time.sleep(int(cid)*0.75)
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	set_seed(args.seed)
	import warnings
	warnings.simplefilter("ignore")
	from network import get_model_fn
	from data import get_dataset_fn
	from utils.shard import get_split_fn
	from utils.fed_utils import Client
	return Client(cid=int(cid),
		num_clients=args.num_clients,
		model_loader=get_model_fn(args.model),
		data_loader=get_dataset_fn(args.dataset),
		split_fn=get_split_fn(args.split),
		batch_size=args.batch_size, seed=args.seed,
		device=args.device, verbose=False,
		temp_dir=f'{PROJECT_DIR}/tmp/'
	)

def create_server(args_msg):
	from utils.utils import (grab_gpu, set_seed)
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	set_seed(args.seed)
	import warnings
	warnings.simplefilter("ignore")
	from network import get_model_fn
	from data import get_dataset_fn
	from utils.fed_utils import Server
	return Server(
		num_rounds=args.num_rounds,
		num_clients=args.num_clients,
		participation=args.participation,
		model_loader=get_model_fn(args.model),
		data_loader=get_dataset_fn(args.dataset),
		init_model=args.init_model, batch_size=args.batch_size,
		device=args.device, verbose=args.verbose, lr=args.lr,
		num_epochs=args.num_epochs, args_msg=args_msg,
		file_logger=os.path.join(args.store_dir, f'codebook_{args.model}_{args.dataset}_{args.split}.log'),
		num_clusters=args.num_clusters, start_rnd=args.start_rnd,
		broadcast_freq=args.broadcast_rate, request_freq=args.request_rate,
	)

def main(args_msg):
	# Create server
	server = create_server(args_msg)
	# Start simulation
	history = fl.simulation.start_simulation(
		client_fn=create_client, server=server, num_clients=args.num_clients,
		ray_init_args= {
			"ignore_reinit_error": True,
			"num_cpus": int(min(args.max_parallel_executions, args.num_clients)),
		},
		config=fl.server.ServerConfig(num_rounds=args.num_rounds, round_timeout=args.timeout),)
	shutil.rmtree(f'{PROJECT_DIR}/tmp/')
	return history

if __name__ == "__main__":
	parsed_args = 'Parameters:\n ' + ' '.join(f'{k}={v}\n' for k, v in vars(args).items())
	print(parsed_args)
	history = main(parsed_args)