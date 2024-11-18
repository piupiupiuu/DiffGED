from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
import random
import numpy as np
import os
import torch

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a genn model.
    """
    seed_everything(0)
    args = parameter_parser()
    tab_printer(args)

    trainer = Trainer(args)

    if args.test or args.val:
        trainer.load(args.enable_astar,args.astar_use_net,eval=True)
        trainer.score(testing_graph_set=args.testset)
    else: # training
        trainer.load(args.enable_astar,args.astar_use_net,eval=False)
        trainer.fit()
    
if __name__ == "__main__":
    main()
