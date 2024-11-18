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
    
    seed_everything(0)
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)

    trainer = Trainer(args)

    trainer.score(testing_graph_set=args.testset, algo=args.model_name)

           

if __name__ == "__main__":
    main()
