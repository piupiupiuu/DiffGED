"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run DiffGED.")
    
    parser.add_argument('--topk-approach', choices=['parallel','sequential'],default='parallel', help="Choose a top-k mapping generation approach: parallel, or sequential.")
    
    parser.add_argument('--test-k', type=int, default=100,help='Set k for inference.')
    
    parser.add_argument('--k-range', type=list, default=[1,10,20,30,40,50,60,70,80,90,100],help='range of k for top-k approach analysis.')
    
    parser.add_argument('--experiment', choices=['test', 'topk_analysis', 'diversity_analysis'],default='test', help="Choose an experiment: test, topk_analysis, or diversity_analysis.")
    
    parser.add_argument('--testset', choices=['test', 'small', 'large'],default='test', help="Choose a testing graph set: test, small, or large.")

    parser.add_argument('--diffusion-steps', type=int, default=1000)

    parser.add_argument('--inference-diffusion_steps', type=int, default=10)

    parser.add_argument("--hidden-dim",
                        type=list,
                        default=[128,64,32,32,32,32],
	                help="List of hidden dimensions.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")


    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")


    parser.add_argument("--abs-path",
                        type=str,
                        default="../",
                        help="the absolute path")

    parser.add_argument("--result-path",
                        type=str,
                        default='result/',
                        help="Where to save the evaluation results")

    parser.add_argument("--model-train",
                        type=int,
                        default=1,
                        help='Whether to train the model')

    parser.add_argument("--model-path",
                        type=str,
                        default='model_save/',
                        help="Where to save the trained model")

    parser.add_argument("--model-epoch-start",
                        type=int,
                        default=0,
                        help="The number of epochs the initial saved model has been trained.")

    parser.add_argument("--model-epoch-end",
                        type=int,
                        default=0,
                        help="The number of epochs the final saved model has been trained.")

    parser.add_argument("--dataset",
                        type=str,
                        default='AIDS',
                        help="dataset name")

    parser.add_argument("--model-name",
                        type=str,
                        default='DiffMatch',
                        help="model name")


    parser.add_argument("--num-delta-graphs",
                        type=int,
                        default=100,
                        help="The number of synthetic delta graph pairs for each graph.")

    parser.add_argument("--num-testing-graphs",
                        type=int,
                        default=100,
                        help="The number of testing graph pairs for each graph.")

    

    return parser.parse_args()
