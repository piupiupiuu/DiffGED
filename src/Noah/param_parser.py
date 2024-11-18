import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Noah.")

    parser.add_argument('--beamsize', type=int, default=100,help='Set beamsize for A* beam search.')

    parser.add_argument('--testset', choices=['test', 'small', 'large'],default='test', help="Choose a testing graph set: test, small, or large.")

    parser.add_argument("--hidden-dim",
                        type=list,
                        default=[128,64,32],
	                help="List of hidden dimensions.")
    
    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Neurons in tensor network layer. Default is 16.")
    
    parser.add_argument("--bottle-neck-neurons",
                        type=list,
                        default=[16],
	                help="List of bottle neck layer neurons.")
    
    parser.add_argument("--target-mode",
                        type=str,
                        default='linear',
                        help="The way of generating target, including [linear, exp].")
    
    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")
    
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout probability. Default is 0.5.")

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
                        default='noah',
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