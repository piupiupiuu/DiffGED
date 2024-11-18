import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Traditional.")

    parser.add_argument("--model-name",
                        choices=['hungarian', 'vj'],
                        default='hungarian',
                        help="model name")
                        
    parser.add_argument("--dataset",
                        type=str,
                        default='AIDS',
                        help="dataset name")
    
    parser.add_argument('--testset', choices=['test', 'small', 'large'],default='test', help="Choose a testing graph set: test, small, or large.")
    parser.add_argument("--num-delta-graphs",
                        type=int,
                        default=100,
                        help="The number of synthetic delta graph pairs for each graph.")

    parser.add_argument("--num-testing-graphs",
                        type=int,
                        default=100,
                        help="The number of testing graph pairs for each graph.")
    
    parser.add_argument("--model-path",
                        type=str,
                        default='model_save/',
                        help="Where to save the trained model")
    
    parser.add_argument("--abs-path",
                        type=str,
                        default="../",
                        help="the absolute path")
    
    parser.add_argument("--result-path",
                        type=str,
                        default='result/',
                        help="Where to save the evaluation results")
    
    return parser.parse_args()