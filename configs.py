import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MacroWoodClassification Configs', add_help=False)
    
    # Train parameters

    parser = argparse.ArgumentParser('Wood Classification', add_help=False)
    
    # Model parameters
    parser.add_argument('--in_features', default = 224, type = int,
                        help = "Input size of feature")
    parser.add_argument('--out_features', default = 5, type = int,
                        help = "Output size of features")
    parser.add_argument('--nb_nodes', default = 3, type = int)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--num_workers', default = 8, type=int,
                        help="Number of worker in DataLoader")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # 
    return parser