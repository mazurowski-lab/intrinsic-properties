# utils for dimensionality estimates of things
import os
import sys

def estimate_intrinsic_dim(dataset, dataset_name, estimator, batchsize=10000, hyperparam_k=20):
    """ 
    basic function for estimating integer intrinsic dim (ID) of a high dim dataset, using code from https://github.com/ppope/dimensions
    
    args:
        dataset: PyTorch dataset. dataset you're estimating ID of.
        dataset_name: string. name of the dataset.
        estimator: string. Name of ID estimator.
        batchsize (optional): int. batchsize used to feed data to estimator.
        hyperparam_k (options): int. hyperparameter k setting to use for final ID estimate
            (not used for all estimators)
    """
    import_dir = 'dimensions'
    sys.path.append(os.path.join(os.getcwd(), import_dir))

    from main import run_mle, run_geomle, run_twonn
    from argparse import Namespace
    
    args = Namespace(
        estimator=estimator,
        k1=25,
        k2=55,# default
        single_k=True,
        eval_every_k=True,
        average_inverse=True,
        max_num_samples=1000,
        save_path='logs/{}_{}_log.json'.format(dataset_name, estimator),
        
        anchor_samples=0,
        anchor_ratio=0,
        bsize=batchsize, #batch size for previous images
        n_workers=1,
        
        # GeoMLE args
        nb_iter1=1,
        nb_iter2=20,
        inv_mle=False
    )
    
    
    if estimator == "mle":
        results = run_mle(args, dataset)
        intrinsic_dim = results['k{}_inv_mle_dim'.format(hyperparam_k)]
    elif estimator == "geomle":
        results = run_geomle(args, dataset)
        intrinsic_dim = results['dim']
    elif estimator == "twonn":
        results = run_twonn(args, dataset)
        intrinsic_dim = results['dim']
    else:
        raise NotImplementedError
        
    return intrinsic_dim