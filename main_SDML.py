def main(config):
    from SDML import Solver
    solver = Solver(config)
    cudnn.benchmark = True
    return solver.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_all', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--just_valid', type=bool, default=False) # wiki, pascal, nus-wide, xmedianet
    parser.add_argument('--multiprocessing', type=bool, default=False)
    parser.add_argument('--running_time', type=bool, default=False)


    parser.add_argument('--lr', type=list, default=[1e-4, 2e-4, 2e-4, 2e-4, 2e-4])
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--output_shape', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--datasets', type=str, default='wiki_doc2vec') # xmedia, wiki_doc2vec, MSCOCO_doc2vec, nus_wide_doc2vec
    parser.add_argument('--view_id', type=int, default=-1)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)

    config = parser.parse_args()

    seed = 123
    print('seed: ' + str(seed))
    import numpy as np
    np.random.seed(seed)
    import random as rn
    rn.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn
    cudnn.enabled = False

    results = main(config)

    # print(config)
    # import scipy.io as sio
    # if config.running_time:
    #     runing_time = []
    #     for i in range(1):
    #         print('%d-th running time test', i)
    #         results = main(config)
    #         runing_time.append(results)
    #     print('average running time: %f', np.mean(runing_time))
    # else:
    #     results = main(config)
    #     if config.just_valid:
    #         sio.savemat('para_results/params_' + config.datasets + '_' + str(config.batch_size) + '_' + str(config.output_shape) + '_' + str(config.alpha) + '_' + str(config.epochs) + '_' + str(config.lr) + '_loss.mat', {'val_d_loss': np.array(results[0]), 'tr_d_loss': np.array(results[1]), 'tr_ae_loss': np.array(results[2])})
    #     else:
    #         sio.savemat('results/params_' + config.datasets + '_' + str(config.batch_size) + '_' + str(config.output_shape) + '_' + str(config.alpha) + '_' + str(config.epochs) + '_' + str(config.lr) + '_resutls.mat', {'results': np.array(results)})
