import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch

from Pruning_SA import NetworkOptimization


def save_csv(arrs, pth, epoch):
    df = pd.DataFrame(np.array(arrs), columns=range(1, epoch + 1))
    if os.path.exists(pth):
        arrs_old = pd.read_csv(pth, index_col=0).values.reshape(-1, epoch)
        arrs = np.concatenate((arrs_old, np.array(arrs)), axis=0)
        df = pd.DataFrame(arrs, columns=range(1, epoch + 1))

    df.to_csv(pth)
    
########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30, help="Epoch size..")
    parser.add_argument('--batch', type=int, default=150, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning r.")
    parser.add_argument('--repeat', type=int, default=5, help="Repeat num.")
    parser.add_argument('--data', type=str, default='mnist',
                        help='select data mnist or fashion.')
    parser.add_argument('--edge', type=int, default=5,
                        help='number of edge changes.')
    # parser.add_argument('--temp', type=float, default=0.2, help="Temperature.")
    # parser.add_argument('--tau', type=str, default='5,1,0.5,0.2,0.05', help="Transverse field.")
    parser.add_argument('--eta', type=float, default=0.98, help="Decay coef.")
    parser.add_argument('--metro', type=str, default='1')
    parser.add_argument('--reduce', type=str, default='10,20,30,40,50,60,70,80,90,95,99,99.8') # 10,20,30,40,50,60,70,80,90,95,99,99.8
    opt = parser.parse_args()

    EPOCH = opt.epoch
    BATCH = opt.batch
    LR = opt.lr
    REPEAT = opt.repeat
    DATA = opt.data
    EDGE = opt.edge
    ETA = opt.eta
    # CONFIG = {'T': opt.temp, 'Tau': opt.tau, 'k': opt.k, 'eta': opt.eta}
    METRO = np.array(opt.metro.split(',')).astype(int)
    REDUCE = np.array(opt.reduce.split(',')).astype(float) * 0.01

    print('Experimental info:', {
          'epoch': opt.epoch, 'batch': opt.batch, 'lr': opt.lr,
          'repeat': opt.repeat, 'data': opt.data,'eta': opt.eta,
          'reduce': REDUCE, 'metro': METRO})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[v] Using device: {}'.format(device))
    
    layer = ['fc1','fc2','fc3','fc4']

    
    nnOpt = copy.deepcopy(NetworkOptimization(batch_size=BATCH,
                                                          learning_rate=LR,
                                                          mask_ratio=[
                                                              0, 0.0, 0.0, 0],
                                                          eta=ETA, data=DATA,
                                                          edge_changed=EDGE))
    losses, accs, epoch = nnOpt.fit(epoch_num=5,  # EPOCH
                                                      save_step=np.inf)
            

    # Set up metropolis loop length. default='1,5,10,20,50,100,200'
    for m in METRO:
        folder = '{}_edge{}_SA/performance_under_different_M/metropolis_{}'.format(DATA, EDGE, m)
        print('Save to: ', folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        print('[v] Start {} metro.'.format(m))
        
        file_loss_untrained = 'loss_untrained.csv'
        file_acc_untrained = 'acc_untrained.csv'
        file_loss_trained = 'loss_trained.csv'
        file_acc_trained = 'acc_trained.csv'
                
        for time in range(REPEAT):
            print('[v] Start {} time try under {} M'.format(time + 1, m))
                
            # file_loss_trained = 'loss_trained.csv'
            # file_acc_trained = 'acc_trained.csv'
            # file_loss_untrained = 'loss_untrained_reduce_{}.csv'.format(r * 100)
            # file_acc_untrained = 'acc_untrained_reduce_{}.csv'.format(r * 100)
            # file_Hq_iter = 'Hq_iter.csv'
            # file_data_iter = 'data_iter.csv'
            
            loss_untrained, acc_untrained = [], []
            loss_trained, acc_trained = [], []
        
            if os.path.exists(os.path.join(folder, file_loss_untrained)):
                arrs_old = pd.read_csv(os.path.join(
                    folder, file_loss_untrained), index_col=0).values
                if len(arrs_old) >= REPEAT:
                    break
                
            loss_untrained.append(losses[-1])
            acc_untrained.append(accs[-1])
            loss_trained.append(losses[-1])
            acc_trained.append(accs[-1])
                
            for r in REDUCE:
                print('[v] Start {:>2}% Reduced Case'.format(r * 100))
            
                nnOpt_SA = copy.deepcopy(nnOpt)
                
                loss, acc, Hq_iter = nnOpt_SA.FCSA(layer, reduce=r, metropolis=m, iteration = 200)
                
                loss_untrained.append(loss)
                acc_untrained.append(acc)

                losses_new, accs_new, epoch = nnOpt_SA.fit(epoch_num=4,  # epoch
                                                            save_step=np.inf)
                loss_trained.append(losses_new[-1])
                acc_trained.append(accs_new[-1])
                
                del nnOpt_SA

                print('[v] Finish {:>2}% Reduced Case'.format(r * 100))
                # break
            
            # del nnOpt
        
            save_csv([loss_untrained], os.path.join(folder, file_loss_untrained), epoch=len(loss_untrained))  # EPOCH
            save_csv([acc_untrained], os.path.join(folder, file_acc_untrained), epoch=len(acc_untrained))  # EPOCH
            save_csv([loss_trained], os.path.join(folder, file_loss_trained), epoch=len(loss_trained))  # EPOCH
            save_csv([acc_trained], os.path.join(folder, file_acc_trained), epoch=len(acc_trained))  # EPOCH
                # save_csv([Hq_iter], os.path.join(folder, file_Hq_iter), epoch=len(Hq_iter))  # EPOCH
                # save_csv([data_iter], os.path.join(folder, file_data_iter), epoch=len(data_iter))  # EPOCH
            print('[v] Finish {} time try under {} M'.format(time + 1, m))
            
        print('[v] Finish {} metro.'.format(m))
        # break
        
    del nnOpt