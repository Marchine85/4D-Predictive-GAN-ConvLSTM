import os
import numpy as np
import matplotlib.pyplot as plt

def render_graphs(save_dir, track_g_loss, track_d_loss, track_d_val_loss, track_MSE_train_loss, track_MSE_val_loss, track_JACCARD_train_loss, track_JACCARD_val_loss, iterations_arr, val_iterations_arr, restart_training): 
    # try: 
        if not os.path.exists(save_dir+'/plots/'):
            os.makedirs(save_dir+'/plots/')
            
        # SAVE HISTORY TO FILE:
        if restart_training:
            LOSSES = np.concatenate((np.asarray(iterations_arr)[...,np.newaxis], np.asarray(track_g_loss)[...,np.newaxis], np.asarray(track_d_loss)[...,np.newaxis], np.asarray(track_MSE_train_loss)[...,np.newaxis], np.asarray(track_JACCARD_train_loss)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,           G_LOSS,           D_LOSS,         MSE_LOSS,     JACCARD_LOSS"
            np.savetxt(save_dir+'/plots/_train_losses_restart.txt' , LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
            
            VAL_LOSSES = np.concatenate((np.asarray(val_iterations_arr)[...,np.newaxis], np.asarray(track_d_val_loss)[...,np.newaxis], np.asarray(track_MSE_val_loss)[...,np.newaxis], np.asarray(track_JACCARD_val_loss)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,       D_VAL_LOSS,     MSE_VAL_LOSS, JACCARD_VAL_LOSS"
            np.savetxt(save_dir+'/plots/_val_losses_restart.txt' , VAL_LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
        else:
            LOSSES = np.concatenate((np.asarray(iterations_arr)[...,np.newaxis], np.asarray(track_g_loss)[...,np.newaxis], np.asarray(track_d_loss)[...,np.newaxis], np.asarray(track_MSE_train_loss)[...,np.newaxis], np.asarray(track_JACCARD_train_loss)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,           G_LOSS,           D_LOSS,         MSE_LOSS,     JACCARD_LOSS"
            np.savetxt(save_dir+'/plots/_train_losses.txt' , LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
            
            VAL_LOSSES = np.concatenate((np.asarray(val_iterations_arr)[...,np.newaxis], np.asarray(track_d_val_loss)[...,np.newaxis], np.asarray(track_MSE_val_loss)[...,np.newaxis], np.asarray(track_JACCARD_val_loss)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,       D_VAL_LOSS,     MSE_VAL_LOSS, JACCARD_VAL_LOSS"
            np.savetxt(save_dir+'/plots/_val_losses.txt' , VAL_LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
    
                      
        track_d_loss = np.ma.masked_invalid(track_d_loss)#.compressed()
        track_g_loss = np.ma.masked_invalid(track_g_loss)#.compressed()
        high_d = np.percentile(track_d_loss, 99)
        high_g = np.percentile(track_g_loss, 99)
        low_d = np.percentile(track_d_loss, 1)
        low_g = np.percentile(track_g_loss, 1)
        if (len(track_d_val_loss) > 0) and (len(val_iterations_arr) > 0):
            track_d_val_loss = np.ma.masked_invalid(track_d_val_loss)#.compressed()
            high_d_val = np.percentile(track_d_val_loss, 95)
            low_d_val = np.percentile(track_d_val_loss, 5)
            high_y = max([high_d, high_g, high_d_val])
            low_y = min([low_d, low_g, low_d_val]) #- 0.5*min([high_d, high_g])
        else:
            high_y = max([high_d, high_g])
            low_y = min([low_d, low_g]) #- 0.5*min([high_d, high_g])


        fig = plt.figure()            
        plt.plot(iterations_arr, track_g_loss, color='crimson', alpha=0.5, label='G-loss')
        plt.plot(iterations_arr, track_d_loss, color='navy', alpha=0.5, label='D-loss')
        #plt.plot(iterations_arr, track_MSE_train_loss, color='seagreen', alpha=0.5, label='MSE-Train-loss')
        
        if (len(track_d_val_loss) > 0) and (len(val_iterations_arr) > 0):
            plt.plot(val_iterations_arr, track_d_val_loss, color='navy', alpha=0.5, linestyle='dashed', label='D-Val-loss')
            #plt.plot(val_iterations_arr, track_MSE_val_loss, color='seagreen', alpha=0.5, linestyle='dashed', label='MSE-Val-loss')
        
        plt.legend() 
        plt.title("LOSS")
        plt.xlabel('Training Iteration (processed batches)')
        plt.ylabel('Loss')
        plt.ylim([low_y - np.abs(0.5*low_y), high_y + np.abs(0.5*high_y)])
        plt.grid(True)
        if restart_training:
            plt.savefig(save_dir+'/plots/DISCGEN_LOSS_restart.png' )
        else:
            plt.savefig(save_dir+'/plots/DISCGEN_LOSS.png' )
        plt.clf()
        plt.close(fig)

        if (len(track_MSE_val_loss) > 0) and (len(val_iterations_arr) > 0):
            fig = plt.figure()          
            plt.plot(iterations_arr, track_MSE_train_loss, color='seagreen', alpha=0.5, label='MSE-Train-loss')
            plt.plot(val_iterations_arr, track_MSE_val_loss, color='navy', alpha=0.5, label='MSE-Val-loss')
            high_MSE_train = np.percentile(track_MSE_train_loss, 95)
            low_MSE_train = np.percentile(track_MSE_train_loss, 5)
            high_MSE_val = np.percentile(track_MSE_val_loss, 95)
            low_MSE_val = np.percentile(track_MSE_val_loss, 5)
            
            high_y = max([high_MSE_train, high_MSE_val])
            low_y = min([low_MSE_train, low_MSE_val])

            plt.legend() 
            plt.title("MSE")
            plt.xlabel('Training Iteration (processed batches)')
            plt.ylabel('Loss')
            plt.ylim([low_y - np.abs(0.5*low_y), high_y + np.abs(0.5*high_y)])
            plt.grid(True)
            if restart_training:
                plt.savefig(save_dir+'/plots/MSE_LOSS_restart.png' )
            else:
                plt.savefig(save_dir+'/plots/MSE_LOSS.png' )
            plt.clf()
            plt.close(fig)

        if (len(track_JACCARD_val_loss) > 0) and (len(val_iterations_arr) > 0):
            fig = plt.figure()          
            plt.plot(iterations_arr, track_JACCARD_train_loss, color='seagreen', alpha=0.5, label='JACCARD-Train-loss')
            plt.plot(val_iterations_arr, track_JACCARD_val_loss, color='navy', alpha=0.5, label='JACCARD-Val-loss')
            high_JACCARD_train = np.percentile(track_JACCARD_train_loss, 95)
            low_JACCARD_train = np.percentile(track_JACCARD_train_loss, 5)
            high_JACCARD_val = np.percentile(track_JACCARD_val_loss, 95)
            low_JACCARD_val = np.percentile(track_JACCARD_val_loss, 5)
            
            high_y = max([high_JACCARD_train, high_JACCARD_val])
            low_y = min([low_JACCARD_train, low_JACCARD_val])

            plt.legend() 
            plt.title("JACCARD")
            plt.xlabel('Training Iteration (processed batches)')
            plt.ylabel('Loss')
            plt.ylim([low_y - np.abs(0.5*low_y), high_y + np.abs(0.5*high_y)]) 
            plt.grid(True)
            if restart_training:
                plt.savefig(save_dir+'/plots/JACCARD_LOSS_restart.png' )
            else:
                plt.savefig(save_dir+'/plots/JACCARD_LOSS.png' )
            plt.clf()
            plt.close(fig)