import os
import numpy as np
import matplotlib.pyplot as plt

def render_graphs(save_dir, BCE_losses_train, MSE_losses_train, JACCARD_losses_train, BCE_losses_val_mean, MSE_losses_val_mean, JACCARD_losses_val_mean, iterations_arr, val_iterations_arr, restart_training): 
    # try: 
        if not os.path.exists(save_dir+'/plots/'):
            os.makedirs(save_dir+'/plots/')
            
        # SAVE HISTORY TO FILE:
        if restart_training:
            LOSSES = np.concatenate((np.asarray(iterations_arr)[...,np.newaxis], np.asarray(BCE_losses_train)[...,np.newaxis], np.asarray(MSE_losses_train)[...,np.newaxis], np.asarray(JACCARD_losses_train)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,         BCE_LOSS,         MSE_LOSS,     JACCARD_LOSS"
            np.savetxt(save_dir+'/plots/_train_losses_restart.txt' , LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
            
            VAL_LOSSES = np.concatenate((np.asarray(val_iterations_arr)[...,np.newaxis], np.asarray(BCE_losses_val_mean)[...,np.newaxis], np.asarray(MSE_losses_val_mean)[...,np.newaxis], np.asarray(JACCARD_losses_val_mean)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,     BCE_VAL_LOSS,     MSE_VAL_LOSS, JACCARD_VAL_LOSS"
            np.savetxt(save_dir+'/plots/_val_losses_restart.txt' , VAL_LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
        else:
            LOSSES = np.concatenate((np.asarray(iterations_arr)[...,np.newaxis], np.asarray(BCE_losses_train)[...,np.newaxis], np.asarray(MSE_losses_train)[...,np.newaxis], np.asarray(JACCARD_losses_train)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,         BCE_LOSS,         MSE_LOSS,     JACCARD_LOSS"
            np.savetxt(save_dir+'/plots/_train_losses.txt' , LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
            
            VAL_LOSSES = np.concatenate((np.asarray(val_iterations_arr)[...,np.newaxis], np.asarray(BCE_losses_val_mean)[...,np.newaxis], np.asarray(MSE_losses_val_mean)[...,np.newaxis], np.asarray(JACCARD_losses_val_mean)[...,np.newaxis]), axis=1)
            header = "TRAINING ITERATION,     BCE_VAL_LOSS,     MSE_VAL_LOSS, JACCARD_VAL_LOSS"
            np.savetxt(save_dir+'/plots/_val_losses.txt' , VAL_LOSSES, fmt='%20d, %16.5f, %16.5f, %16.5f', delimiter=',', header=header)
    
                      
        track_BCE_loss = np.ma.masked_invalid(BCE_losses_train)#.compressed()

        high_BCE = np.percentile(track_BCE_loss, 99)
        low_BCE = np.percentile(track_BCE_loss, 1)
        if (len(BCE_losses_val_mean) > 0) and (len(val_iterations_arr) > 0):
            BCE_losses_val_mean = np.ma.masked_invalid(BCE_losses_val_mean)#.compressed()
            high_BCE_val = np.percentile(BCE_losses_val_mean, 95)
            low_BCE_val = np.percentile(BCE_losses_val_mean, 5)
            high_y = max([high_BCE, high_BCE_val])
            low_y = min([low_BCE, low_BCE_val]) #- 0.5*min([high_d, high_g])
        else:
            high_y = max(high_BCE)
            low_y = min(low_BCE) #- 0.5*min([high_d, high_g])


        fig = plt.figure()            
        plt.plot(iterations_arr, track_BCE_loss, color='crimson', alpha=0.5, label='BCE-loss')
        #plt.plot(iterations_arr, track_MSE_train_loss, color='seagreen', alpha=0.5, label='MSE-Train-loss')
        
        if (len(BCE_losses_val_mean) > 0) and (len(val_iterations_arr) > 0):
            plt.plot(val_iterations_arr, BCE_losses_val_mean, color='navy', alpha=0.5, label='BCE-Val-loss')
            #plt.plot(val_iterations_arr, track_MSE_val_loss, color='seagreen', alpha=0.5, linestyle='dashed', label='MSE-Val-loss')
        
        plt.legend() 
        plt.title("LOSS")
        plt.xlabel('Training Iteration (processed batches)')
        plt.ylabel('Loss')
        plt.ylim([low_y - np.abs(0.5*low_y), high_y + np.abs(0.5*high_y)])
        plt.grid(True)
        if restart_training:
            plt.savefig(save_dir+'/plots/BCE_LOSS_restart.png' )
        else:
            plt.savefig(save_dir+'/plots/BCE_LOSS.png' )
        plt.clf()
        plt.close(fig)

        if (len(MSE_losses_val_mean) > 0) and (len(val_iterations_arr) > 0):
            fig = plt.figure()          
            plt.plot(iterations_arr, MSE_losses_train, color='crimson', alpha=0.5, label='MSE-Train-loss')
            plt.plot(val_iterations_arr, MSE_losses_val_mean, color='navy', alpha=0.5, label='MSE-Val-loss')
            high_MSE_train = np.percentile(MSE_losses_train, 95)
            low_MSE_train = np.percentile(MSE_losses_train, 5)
            high_MSE_val = np.percentile(MSE_losses_val_mean, 95)
            low_MSE_val = np.percentile(MSE_losses_val_mean, 5)
            
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

        if (len(JACCARD_losses_val_mean) > 0) and (len(val_iterations_arr) > 0):
            fig = plt.figure()          
            plt.plot(iterations_arr, JACCARD_losses_train, color='crimson', alpha=0.5, label='JACCARD-Train-loss')
            plt.plot(val_iterations_arr, JACCARD_losses_val_mean, color='navy', alpha=0.5, label='JACCARD-Val-loss')
            high_JACCARD_train = np.percentile(JACCARD_losses_train, 95)
            low_JACCARD_train = np.percentile(JACCARD_losses_train, 5)
            high_JACCARD_val = np.percentile(JACCARD_losses_val_mean, 95)
            low_JACCARD_val = np.percentile(JACCARD_losses_val_mean, 5)
            
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