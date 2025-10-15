import matplotlib.pyplot as plt
import numpy as np
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt
import pandas as pd

def plot_matrix(p):
    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    # Calculate the minimum and maximum values across both the real and imaginary parts of p1
    vmin = np.min([p.real.min(), p.imag.min()])
    vmax = np.max([p.real.max(), p.imag.max()])

    # Plot the first figure and add a colorbar
    im1 = axs[0].matshow(p.real, vmin=-1, vmax=1,cmap = 'seismic')
    axs[0].set_title('Real Part')
    plt.colorbar(im1, ax=axs[0])

    # Plot the second figure and add a colorbar
    im2 = axs[1].matshow(p.imag, vmin=-1, vmax=1,cmap = 'seismic')
    axs[1].set_title('Imaginary Part')
    plt.colorbar(im2, ax=axs[1])

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show the plot
    plt.show()
    
def plot_distributions(ps,titles):
    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, len(ps), figsize=(15,5))

    for i, p in enumerate(ps):
        # Calculate the minimum and maximum values across both the real and imaginary parts of p1
        vmin = np.min([p.min(), p.min()])
        vmax = np.max([p.max(), p.max()])

        # Plot the first figure and add a colorbar
        im1 = axs[i].matshow(p, vmin=0, vmax=1,cmap = 'hot_r')
        axs[i].set_title(titles[i])
    plt.colorbar(im1)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show the plot
    plt.show()
    
def plot_success_rate(success, window_size = 20, label = 'Success Rate'):
    
    # Calculate the moving average success rate
    moving_avg = np.convolve(success, np.ones(window_size)/window_size, mode='valid')

    print("success", success)
    print("moving_avg", moving_avg)
          
    # Plot the success rate and moving average
    #plt.plot(success, label='Success Rate')
    plt.plot(moving_avg, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.legend()

def plot_average_success_rates(success_rates, val_success_rates, trials, hyperparams, window_size=10):
    # Calculate mean and std for success rates
    mean_success = success_rates.mean(axis=0)
    
    
    # Calculate mean and std for validation success rates
    mean_val_success = val_success_rates.mean(axis=0)
    
    # Calculate moving average and moving std for success rates
    mean_success_ma = pd.Series(mean_success).rolling(window=window_size).mean()
    
    # Calculate moving average and moving std for validation success rates
    mean_val_success_ma = pd.Series(mean_val_success).rolling(window=window_size).mean()
    
    plt.figure()
    plt.title(f'Success Rate Plot averaged over {trials} Trials \nHyperparameters: {hyperparams}')
    
    
    # Plot moving average and std for validation success rates
    plt.plot(mean_val_success_ma, label='Mean Validation Success Rate (MA)')

    # Plot individual validation success rates with rolling window
    for i in range(val_success_rates.shape[0]):
        val_success_ma = pd.Series(val_success_rates[i]).rolling(window=window_size).mean()
        plt.plot(val_success_ma, alpha=0.3, label=f'Trial {i+1} Validation Success Rate')

    plt.ylabel('Success Rate')
    plt.xlabel('Epoch')
    # put legend outside of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
        

    pa = qi.DensityMatrix(pa)
    pb = qi.DensityMatrix(pb)
    pab = qi.DensityMatrix(pab)
    
    return (qi.entropy(pa,base=base) + qi.entropy(pb,base=base) - qi.entropy(pab,base=base))/2

