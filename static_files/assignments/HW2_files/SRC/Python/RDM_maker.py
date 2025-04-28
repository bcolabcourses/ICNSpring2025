import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

def RDM_maker(data, region, data_type):
    """
    Generate Representational Dissimilarity Matrix (RDM) from neural data.
    
    Parameters:
    - data: 3D numpy array [n_neurons x n_stimuli x n_times]
    - region: string name of the brain region
    - data_type: string representing data type
    
    Returns:
    - RDM: 3D numpy array [n_stimuli x n_stimuli x n_times]
    """

    n_neurons, n_stimuli, n_times = data.shape
    RDM = np.zeros((n_stimuli, n_stimuli, n_times))

    for t in range(n_times):
        slice_data = data[:, :, t]  # shape: [n_neurons x n_stimuli]

        for i in range(n_stimuli):
            for j in range(n_stimuli):
                cor = np.corrcoef(slice_data[:, i], slice_data[:, j])[0, 1]
                RDM[i, j, t] = 1 - cor  # dissimilarity

    # Visualize one time slice (e.g., time index 60)
    plt.imshow(RDM[:, :, 59], cmap='hot', interpolation='nearest')
    plt.title(f'RDM at time slice 60')
    plt.colorbar()
    plt.show()

    # Save the RDM to .mat file
    savemat(f'RDM_Matrix_{data_type}_{region}.mat', {'RDM': RDM})

    return RDM
