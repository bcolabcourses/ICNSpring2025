import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.metrics import confusion_matrix
from gen_fx_get_svm import gen_fx_get_svm  # Make sure this is implemented correctly

def svm_classifier_psth(manmade_data, body_data, natural_data, face_data, number_of_time_slices, region):
    iteration = 10
    accuracy_w = np.zeros(number_of_time_slices)
    featuresMat = []
    out = []

    for ts in range(number_of_time_slices):
        manmade_features = manmade_data[:, :, ts].T
        body_features = body_data[:, :, ts].T
        natural_features = natural_data[:, :, ts].T
        face_features = face_data[:, :, ts].T

        manmade_labels = -1 * np.ones((manmade_features.shape[0], 1))
        body_labels = 1 * np.ones((body_features.shape[0], 1))
        natural_labels = -1 * np.ones((natural_features.shape[0], 1))
        face_labels = 1 * np.ones((face_features.shape[0], 1))

        features = np.vstack([manmade_features, body_features, natural_features, face_features])
        labels = np.vstack([manmade_labels, body_labels, natural_labels, face_labels]).ravel()
        featuresMat.append(features)

        print(f"Accuracy for time slice {ts+1}")
        result = gen_fx_get_svm(labels, features, 0.5, iteration)
        out.append(result)
        accuracy_w[ts] = np.mean(result['pt']) * 100

    time = np.linspace(-200, 700, number_of_time_slices)

    plt.figure()
    plt.plot(time, accuracy_w, '-o', linewidth=2)
    plt.plot(time, np.convolve(accuracy_w, np.ones(10)/10, mode='same'), linewidth=2)
    plt.grid(True)
    plt.title(f"Accuracy on rate {region}", fontsize=16, color='black')
    plt.xlabel("time (ms)", fontsize=16, color='blue')
    plt.ylabel("Percentage of Accuracy", fontsize=16, color='red')
    plt.legend(['Accuracy', 'Smooth Accuracy'], fontsize=14)
    plt.savefig(f"Accuracy_animate_inanimate_rate_{region}.png")
    savemat(f"Accuracy_animate_inanimate_rate_{region}.mat", {"accuracy_w": accuracy_w})

    recall_manmade = np.zeros(number_of_time_slices)
    recall_body = np.zeros(number_of_time_slices)

    for ts in range(number_of_time_slices):
        confusion_matrices = out[ts]['C']
        recall_manmade_temp = []
        recall_body_temp = []

        for i in range(iteration):
            cm = confusion_matrices[:, :, i]
            recall_manmade_temp.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
            recall_body_temp.append(cm[1, 1] / (cm[1, 0] + cm[1, 1]))

        recall_manmade[ts] = np.mean(recall_manmade_temp) * 100
        recall_body[ts] = np.mean(recall_body_temp) * 100

    savemat(f"Recall_animate_rate_{region}.mat", {"recall_manmade": recall_manmade})
    savemat(f"Recall_inanimate_rate_{region}.mat", {"recall_body": recall_body})

    plt.figure()
    plt.plot(time, np.convolve(recall_manmade, np.ones(10)/10, mode='same'), linewidth=2)
    plt.plot(time, np.convolve(recall_body, np.ones(10)/10, mode='same'), linewidth=2)
    plt.grid(True)
    plt.title(f"Recall rate in {region}", fontsize=16, color='black')
    plt.xlabel("time (ms)", fontsize=16, color='blue')
    plt.ylabel("Percentage of Recall", fontsize=16, color='red')
    plt.legend(['Recall Inanimate', 'Recall Animate'], fontsize=14)
    plt.savefig(f"Recall_rate_{region}.png")

    return accuracy_w, recall_manmade, recall_body, np.stack(featuresMat, axis=2), labels
