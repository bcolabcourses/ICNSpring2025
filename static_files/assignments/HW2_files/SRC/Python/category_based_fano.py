def category_based_fano(spike_train_all, category, category_labels,
                        mean_vec_cat, var_vec_cat, number_of_neurons,
                        sliding_step, window_length, number_of_time_slices):
    import numpy as np

    for i in range(number_of_neurons):
        cm = spike_train_all[i]['cm']
        spike_data = spike_train_all[i]['data']

        if category == 'face':
            key_data = 'faceData'
            key_count = 'countSpikes_face'
        elif category == 'body':
            key_data = 'BodyData'
            key_count = 'countSpikes_body'
        elif category == 'natural':
            key_data = 'NaturalData'
            key_count = 'countSpikes_natural'
        elif category == 'artifact':
            key_data = 'ArtifactData'
            key_count = 'countSpikes_artifact'
        else:
            key_data = 'nonfaceData'
            key_count = 'countSpikes_nonface'

        data = np.vstack([spike_data[np.where(label == cm)[0], :] for label in category_labels])
        spike_train_all[i][key_data] = data
        spike_counts = np.zeros((data.shape[0], number_of_time_slices))

        for j in range(data.shape[0]):
            for u in range(number_of_time_slices):
                start = sliding_step * u
                end = start + window_length
                temp = data[j, start:end]
                spike_counts[j, u] = np.sum(temp)

        spike_train_all[i][key_count] = spike_counts

        for u in range(number_of_time_slices):
            mean_vec_cat[i, u] = np.mean(spike_counts[:, u])
            var_vec_cat[i, u] = np.var(spike_counts[:, u])

    return mean_vec_cat, var_vec_cat, spike_train_all