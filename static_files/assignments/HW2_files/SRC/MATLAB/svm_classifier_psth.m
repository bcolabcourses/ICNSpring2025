function [accuracy_w, recall_manmade, recall_body, featuresMat , labels] = svm_classifier_psth(manmade_data, body_data, natural_data, face_data, number_of_time_slices, region)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SVM Classifier on PSTH in 4 Categories
    % Inputs:
    %   - manmade_data: manmade PSTH data (samples x features x time slices)
    %   - body_data: Body PSTH data (samples x features x time slices)
    %   - natural_data: Natural PSTH data (samples x features x time slices)
    %   - face_data: Face PSTH data (samples x features x time slices)
    %   - number_of_time_slices: Number of time slices
    %   - region: Region label for saving results
    % Outputs:
    %   - accuracy_w: Accuracy for each time slice
    %   - recall_manmade: Recall for manmade category across time slices
    %   - recall_body: Recall for body category across time slices
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    iteration = 10; % Number of iterations
    out = struct(); % Initialize output struct
    accuracy_w = zeros(number_of_time_slices, 1); % Initialize accuracy vector

    % Perform SVM decoding for each time slice
    featuresMat =[];
    for ts = 1:number_of_time_slices
        % Prepare features for each category
        manmade_features = transpose(manmade_data(1:end, :, ts));
        body_features = transpose(body_data(1:end, :, ts));
        natural_features = transpose(natural_data(1:end, :, ts));
        face_features = transpose(face_data(1:end, :, ts));

        % Create labels
        manmade_labels = -1 * ones(size(manmade_features, 1), 1);
        body_labels = 1 * ones(size(body_features, 1), 1);
        natural_labels = -1 * ones(size(natural_features, 1), 1);
        face_labels = 1 * ones(size(face_features, 1), 1);

        % Stack features and labels
        features = [manmade_features; body_features; natural_features; face_features];
        labels = [manmade_labels; body_labels; natural_labels; face_labels];
        featuresMat = cat(3,featuresMat,features);
        % Perform SVM decoding
        disp(['Accuracy for time slice ' num2str(ts)]);
        out(ts).out = gen_fx_get_svm(labels, features, 0.5, iteration);
        accuracy_w(ts) = mean(out(ts).out.pt, 1) * 100;
    end

    % Time vector for plotting
    time = linspace(-200, 700, number_of_time_slices);

    % Plot accuracy
    figure();
    plot(time, accuracy_w, '-o', 'LineWidth', 5);
    hold on;
    plot(time, movmean(accuracy_w, 10), 'LineWidth', 5);
    grid on;
    legend({'Smooth Accuracy', 'Accuracy'}, 'FontSize', 16, 'Location', 'best');
    legend('boxoff');
    [t, s] = title(['Accuracy on rate ', region], 'Color', 'black');
    t.FontSize = 16;
    s.FontAngle = 'italic';
    xlabel('time (ms)', 'FontSize', 16, 'Color', 'b');
    ylabel('Percentage of Accuracy', 'FontSize', 16, 'Color', 'r');
    save(['Accuracy_animate_inanimate_rate_', region, '.mat'], 'accuracy_w', '-v7.3');

    % Recall calculations
    recall_manmade = zeros(1, number_of_time_slices);
    recall_body = zeros(1, number_of_time_slices);

    for ts = 1:number_of_time_slices
        confusion_mat = out(ts).out.C;
        recall_manmade_temp = zeros(1, iteration);
        recall_body_temp = zeros(1, iteration);

        for i = 1:iteration
            temp_confusion_matrix = confusion_mat(:, :, i);
            recall_manmade_temp(i) = temp_confusion_matrix(1, 1) / (temp_confusion_matrix(1, 1) + temp_confusion_matrix(1, 2));
            recall_body_temp(i) = temp_confusion_matrix(2, 2) / (temp_confusion_matrix(2, 1) + temp_confusion_matrix(2, 2));
        end

        recall_manmade(ts) = mean(recall_manmade_temp, 2) * 100;
        recall_body(ts) = mean(recall_body_temp, 2) * 100;
    end

    % Save recall data
    save(['Recall_animate_rate_', region, '.mat'], 'recall_manmade', '-v7.3');
    save(['Recall_inanimate_rate_', region, '.mat'], 'recall_body', '-v7.3');
    
    % Plot recall
    figure();
    plot(time, movmean(recall_manmade, 10), 'LineWidth', 5);
    hold on;
    plot(time, movmean(recall_body, 10), 'LineWidth', 5);
    grid on;
    legend({'Recall Inanimate', 'Recall Animate'}, 'FontSize', 16, 'Location', 'best');
    legend('boxoff');
    [t, s] = title(['Recall rate in ', region], 'Color', 'black');
    t.FontSize = 16;
    s.FontAngle = 'italic';
    xlabel('time (ms)', 'FontSize', 16, 'Color', 'b');
    ylabel('Percentage of Recall', 'FontSize', 16, 'Color', 'r');
end
