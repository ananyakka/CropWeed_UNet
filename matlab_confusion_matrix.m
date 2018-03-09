predict_folder_info = dir('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results');
predict_folder_info(1:2) = [] ;
confusion_matrix= zeros(3,3);

h = waitbar(0, 'Calculating confusion matrix');

for kImageFolder= 1:numel(predict_folder_info)
    
    waitbar(kImageFolder/numel(predict_folder_info));
    image_info = dir(['/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/npydata/Augmented Data/results' '/' predict_folder_info(kImageFolder).name '/*.png']);
    actual_info = dir(['/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Labelled Data/Augmented_train_labels/' predict_folder_info(kImageFolder).name '/*.png']);
    
    if numel(image_info) == numel(actual_info)
        for iImage= 1:numel(image_info)
            predicted_image = imread([image_info(iImage).folder '/' image_info(iImage).name]);
            [~, predicted_label] = max(predicted_image, [], 3);
            
            actual_image = imread([actual_info(iImage).folder '/' actual_info(iImage).name]);
            [~, actual_label] = max(actual_image, [], 3);    
            
            confusion_matrix = confusion_matrix + confusionmat(actual_label(:), predicted_label(:));
            
            
        end
    else
        fprintf('Unequal number of images \n');
        C = setdiff({actual_info.name}, {image_info.name})
    end
end

confusion_matrix = confusion_matrix/sum(confusion_matrix, 2)*100;
close(h)