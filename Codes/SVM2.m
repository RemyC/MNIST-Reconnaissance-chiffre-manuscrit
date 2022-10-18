function [result] = SVM2 (train_ACP,trainlabels,test_ACP,testlabels)
%% ======================= SVM CrossValidation =======================
% lance le chrono et définit les variables
tic; nblabels=max(trainlabels); SVMModel=cell(nblabels,1);
FirstModel2=cell(10); Scores=zeros(size(test_ACP,1),nblabels);
lig=4; col=2;c=10^(lig);g=10^(col); % Attention, lig et col hardcodes

% Entrainement des 10 SVM, avec la CrossValidation
for i=0:nblabels %attention a la classe 0
    SVMModel{i+1}=fitcsvm(train_ACP,trainlabels==i,...
        'KernelFunction','rbf','BoxConstraint',c,...
        'ClassNames',[false true],'KernelScale',g);
    CVSVMModel = crossval(SVMModel{i+1});
    FirstModel2{i+1} = CVSVMModel.Trained{1};
end

% On obtient les proba d'appartennace à chaque classe d'une image
% Ici, on reprend bien le modele obtenu avec la CrossValidation FirstModel
for i=0:nblabels
   [~,score]=predict(FirstModel2{i+1},test_ACP) ;
   Scores(:,i+1)=score(:,2); %stock toutes les proba dans Scores
end
% le label predict = # colonne avec le plus grand score
[~,maxScore]=max(Scores,[],2); 
result(:,2)=maxScore.';

% Dans la première colones, on stock le VRAI label
% Dans la colonne suivante, on stock le label predict avec le SVM
result(:,2)=result(:,2)-1;
result(:,1)=testlabels; 

% Affiche le temps d'entrainement et la memoire utilisee
time=toc; disp(['Test time : ', num2str(time),' secondes']); memory