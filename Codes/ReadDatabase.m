function [train_ACP,trainlabels,test_ACP,testlabels]=ReadDatabase(model)

% ------------------ Lecture des donnees d'entrainement ------------------
if strcmp(model,'Rt')
    load 'ACPFeatures_Rt'
elseif strcmp(model,'Pr')
    load 'ACPFeatures_Pr'
else
    disp('Choisir entre Rt et Pr !'),return
end

% -------------------- Melange les donnees d'entrainement ----------------
A=randperm(size(trainlabels,2));
train_ACP=train_ACP(A,:);
trainlabels=trainlabels(A);