clear all; close all; clc;


x=
if strcmp(all,'Yes')
    [result1] = Bayes (train_ACP,trainlabels,test_ACP,testlabels);
    [result2] = Knn2 (train_ACP,trainlabels,test_ACP,testlabels,K);
    [result3] = SVM (train_ACP,trainlabels,test_ACP,testlabels);
    % Prend le plus frequent des trois. Si égalité, prend la valeur la plus
    % basse
    result(:,2)=mode([result1 result2 result3],2);
    result(:,1)=result1(:,1);
    disp(result)
    
elseif strcmp(Classifieur,'Bayes')
    [result] = Bayes (train_ACP,trainlabels,test_ACP,testlabels);
elseif strcmp(Classifieur,'Knn')
    [result] = Knn (train_ACP,trainlabels,test_ACP,testlabels,K);
elseif strcmp(Classifieur,'Knn2')
    [result] = Knn2 (train_ACP,trainlabels,test_ACP,testlabels,K);
    disp(result)
elseif strcmp(Classifieur,'SVM')
    [result] = SVM (train_ACP,trainlabels,test_ACP,testlabels);
elseif strcmp(Classifieur,'SVM2')
    [result] = SVM2 (train_ACP,trainlabels,test_ACP,testlabels);
else
    disp('Choisir entre Bayes, Knn et SVM !'),return
end