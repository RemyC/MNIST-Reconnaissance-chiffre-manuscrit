function [result]= Choose (Classifieur,train_ACP,trainlabels,test_ACP,...
    testlabels,K,all)

% ------------------ Choix du classificateur ALL ? -----------------------
if strcmp(all,'Yes')
    % Lance les trois fonctions ci dessous
    [result1] = Bayes (train_ACP,trainlabels,test_ACP,testlabels);
    [result2] = Knn2 (train_ACP,trainlabels,test_ACP,testlabels,K);
    [result3] = SVM (train_ACP,trainlabels,test_ACP,testlabels);
    % Prend le label le plus frequent des trois. 
    % Si égalité, prend la valeur la plus basse.
    result(:,2)=mode([result1 result2 result3],2);
    result(:,1)=result1(:,1);

%  ------------------ Choix du classificateur ----------------------------
elseif strcmp(Classifieur,'Bayes')
    [result] = Bayes (train_ACP,trainlabels,test_ACP,testlabels);
elseif strcmp(Classifieur,'Knn')
    [result] = Knn (train_ACP,trainlabels,test_ACP,testlabels,K);
elseif strcmp(Classifieur,'Knn2')
    [result] = Knn2 (train_ACP,trainlabels,test_ACP,testlabels,K);
elseif strcmp(Classifieur,'SVM')
    [result] = SVM (train_ACP,trainlabels,test_ACP,testlabels);
elseif strcmp(Classifieur,'SVM2')
    [result] = SVM2 (train_ACP,trainlabels,test_ACP,testlabels);
else
    % si le nom du classifieur n'est pas un des 5 ci-dessous alors arret.
    disp('Choisir entre Bayes, Knn, Knn2, SVM et SVM2 !'),return
end
