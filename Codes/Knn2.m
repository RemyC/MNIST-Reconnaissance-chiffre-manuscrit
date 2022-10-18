function [result] = Knn2 (train_ACP,trainlabels,test_ACP,testlabels,K)
%% ======================= Knn2 Train =======================
% lance le chrono et définit les variables
tic; Knn2={K};err=zeros(K);

% decoupe en 1/3, 2/3
Train=train_ACP(1:4000,:);
Valid=train_ACP(4001:6000,:);
TrainLabels=trainlabels(:,1:4000);
ValidLabels=trainlabels(:,4001:6000);

for k=1:K
    % "Entraine" le Knn pour les k cas à tester
    Knn2{k}=fitcknn(Train,TrainLabels,'NumNeighbors',k);
end

Scores(:,1)=ValidLabels;
for k=1:K
    % On obtient les predictions avec la fonction predict
    % stock toutes les label obtenues dans Scores
    % On calcul l'erreur pour chacun des k
    [label,~]=predict(Knn2{k},Valid) ;
    Scores(:,k+1)=label; 
    err(k)=mean(Scores(:,1)~=Scores(:,k+1));
end

% On garde le meilleur K
[~,I]=sort(err); bestk=I(1,1);
% Affiche le temps d'entrainement
time=toc; disp(['Train time : ', num2str(time),' secondes'])
% Affiche la courbe d'evolution de l'erreur en fonction de k
plot(err), pause;

%% ======================= Test Knn2 =======================
% lance le chrono
tic

% Nouvel entrainement et test avec toutes les donnees et le meilleur k
Knn2bestk=fitcknn(train_ACP,trainlabels,'NumNeighbors',bestk);
[labelk,~]=predict(Knn2bestk,test_ACP);

% On stock les resultats obtenu dans result, 1er = vrai, 2e = predict
result(:,1)=testlabels;
result(:,2)=labelk;

% Affiche le temps d'entrainement et la memoire utilisee
time=toc; disp(['Test time : ', num2str(time),' secondes']); memory