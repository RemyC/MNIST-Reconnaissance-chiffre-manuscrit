function [result] = Knn (train_ACP,trainlabels,test_ACP,testlabels,K)
%% =================== Train (déterminer le meilleur K) ==================
% lance le chrono et définit les variables
tic; resultbestK=zeros(); err=zeros();

% Decoupe les donnees : 2/3 entrainement + 1/3 validation
dataknn=train_ACP(1:4000,:);
validknn=train_ACP(4001:6000,:);
dataknnlabels=trainlabels(:,1:4000);
validknnlabels=trainlabels(:,4001:6000);

for i=1:size(validknn,1)
    distance=zeros();
    for j=1:size(dataknn,1)
        % On calcul les distances de chaque individus d'evaluation et tous
        % les individus d'entrainement
        distance(i,j)=dist(validknn(i,:),dataknn(j,:)');
    end 
    disp(['Recherche du meilleur k : ' num2str(round((i-1)/2)/10) ' %'])
    % On trie la matrice des distances en ordre croissant
    % Dans la première colones, on stock le VRAI label
    % Dans les colonnes suivantes, on stock le label obtenu avec le Knn
    [~,I]=sort(distance(i,:));
    resultbestK(i,1)=validknnlabels(1,i); 
    for k = 1:K
        %On recupere l'indice des k plus petites distances dans I(1,1:k),
        %puis les labels des k plus petites distances.
        %On obtient le label le plus fréquent avec la fonction 'mode'.
        %NB : En cas d'égalité, le label le plus petit est gardé !
        resultbestK(i,k+1)=mode(dataknnlabels(1,I(1,1:k))); 
    end
end

% Compare les resultats pour chaque k avec le vrai label (taux d'erreur)
for k=1:K
    err(k)=mean(resultbestK(:,1)~=resultbestK(:,k+1));
end

% On garde le meilleur K
[~,I]=sort(err); bestk=I(1,1); 
% Affiche le temps d'entrainement
time=toc; disp(['Train time : ', num2str(time),' secondes'])
% Affiche la courbe d'evolution de l'erreur en fonction de k
plot(err), pause;

%% ============================ Test(K fixé) ============================
% lance le chrono et définit les variables
tic; result=zeros();

for i=1:size(test_ACP,1)
    distanceK=zeros();
    for j=1:size(train_ACP,1)
        % On calcul les distances de chaque individus de test et tous
        % les individus d'entrainement (train+valid)
        distanceK(i,j)=dist(test_ACP(i,:),train_ACP(j,:)');
    end 
    disp(['Calcul du K-nn : ' num2str(round(i-1)/10) ' %'])
    % On trie la matrice des distances en ordre croissant
    % Dans la première colones, on stock le VRAI label
    % Dans la colonne suivante, on stock le label predict avec le Knn
    [~,I]=sort(distanceK(i,:));     
    result(i,1)=testlabels(1,i); 
    result(i,2)=mode(trainlabels(1,I(1,1:bestk))); 
end

% Affiche le temps d'entrainement et la memoire utilisee
time=toc; disp(['Test time : ', num2str(time),' secondes']); memory