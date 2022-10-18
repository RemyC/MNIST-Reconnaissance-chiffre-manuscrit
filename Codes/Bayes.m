function [result] = Bayes (train_ACP,trainlabels,test_ACP,testlabels)
%% ======================= Entrainement Bayes =======================
% lance le chrono et définit les variables
tic; sigma={size(train_ACP,1)};mu={size(train_ACP,1)};

% Calcul de la covariance et du vecteur moyen
for n=1:size(train_ACP,1)
    % Determiner les indexs des classes de labels
    % Utiliser l index etabli pour recuperer les caracteristiques associees
    T = train_ACP(transpose(trainlabels) == n-1,:);
    sigma{n} = cov(T); % Calcul de sigma et mu
    mu{n} = mean(T);
end

% Affiche le temps d'entrainement
time=toc; disp(['Train time : ', num2str(time),' secondes'])

%% ======================= Test Bayes =======================
% lance le chrono et définit les variables
tic; result = zeros(size(test_ACP,1),2);dist=zeros(size(test_ACP,1),10);

for t=1:size(test_ACP,1)
    for n=1:10
        % parametre n 
        diff = test_ACP(t,:) - mu{n};
        % Calcul de la distance
        dist(t,n) = -(1/2)*log(det(sigma{n}))...
            -(1/2)*diff/(sigma{n})*transpose(diff);  
    end
    % Contient le label associe a la distance maximum
    % Stock le label predit dans la deuxieme colonne de result
    % Et le vrai label de l'image dans la colonne 1
    [~,labelfound] = max(dist(t,:));
    result(t,2) = labelfound-1;
    result(t,1) = testlabels(1,t);
end

% Affiche le temps d'entrainement et la memoire utilisee
time=toc; disp(['Test time : ', num2str(time),' secondes']); memory