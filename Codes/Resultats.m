function Resultats(result,NbExp,Classifieur,mode)
%% ------------------- Affiche les resultats obtenus ---------------------
% Affiche le taux d'erreur
a=mean(result(:,1)~=result(:,2)); disp(['Erreur : ', num2str(a*100), ' %'])

% Affiche la matrice de confusion
[C,~]=confusionmat(result(:,1),result(:,2));
disp('Confusion matrix :    (colonne:predict, ligne:reel)'), disp(C)

% Affiche des exemples d'images mal classifiées
repertoire = [pwd, '\Database\Test\'];
Erreur=result(:,1)~=result(:,2); %colonne binaire, 1 si erreur, 0 sinon
NbErr=sum(Erreur, 1);           % Compte le nombre total d'erreur
IndErr=find(1==Erreur);         % Renvoie l'indice des images mal classees
NbExp=min(NbErr,NbExp);         % Evite les erreurs si NbExp>NbErr ...
ErreurChoisi=randperm(NbErr,NbExp); % Selection des indices (sans doubles)    

% Va afficher NbExp images
for j=1:NbExp
    
    % Va recuperer le chemin des images selectionnees pour etre affichees
    
    % Indice: on sait qu'il y a 100 images par dossiers par classes, on a
    % donc diviser le nombre de l'indice par 100 pour trouver le bon numéro
    % de dossier
    i=IndErr(ErreurChoisi(j));
    dossier = [num2str(floor((i-1)/100)) '\'];
    fichier = ['000' num2str(600+i-(floor((i-1)/100))*100) '.tif'];
    chemin =[repertoire dossier fichier];
    cimg=imread(chemin);
    
    % Affiche l'image avec la prevision et la vrai label
    imshow(cimg,'border','loose','InitialMagnification','fit')
    Reel=['Reel (image) : ' num2str(result(i,1))];
    Prev=['Prevision : ' num2str(result(i,2))];
    title({Prev Reel})
    pause(0.1)

    % enregistre l'image en .png
    name=([Classifieur,mode,num2str(j),'-R',num2str(result(i,1))...
        ,'-P',num2str(result(i,2)),'.png']);
    imwrite(cimg,name)
end

close