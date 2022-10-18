% LABORATOIRE 2
clear all; close all; clc;

%% 1. Charger et melange la base de donn�e
% choisir Rt ou Pr
model='Rt';
[train_ACP,trainlabels,test_ACP,testlabels]=ReadDatabase(model);

%% 2. Envoi le classifieur
% choisir pour Classifieur: Bayes, Knn, Knn2 ou SVM
% choisir k, le nombre maximum de k plus proche voisins test�
% choisir pour all: Yes ou "n'importe quoi"
Classifieur=''; K=25; all='Yes';
[result]= Choose (Classifieur,train_ACP,trainlabels,test_ACP,...
    testlabels,K,all);

%% 3. Resultats
% choisir le nombre d'exemple a afficher
NbExp=10;
Resultats(result,NbExp,Classifieur,model)

