
% ECE 236A - Final Project
% Peter Racioppo & David Martinez

%% MAIN

close all;
clear all;
clc;

% Load data:
load pendigits_tes.mat
load pendigits_tra.mat

% Data dimensions:
% s_train: 7494x1
% y_train: 7494x16
% s_test: 3498x1
% y_test: 3498x16

K = 10; % Number of classes
M = size(y_train,2); % Number of features

% lambda = 3;
% p = poissrnd(lambda, [1,M])/100;
% test_data_c = f_Corrupt(y_test,p,M);

%% Classifier 1, Run Once
tic
obj1 = MyClassifier1(K,M); % MyClassifier1
obj1 = train(obj1,y_train,s_train); % Train
label1 = classify(obj1,y_test); % Classify
acc1 = sum(label1==s_test)/length(s_test); % Percent Accuracy
disp(['Accuracy = ', num2str(round(100*acc1,2)), '%']); % Print accuracy
runtime_Cl = toc;
disp(['Runtime = ', num2str(runtime_Cl), ' seconds']); % Print accuracy

%% Classifier 2, Run Once
tic
obj2 = MyClassifier2(K,M); % MyClassifier2
obj2 = train(obj2,y_train,s_train); % Train
label2 = classify(obj2,y_test); % Classify
acc2 = sum(label2==s_test)/length(s_test); % Percent Accuracy
disp(['Accuracy = ', num2str(round(100*acc2,2)), '%']); % Print accuracy
runtime_C2 = toc;
disp(['Runtime = ', num2str(runtime_C2), ' seconds']); % Print accuracy

%% Dropout probability:
p = poissrnd(5, [1,M])/100;
k = 0:0.01:1;
k = 0.2;
p = k*ones(1,M);

%% Classifier 3v2, Run Once
tic
obj3v2 = MyClassifier3v2(K,M); % MyClassifier3
obj3v2 = train(obj3v2,y_train,s_train); % Train
label3v2 = TestCorrupted1(obj3v2,y_test,p); % Classify
acc3v2 = sum(label3v2==s_test)/length(s_test); % Accuracy
disp(['Accuracy = ', num2str(round(100*acc3v2,2)), '%']); % Print accuracy
runtime_C3v2 = toc;
disp(['Runtime = ', num2str(runtime_C3v2), ' seconds']); % Print accuracy

%% Classifier 4, Run Once
tic
obj4 = MyClassifier4(K,M,p); % MyClassifier4
obj4 = train(obj4,y_train,s_train); % Train
label4 = TestCorrupted2(obj4,y_test,p); % Classify
acc4 = sum(label4==s_test)/length(s_test); % Accuracy
disp(['Accuracy = ', num2str(round(100*acc4,2)), '%']); % Print accuracy
runtime_C4 = toc;
disp(['Runtime = ', num2str(runtime_C4), ' seconds']); % Print accuracy

%% Classifier 3, Run Once
tic
obj3 = MyClassifier3(K,M); % MyClassifier1
obj3 = train(obj3,y_train,s_train); % Train
label3 = TestCorrupted1(obj3,y_test,p); % Classify
acc3 = sum(label3==s_test)/length(s_test); % Percent Accuracy
disp(['Accuracy = ', num2str(round(100*acc3,2)), '%']); % Print accuracy
runtime_C3 = toc;
disp(['Runtime = ', num2str(runtime_C3), ' seconds']); % Print accuracy


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% WARNING FOLLOWING SECTIONS ARE TIME INTENSIVE %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generating Figures for Part I
% num__i=31;
% runtime_C1_his=zeros(1,num__i);
% runtime_C2_his=zeros(1,num__i);
% acc_C1_his=zeros(1,num__i);
% acc_C2_his=zeros(1,num__i);
% obj1 = MyClassifier1(K,M); % MyClassifier1
% obj1.gamma=10.7784878220700; %[10,1,-0.2,30]; %10.7784878220700
% obj1 = train(obj1,y_train,s_train); % Train
% gamma=obj1.gamma;
% 
% for i=1:num__i
%     tic
%     obj1 = MyClassifier1(K,M); % MyClassifier1
%     obj1.gamma=gamma;
%     obj1 = train(obj1,y_train,s_train); % Train
%     obj1 = classify(obj1,y_test); % Classify
%     acc_C1_his(i) = sum(obj1==s_test)/length(s_test); % Percent Accuracy
%     runtime_C1_his(i)=toc;
%     tic
%     obj2 = MyClassifier2(K,M); % MyClassifier2
%     obj2.gamma=gamma;
%     obj2 = train(obj2,y_train,s_train); % Train
%     obj2 = classify(obj2,y_test); % Classify
%     acc_C2_his(i) = sum(obj2==s_test)/length(s_test); % Percent Accuracy
%     runtime_C2_his(i)=toc;
%     
% end
% 
% save Part_1.mat

%% Averages Part 1

% avg_runtime=sum([runtime_C1_his(1:num__i);runtime_C2_his(1:num__i)],2)/num__i
% avg_acc=sum([acc_C1_his(1:num__i);acc_C2_his(1:31)],2)/num__i

%% Plotting Part I
% figure;
% subplot(2,1,1)
% plot(1:31, acc_C1_his(1:31),'k')
% title('Predicition Accuracy Over 31 Iterations')
% xlabel('Testing Iterations')
% ylabel('C1 Accuracy (%)')
% hold on
% yyaxis right
% plot(1:31,acc_C2_his(1:31))
% ylabel('C2 Accuracy (%)')
% legend('Classifer 1','Classifer 2','location','south')
% axis tight
% 
% subplot(2,1,2)
% plot(1:31, runtime_C1_his(1:31),'k')
% title('Runtime Over 31 Iterations')
% xlabel('Testing Iterations')
% ylabel('C1 Runtime (sec)')
% yyaxis right
% plot(1:31,runtime_C2_his(1:31))
% ylabel('C2 Runtime (sec)')
% % legend('Classifer 1','Classifer 2')
% axis tight

%% Generating Figures for Part II C3

% num__i=10;
% prob3=linspace(0,1,num__i);
% acc_C3_his=zeros(1,num__i);
% runtime_C3_his=zeros(1,num__i);
% for i=1:num__i
%     tic
%     obj3 = MyClassifier3(K,M); % MyClassifier1
%     obj3 = train(obj3,y_train,s_train); % Train
%     obj3.p=prob3(i)*ones(1,M);
%     label_pred = TestCorrupted1(obj3,y_test,obj3.p); % Classify
%     acc_C3_his(i) = sum(label_pred==s_test)/length(s_test) % Percent Accuracy
%     runtime_C3_his(i) = toc;
% end
% 
% % save('Classifier3_data.mat','prob3','acc_C3_his','runtime_C3_his')

%% Plot Figure C3 for Part II
% figure;
% subplot(2,1,1)
% plot(prob3(1:10), acc_C3_his(1:10))
% title('Classifier 3 Predicition Accuracy')
% ylabel('Accuracy (%)')
% xlabel('Erasure Probability (%)')
% axis tight
% 
% subplot(2,1,2)
% plot(1:length(acc_C3_his(1:10)), runtime_C3_his(1:10))
% title('Classifier 3 Runtime')
% xlabel('Testing Iterations')
% ylabel('Runtime (sec)')
% axis tight

%% Generating Figures for Part II C3v2

% num__i=10;
% prob3v2=linspace(0,1,num__i);
% acc_C3v2_his=zeros(1,num__i);
% runtime_C3v2_his=zeros(1,num__i);
% for i=1:num__i
%     tic
%     obj3v2 = MyClassifier3v2(K,M); % MyClassifier1
%     obj3v2 = train(obj3v2,y_train,s_train); % Train
%     obj3v2.p=prob3v2(i)*ones(1,M);
%     label_pred = TestCorrupted1(obj3v2,y_test,obj3v2.p); % Classify
%     acc_C3v2_his(i) = sum(label_pred==s_test)/length(s_test) % Percent Accuracy
%     runtime_C3v2_his(i) = toc;
% end

% save('Classifier3v2_data.mat','prob3v2','acc_C3v2_his','runtime_C3v2_his')

%% Plot Figure C3 for Part II
% figure;
% subplot(2,1,1)
% plot(prob3v2(1:10), acc_C3v2_his(1:10))
% title('Classifier 3v2 Predicition Accuracy')
% ylabel('Accuracy (%)')
% xlabel('Erasure Probability (%)')
% axis tight
% 
% subplot(2,1,2)
% plot(1:length(acc_C3v2_his(1:10)), runtime_C3v2_his(1:10))
% title('Classifier 3v2 Runtime')
% xlabel('Testing Iterations')
% ylabel('Runtime (sec)')
% axis tight

%% Generating Figures for Part II C4

% num__i=10;
% prob4=linspace(0,1,num__i);
% acc_C4_his=zeros(1,num__i);
% runtime_C4_his=zeros(1,num__i);
% for i=1:num__i
%     tic
%     obj4 = MyClassifier4(K,M,prob4(i)*ones(1,M)); % MyClassifier1
%     obj4 = train(obj4,y_train,s_train); % Train
%     obj4.p=prob4(i)*ones(1,M);
%     label_pred = TestCorrupted2(obj4,y_test,obj4.p); % Classify
%     acc_C4_his(i) = sum(label_pred==s_test)/length(s_test) % Percent Accuracy
%     runtime_C4_his(i) = toc;
% end

%% Plot Figure C4 for Part II
% figure;
% subplot(2,1,1)
% plot(prob4(1:10), acc_C4_his(1:10))
% title('Classifier 4 Predicition Accuracy')
% ylabel('Accuracy (%)')
% xlabel('Erasure Probability (%)')
% axis tight
% 
% subplot(2,1,2)
% plot(1:length(acc_C4_his(1:10)), runtime_C4_his(1:10))
% title('Classifier 4 Runtime')
% xlabel('Testing Iterations')
% ylabel('Runtime (sec)')
% axis tight

% save('Classifier4_data.mat','prob4','acc_C4_his','runtime_C4_his')

%% Plot all Accuracies and Runtimes Together
% figure;
% subplot(2,1,1)
% plot(prob3(1:10), acc_C3_his(1:10))
% hold on
% plot(prob3v2(1:10), acc_C3v2_his(1:10))
% plot(prob4(1:10), acc_C4_his(1:10))
% title('Classifier Predicition Accuracy')
% ylabel('Accuracy (%)')
% xlabel('Erasure Probability (%)')
% legend('C_3', 'C_{3v2}', 'C_4')
% axis tight
% ylim([0 1])
% yticks([0 .25 .5 .75 .9 1])
% grid on
% 
% subplot(2,1,2)
% plot(1:length(acc_C3_his(1:10)), runtime_C3_his(1:10))
% hold on
% plot(1:length(acc_C3v2_his(1:10)), runtime_C3v2_his(1:10))
% plot(1:length(acc_C4_his(1:10)), runtime_C4_his(1:10))
% title('Classifier Runtime')
% xlabel('Testing Iterations')
% ylabel('Runtime (sec)')
% grid on


%% Generate and Plot Erasure Histogram
% Ones = ones(100,16);
% obj3 = MyClassifier3(K,M);
% obj3.p = Ones*0.25;
% out = f_Corrupt(obj3,Ones);
% sum(sum(isnan(out)))
% x = [0.01, 0.1, 0.25, 0.5, 0.75];
% y = [0.17, 1.58, 3.98, 8.2, 11.86];
% bar(x,y);
% xlabel('Erasure Probability');
% ylabel('Number of Erased Features');
% grid on;
% title('Feature Erasure with 0.01, 0.1, 0.25, 0.5, 0.75 Probability')

%% Save Figures to PNG Files

% figures = [];
% for f = 1:length(findobj('type','figure'))
%       fig = figure(f);
%       print(fig,strcat(pwd, '\html\','Figure ', string(f),'_' , ...
% datestr(now, 'dd_mm_yy-HH_MM'),'.png'),'-dpng')
% end
% close all

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 