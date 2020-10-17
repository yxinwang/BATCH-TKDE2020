close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);
end

db = {'MIRFLICKR','NUSWIDE21'};
loopnbits = [16 32 64];

param.top_K = 2000;

for dbi = 1     :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    
    %% load dataset
    load(['./datasets/',db_name,'_deep.mat'])
    result_name = [result_URL 'deep_' db_name '_result' '.mat'];

    if strcmp(db_name, 'MIRFLICKR')
        R = randperm(size(X,1));
        queryInds = R(1:2000); 
        sampleInds = R(2001:end);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        param.nAnchors = 1000;
        param.use_kmeans = 1;

    elseif strcmp(db_name, 'NUSWIDE21')
        R = randperm(size(X,1));
        queryInds = R(1:2100);
        sampleInds = R(2101:2100+10500);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        param.nAnchors = 1500;
        param.use_kmeans = 0;
    end
    
    clear X Y L PCA_Y R XAll
    
    %% Kernel representation
    param.nXanchors = 500; param.nYanchors = 1000;
    if 1
        anchor_idx = randsample(size(XTrain,1), param.nXanchors);
        XAnchors = XTrain(anchor_idx,:);
        anchor_idx = randsample(size(XTrain,1), param.nYanchors);
        YAnchors = YTrain(anchor_idx,:);
    else
        [~, XAnchors] = litekmeans(XTrain, param.nXanchors, 'MaxIter', 30);
        [~, YAnchors] = litekmeans(YTrain, param.nYanchors, 'MaxIter', 30);
    end
    
    [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,XAnchors);
    [YKTrain,YKTest]=Kernel_Feature(YTrain,YTest,YAnchors);
    
    %% Label Format
    if isvector(LTrain)
        LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
        LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
    end
 
    %% BATCH
    eva_info = cell(1,length(loopnbits));
    
    for ii =1:length(loopnbits)
        fprintf('======start %d bits encoding======\n\n', loopnbits(ii));
        param.nbits = loopnbits(ii);
        BATCHparam = param;
        BATCHparam.eta1 = 0.05; BATCHparam.eta2 = 0.05; BATCHparam.eta0 = 0.9;
        BATCHparam.omega = 0.01; BATCHparam.xi = 0.01; param.max_iter = 6;
        eva_info_ = evaluate_BATCH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);
        eva_info{1,ii} = eva_info_;
        clear eva_info_
    end
    
    %% Results
    for j = 1
        for jj = 1:length(loopnbits)
            % MAP
            Image_VS_Text_MAP{j,jj} = eva_info{j,jj}.Image_VS_Text_MAP;
            Text_VS_Image_MAP{j,jj} = eva_info{j,jj}.Text_VS_Image_MAP;

            % Precision VS Recall
            Image_VS_Text_recall{j,jj,:}    = eva_info{j,jj}.Image_VS_Text_recall';
            Image_VS_Text_precision{j,jj,:} = eva_info{j,jj}.Image_VS_Text_precision';
            Text_VS_Image_recall{j,jj,:}    = eva_info{j,jj}.Text_VS_Image_recall';
            Text_VS_Image_precision{j,jj,:} = eva_info{j,jj}.Text_VS_Image_precision';

            % Top number Precision
            Image_To_Text_Precision{j,jj,:} = eva_info{j,jj}.Image_To_Text_Precision;
            Text_To_Image_Precision{j,jj,:} = eva_info{j,jj}.Text_To_Image_Precision;

            % Time
            trainT{j,jj} = eva_info{j,jj}.trainT;
        end
    end

    save(result_name,'eva_info','BATCHparam','loopnbits','sampleInds','queryInds',...
        'trainT','Image_VS_Text_MAP','Text_VS_Image_MAP','Image_VS_Text_recall','Image_VS_Text_precision',...
        'Text_VS_Image_recall','Text_VS_Image_precision','Image_To_Text_Precision','Text_To_Image_Precision');
end