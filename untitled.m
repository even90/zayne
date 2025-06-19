%% 绝缘子憎水性等级分类 - 基于边缘密度特征
clc; clear; close all;

%% 1. 设置路径和参数
datasetPath = fullfile('test'); % 数据集路径
classes = {'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6', 'HC7'}; % 7个类别
numClasses = length(classes);

% 边缘检测参数
edgeMethod = 'canny'; % 使用Canny边缘检测
edgeThreshold = 0.1; % 边缘检测阈值

% 特征参数
gridSize = 8; % 将图像划分为8x8的网格

%% 2. 数据准备和特征提取
disp('开始特征提取...');

features = []; % 存储所有特征
labels = []; % 存储所有标签

for i = 1:numClasses
    className = classes{i};
    classPath = fullfile(datasetPath, className);
    imageFiles = dir(fullfile(classPath, '*.jpg')); % 获取所有jpg文件
    
    for j = 1:length(imageFiles)
        % 读取图像
        imgPath = fullfile(classPath, imageFiles(j).name);
        img = imread(imgPath);
        
        % 转换为灰度图像
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        
        % 边缘检测
        edgeImg = edge(grayImg, edgeMethod, edgeThreshold);
        
        % 计算边缘密度特征
        feature = computeEdgeDensity(edgeImg, gridSize);
        
        % 存储特征和标签
        features = [features; feature];
        labels = [labels; i]; % 使用数字表示类别
        
        % 显示进度
        fprintf('处理: %s, 进度: %d/%d\n', className, j, length(imageFiles));
    end
end

disp('特征提取完成!');

%% 3. 数据标准化
features = zscore(features); % Z-score标准化

%% 4. 数据集划分
rng(1); % 设置随机种子保证可重复性
cv = cvpartition(labels, 'HoldOut', 0.3); % 70%训练，30%测试

trainFeatures = features(cv.training,:);
trainLabels = labels(cv.training);
testFeatures = features(cv.test,:);
testLabels = labels(cv.test);

%% 5. 训练分类器
disp('开始训练分类器...');

% 使用支持向量机(SVM)作为分类器
template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
model = fitcecoc(trainFeatures, trainLabels, 'Learners', template);

disp('分类器训练完成!');

%% 6. 模型评估
predictedLabels = predict(model, testFeatures);

% 计算准确率
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
fprintf('测试集准确率: %.2f%%\n', accuracy*100);

% 混淆矩阵
figure;
confusionchart(testLabels, predictedLabels, 'RowSummary', 'row-normalized');
title('混淆矩阵 (标准化行显示)');

%% 7. 辅助函数 - 计算边缘密度特征
function feature = computeEdgeDensity(edgeImg, gridSize)
    [rows, cols] = size(edgeImg);
    
    % 计算每个网格的大小
    gridHeight = floor(rows / gridSize);
    gridWidth = floor(cols / gridSize);
    
    feature = [];
    
    % 遍历每个网格
    for i = 1:gridSize
        for j = 1:gridSize
            % 计算当前网格的边界
            rowStart = (i-1)*gridHeight + 1;
            rowEnd = min(i*gridHeight, rows);
            colStart = (j-1)*gridWidth + 1;
            colEnd = min(j*gridWidth, cols);
            
            % 提取当前网格
            grid = edgeImg(rowStart:rowEnd, colStart:colEnd);
            
            % 计算边缘密度(边缘像素比例)
            density = sum(grid(:)) / numel(grid);
            
            % 添加到特征向量
            feature = [feature, density];
        end
    end
end