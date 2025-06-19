%% 绝缘子憎水性等级分类 - 基于边缘密度特征
clc; clear; close all;
warning('off', 'images:imshow:mimeType'); % 关闭无关警告

%% 1. 初始化设置
% ================= 用户可修改参数 =================
datasetPath = 'test';          % 数据集路径（相对或绝对路径）
edgeMethod = 'canny';          % 边缘检测方法：'canny'/'sobel'/'prewitt'
edgeThreshold = 0.15;          % 边缘检测阈值(0.05-0.2)
gridSize = 6;                  % 图像划分网格大小(4-10)
trainRatio = 0.7;              % 训练集比例(0-1)
% ================================================

%% 2. 数据准备阶段
disp('=== 数据准备阶段 ===');

% 2.1 验证文件夹结构
classes = {'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6', 'HC7'};
numClasses = length(classes);

% 检查主文件夹是否存在
if ~isfolder(datasetPath)
    error('数据集文件夹不存在: %s', fullfile(pwd, datasetPath));
end

% 检查每个类别文件夹
missingFolders = {};
for i = 1:numClasses
    if ~isfolder(fullfile(datasetPath, classes{i}))
        missingFolders{end+1} = classes{i};
    end
end
if ~isempty(missingFolders)
    error('缺少以下类别文件夹: %s', strjoin(missingFolders, ', '));
end

% 2.2 扫描所有图像文件
filePaths = {};
labels = [];
validClasses = [];

disp('正在扫描图像文件...');
for classIdx = 1:numClasses
    classDir = fullfile(datasetPath, classes{classIdx});
    imgFiles = dir(fullfile(classDir, '*.jpg'));
    
    if isempty(imgFiles)
        warning('类别 %s 中没有找到JPG图像', classes{classIdx});
        continue;
    end
    
    % 添加到有效类别
    validClasses(end+1) = classIdx;
    
    % 记录文件路径和标签
    for j = 1:length(imgFiles)
        filePaths{end+1} = fullfile(classDir, imgFiles(j).name);
        labels(end+1) = classIdx;
    end
    
    fprintf('类别 %s: 找到 %d 个图像\n', classes{classIdx}, length(imgFiles));
end

% 更新有效类别
classes = classes(validClasses);
numClasses = length(classes);
totalImages = length(filePaths);

if totalImages == 0
    error('没有找到任何可用的图像文件！');
end
fprintf('总计: %d 个图像, %d 个有效类别\n', totalImages, numClasses);

%% 3. 特征提取阶段
disp(newline + '=== 特征提取阶段 ===');

% 预分配特征矩阵
features = zeros(totalImages, gridSize * gridSize);
successCount = 0;

% 创建等待条
h = waitbar(0, '正在提取特征...', 'Name', '进度');

for imgIdx = 1:totalImages
    try
        % 读取图像
        img = imread(filePaths{imgIdx});
        
        % 转换为灰度图像
        if ndims(img) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        
        % 边缘检测
        edgeImg = edge(grayImg, edgeMethod, edgeThreshold);
        
        % 计算边缘密度特征
        features(imgIdx, :) = computeEdgeDensity(edgeImg, gridSize);
        successCount = successCount + 1;
        
        % 更新进度条
        if mod(imgIdx, 10) == 0
            waitbar(imgIdx/totalImages, h, ...
                sprintf('进度: %d/%d (%.1f%%)', imgIdx, totalImages, 100*imgIdx/totalImages));
        end
    catch ME
        warning('图像处理失败: %s (错误: %s)', filePaths{imgIdx}, ME.message);
        features(imgIdx, :) = NaN;
    end
end
close(h);

% 移除失败项
validIdx = ~isnan(features(:,1));
features = features(validIdx, :);
labels = labels(validIdx)';
filePaths = filePaths(validIdx);

fprintf('成功提取 %d/%d 个图像的特征\n', sum(validIdx), totalImages);

%% 4. 数据预处理
disp(newline + '=== 数据预处理 ===');

% 标准化特征
features = normalize(features, 'zscore');

% 转换为分类变量
labels = categorical(labels, 1:numClasses, classes);

%% 5. 数据集划分
disp(newline + '=== 数据集划分 ===');
rng(42); % 固定随机种子

% 分层抽样保持类别比例
cv = cvpartition(labels, 'HoldOut', 1-trainRatio);

trainFeatures = features(cv.training, :);
trainLabels = labels(cv.training);
testFeatures = features(cv.test, :);
testLabels = labels(cv.test);

fprintf('训练集: %d 样本\n测试集: %d 样本\n', ...
    length(trainLabels), length(testLabels));

%% 6. 模型训练
disp(newline + '=== 模型训练 ===');

% 设置SVM模板
svmTemplate = templateSVM(...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'KernelScale', 'auto', ...
    'Standardize', false); % 已手动标准化

% 训练多类分类器
model = fitcecoc(...
    trainFeatures, trainLabels, ...
    'Learners', svmTemplate, ...
    'Coding', 'onevsone', ...
    'Verbose', 1);

disp('模型训练完成！');

%% 7. 模型评估
disp(newline + '=== 模型评估 ===');

% 测试集预测
predictedLabels = predict(model, testFeatures);

% 计算准确率
accuracy = mean(predictedLabels == testLabels);
fprintf('整体准确率: %.2f%%\n', accuracy*100);

% 生成混淆矩阵
[confMat, order] = confusionmat(testLabels, predictedLabels);

% 确保order是元胞数组
if ~iscell(order)
    order = cellstr(order); % 转换为元胞数组
end

% 计算各类别指标
classStats = zeros(numClasses, 4); % [准确率, 精确率, 召回率, F1]
for i = 1:numClasses
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;
    
    classStats(i,1) = (TP + TN) / sum(confMat(:)); % 准确率
    classStats(i,2) = TP / (TP + FP);              % 精确率
    classStats(i,3) = TP / (TP + FN);              % 召回率
    classStats(i,4) = 2 * (classStats(i,2)*classStats(i,3)) / ...
                     (classStats(i,2)+classStats(i,3)); % F1
end

% 显示分类报告
disp('分类性能报告:');
fprintf('%-6s %-8s %-8s %-8s %-8s\n', '类别', '准确率', '精确率', '召回率', 'F1');
for i = 1:numClasses
    fprintf('%-6s %.2f%%   %.2f%%   %.2f%%   %.2f%%\n', ...
        order{i}, ...               % 使用元胞数组访问方式
        classStats(i,1)*100, ...
        classStats(i,2)*100, ...
        classStats(i,3)*100, ...
        classStats(i,4)*100);
end

% 绘制混淆矩阵
figure('Position', [100, 100, 800, 600]);
confusionchart(testLabels, predictedLabels, ...
    'Title', '绝缘子憎水性等级分类混淆矩阵', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
set(gca, 'FontSize', 10);
saveas(gcf, 'confusion_matrix.png');
%% 8. 辅助函数
function densityFeatures = computeEdgeDensity(edgeImg, gridSize)
    [h, w] = size(edgeImg);
    densityFeatures = zeros(1, gridSize * gridSize);
    
    % 计算每个网格的尺寸
    cellH = floor(h / gridSize);
    cellW = floor(w / gridSize);
    
    idx = 1;
    for i = 1:gridSize
        for j = 1:gridSize
            % 计算网格边界
            rowStart = (i-1)*cellH + 1;
            rowEnd = min(i*cellH, h);
            colStart = (j-1)*cellW + 1;
            colEnd = min(j*cellW, w);
            
            % 提取网格区域
            cell = edgeImg(rowStart:rowEnd, colStart:colEnd);
            
            % 计算边缘密度
            densityFeatures(idx) = sum(cell(:)) / numel(cell);
            idx = idx + 1;
        end
    end
end