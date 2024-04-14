% Load data
load databody

% Define input and output data for the neural network
datainnet = [data1(:,1:3); data2(:,1:3); data3(:,1:3); data4(:,1:3); data5(:,1:3)]'; 
dataoutnet = [repmat([1 0 0 0 0]', 1, size(data1,1)), ...
              repmat([0 1 0 0 0]', 1, size(data2,1)), ...
              repmat([0 0 1 0 0]', 1, size(data3,1)), ...
              repmat([0 0 0 1 0]', 1, size(data4,1)), ...
              repmat([0 0 0 0 1]', 1, size(data5,1))];

% Create and configure the neural network
pocet_neuronov = 15; % Number of neurons in the hidden layer

net = patternnet(pocet_neuronov);

net.divideFcn = 'dividerand'; % Data division function

net.divideParam.trainRatio = 0.8;%trenovacie
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.2;%testovacie

% Set training parameters
net.trainParam.goal = 0; % Termination condition for error
net.trainParam.show = 20; % Frequency of displaying error
net.trainParam.epochs = 10000; % Maximum number of training epochs
net.trainParam.max_fail = 2; % Maximum validation failures

% Train the neural network
net = train(net, datainnet, dataoutnet);

% Display network structure
view(net)

% Simulation of network output for training data
outnetsim = sim(net, datainnet);

% Calculate network error
err = gsubtract(dataoutnet, outnetsim);

% Percentage of incorrectly classified points
c = confusion(dataoutnet, outnetsim);

% Display confusion matrix
figure
plotconfusion(dataoutnet, outnetsim)
% Define imaginary points
x1 = 0.1; y1 = 0.2; z1 = 0.3;
x2 = 0.4; y2 = 0.5; z2 = 0.6;
x3 = 0.7; y3 = 0.8; z3 = 0.9;
x4 = 0.2; y4 = 0.3; z4 = 0.4;
x5 = 0.5; y5 = 0.6; z5 = 0.7;

% Define the 5 new points
new_points = [x1, y1, z1; 
              x2, y2, z2; 
              x3, y3, z3; 
              x4, y4, z4; 
              x5, y5, z5];

% Classify the 5 new points using the trained neural network
output = sim(net, new_points');

% Analyze the output to classify the new points into respective groups
% For each output, find the index of the maximum value, which corresponds to the predicted group
[~, predicted_groups] = max(output);

% Display the classification results
disp('Classification Results:');
disp(predicted_groups);