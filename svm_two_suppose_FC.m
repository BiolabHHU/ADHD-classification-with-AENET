function [train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label]=svm_two_suppose_FC(index, data_name)
% Include dependencies
addpath('./lib');                      % dependencies
addpath('./methods');            % FS methods
addpath(genpath('./lib/drtoolbox'));

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe

% Load the data and select features for classification
load([data_name '.mat'])
X_temp = inform.brain_conn;                    % FC's upper triangle matrix without diagonal elements
Y_temp = inform.tag_C;
X = [];

for i=1:max(size(X_temp))    
   temp = X_temp{1,i}';
   X = [temp; X];
end

Y = nominal(ismember(Y_temp,1)); 

[train_h0_data, train_h0_label, test_h0_label] = train_h0(index, X, Y, selection_method, Y_temp);

[train_h0_data, train_h0_label] = energy_normalization(train_h0_data, train_h0_label);

[train_h1_data, train_h1_label, test_h1_label] = train_h1(index, X, Y, selection_method, Y_temp);

[train_h1_data, train_h1_label] = energy_normalization(train_h1_data, train_h1_label);

end

function [train_data_out, train_label_out] = energy_normalization(train_data, train_label)

    tmp = train_data';
    sample_energy_tmp = sqrt(sum(tmp.^2));
    
    agv_energy_1 = mean(sample_energy_tmp(train_label));
    avg_energy_0 = mean(sample_energy_tmp(~train_label));
    sizeoftmp = size(tmp);
    sample_energy_map = ones(1, sizeoftmp(2));
    sample_energy_map(train_label) = agv_energy_1;
    sample_energy_map(~train_label) = avg_energy_0;
    
    energy_map = ones(size(tmp,1),1) * sample_energy_map;
    
    train_data_out = (tmp ./ energy_map)'; 
    train_label_out = train_label;
    
    return
end

function [train_h0_data, train_h0_label,  test_h0_label] = train_h0(index, X, Y, selection_method, Y_temp)

X_train = double(X);
Y_train = (double(Y)-1)*2-1;                       %  labels:  neg_class: -1, pos_class: +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1;           %  labels:   neg_class: -1, pos_class: +1
test_h0_label = double(Y(index));


numF = 50;      %selected FC number . Important parameter!

% feature Selection on training data
    switch lower(selection_method)
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights, subset] = ILFS_auto(X_train, Y_train , 4, 0 );
        case 'mrmr'
            ranking = mRMR(X_train, Y_train, numF);
        
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
        
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); 
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
        
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5; 
            options.nUseEigenfunction = 4;  
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
        
        case 'rfe'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;   
            sup = 1;        
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5; 
            ranking = ECFS( X_train, Y_train, alpha )  ;
        
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 2;
            ranking = UDFS(X_train , nClass ); 
        
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);     
        
        case 'llcfs'   
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
        
        otherwise
            disp('Unknown method.')
    end


    k = numF;     
 
    train_h0_data = X_train(:,ranking(1:k));
    train_h0_label = Y_temp;

    train_h0_data(index,:)=[];
    train_h0_label(index)=[];

    [~,indx_1] = sort(train_h0_label,'descend');
    train_h0_label= train_h0_label(indx_1);
    train_h0_data = train_h0_data(indx_1,:);

    return
end


function [train_h1_data, train_h1_label, test_h1_label] = train_h1(index, X, Y, selection_method, Y_temp)

X_train = double(X);
Y_temp(index) = ~Y_temp(index);
if Y(index) == 'true'
    Y(index,1) = 'false';
else
    Y(index,1) = 'true';
end

Y_train = (double(Y)-1)*2-1;                % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1;    % labels: neg_class -1, pos_class +1
test_h1_label = double(Y(index));

numF = 50;   

% feature Selection on training data
    switch lower(selection_method)
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights, subset] = ILFS_auto(X_train, Y_train , 4, 0 );
        case 'mrmr'
            ranking = mRMR(X_train, Y_train, numF);
        
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
        
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); 
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
        
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5;
            options.nUseEigenfunction = 4;  %You should tune this parameter.
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
        
        case 'rfe'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;   
            sup = 1;      
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5;
            ranking = ECFS( X_train, Y_train, alpha )  ;
        
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 2;
            ranking = UDFS(X_train , nClass ); 
        
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);     
        
        case 'llcfs'   
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
        
        otherwise
            disp('Unknown method.')
    end

    k =numF; 
    
    train_h1_data = X_train(:,ranking(1:k));
    train_h1_label = Y_temp;

    train_h1_data(index,:)=[];
    train_h1_label(index)=[];

    [~,indx_1] = sort(train_h1_label,'descend');
    train_h1_label= train_h1_label(indx_1);
    train_h1_data = train_h1_data(indx_1,:);

    return
end
