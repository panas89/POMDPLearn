% Make and train a DBN from data
% 
% Input:
% data: 2d array of training data 
% dataTrain: Training data without interpolation
% dataTrainMiss: training data with interpolation and missing data
% max_iter: maximum number of iteration to convergence by the EM algorithm 
% intraLength: integer number of variables
% interLength: integer number dynamic variables
% horizon: integer number of time points
%
%
% Output:
% cell - array of probabilities for each node/variable of the learned DBN model.


function C = dbnMDP_BNT(intraLength, interLength, ns, horizon, data, max_iter)

%%%%%%%%%% clear output & turn off matlab-octave short circuit warnings %%%%%%%%
clc;
warning('off', 'Octave:possible-matlab-short-circuit-operator');
%%%%%%%%%%%%%%%%%%%% get path to BNT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
origPath = pwd;
cd ./BNT
addpath(genpathKPM(pwd))
cd(origPath)
%%%%%%%%%%%%%%%%%%%% define in slice edges %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
intra = zeros(intraLength);

%%%%%%%%%%%%%%%%%%%% define edges between slices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inter = zeros(interLength);
inter(1,1) = 1; 

%%%%%%%%%%%%%%%%%%%%% DBN definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dnodes = 1:intraLength;

bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes);%, 'observed', onodes);

% nodes initialization - rnadomized
for i=1:intraLength+1
  bnet.CPD{i} = tabular_CPD(bnet, i,'CPT','rnd');
end

%%%%%%%%%%%%%%%%%%% engine definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

engine = smoother_engine(jtree_2TBN_inf_engine(bnet));

%%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ss = intraLength;%slice size(ss)
T = horizon;

data(isnan(data)) = -1;

sizeData = size(data);
ncases = sizeData(1);
ncolumns = sizeData(2);

cases = cell(1, ncases);
for i=1:ncases
  cases{i} = cell(ss,T);
  for j=1:ncolumns
    if data(i,j)==-1
      cases{i}{j} = [];
    else
      cases{i}{j} = data(i,j);
    end
  end
end

%%%%%%%%%%%%%%%%%%% learn dbn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[bnet2, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', max_iter);

%%%%%%%%%%%%%%%%%%% return dbn components %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C = {};

for i=1:intraLength+1
  C{i} = struct(bnet2.CPD{i}).CPT;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


