% Make a DBN with the following inter-connectivity matrix
%    1
%   /  \
%  2   3
%   \ /
%    4 
%    |
%    5
% where all arcs point down. In addition, there are persistence arcs from each node to itself.
% There are no intra-slice connections.
% Nodes have noisy-or CPDs.
% Node 1 turns on spontaneously due to its leaky source.
% This effect trickles down to the other nodes in the order shown.
% All the other nodes inhibit their leaks.
% None of the nodes inhibit the connection from themselves, so that once they are on, they remain
% on (persistence).
%
% This model was used in the experiments reported in
% - "Learning the structure of DBNs", Friedman, Murphy and Russell, UAI 1998.
% where the structure was learned even in the presence of missing data.
% In that paper, we used the structural EM algorithm.
% Here, we assume full observability and tabular CPDs for the learner, so we can use a much
% simpler learning algorithm.

clc;clear;
warning('off', 'Octave:possible-matlab-short-circuit-operator');
%%%%%%%%%%%%%%%%%%%% get path to BNT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd /home/ppetousis/Desktop/gitRepositories/pomdp-lung-cancer-screening/BNT/
addpath(genpathKPM(pwd))

intraLength = 4;
horizon = 5;
max_fan_in = 10; % let's cheat a little here

%%%%%%%%%%%%%%%%%%%%%%%% import data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = dlmread('/home/ppetousis/Desktop/gitRepositories/pomdp-lung-cancer-screening/Data/DataMultObs/dfDBNObsLearnStructComplete.csv',',',1,0);
%%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ns = [2,8,5,4];
ss = intraLength;%slice size(ss)
T = horizon;

cases = data2cell(data, ss, T,to_replace=0);

%%%%%%%%%%%%%%%%%% learn interslice structure of DBN %%%%%%%%%%%%%%%%%%%%%%%%%%%
inter2 = learn_struct_dbn_reveal(cases, ns, max_fan_in);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% DBN - Naive Bayes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
intra2 = zeros(intraLength);

for i=2:intraLength
  intra2(1,i) = 1; 
  intra2(1,i) = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% DBN creation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bnet = createDBN(intraLength, intra=intra2, inter=inter2, ns, numNodes=8);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% import data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = dlmread('/home/ppetousis/Desktop/gitRepositories/pomdp-lung-cancer-screening/Data/DataMultObs/dfDBNObsLearnStruct.csv',',',1,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ss = intraLength;%slice size(ss)
T = horizon;

cases = data2cell(data, ss, T,to_replace=0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% engine definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% learn dbn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter = 100;
[bnet2, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', max_iter);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%% Set evidence and learn marginals %%%%%%%%%%%%%%%%%%%%%%%%
dataMarginals = dbnInference(cases, ss, T, ncases=30, bnet=bnet2)
