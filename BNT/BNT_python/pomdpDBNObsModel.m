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
function dataTrainValid = pomdpDBNObsModel(dataTrainComplete, dataTrain, dataValid)

          %%%%%%%%%% clear output & turn off matlab-octave short circuit warnings %%%%%%%%
          %%%%%%%%%% clear output & turn off matlab-octave short circuit warnings %%%%%%%%
          clc;
          warning('off', 'Octave:possible-matlab-short-circuit-operator');
          %%%%%%%%%%%%%%%%%%%% get path to BNT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          origPath = pwd;
          cd ../BNT
          addpath(genpathKPM(pwd))
          cd(origPath)
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          intraLength = 4;
          horizon = 5;
          max_fan_in = 10; % let's cheat a little here
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          ns = [2,8,5,4];
          ss = intraLength;%slice size(ss)
          T = horizon;

          casesTrainComplete = data2cell(dataTrainComplete, ss, T,to_replace=-1);

          %%%%%%%%%%%%%%%%%% learn interslice structure of DBN %%%%%%%%%%%%%%%%%%%%%%%%%%%
          inter2 = learn_struct_dbn_reveal(casesTrainComplete, ns, max_fan_in);
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

          %%%%%%%%%%%%%%%%%%%%%%%% create cases from dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          ss = intraLength;%slice size(ss)
          T = horizon;

          casesTrain = data2cell(dataTrain, ss, T,to_replace=-1);
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%% engine definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%% learn dbn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          max_iter = 100;
          [bnet2, LLtrace] = learn_params_dbn_em(engine, casesTrain, 'max_iter', max_iter);
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          %%%%%%%%%%%%%%%%%%%%%% Set evidence and learn marginals %%%%%%%%%%%%%%%%%%%%%%%%
          dataTrainMarginals = dbnInference(casesTrain, ss, T, ncases=size(dataTrain)(1), bnet=bnet2);


          casesValid = data2cell(dataValid, ss, T,to_replace=-1);
          dataValidMarginals = dbnInference(casesValid, ss, T, ncases=size(dataValid)(1), bnet=bnet2);
          
          dataTrainValid = [dataTrainMarginals; dataValidMarginals];
end

