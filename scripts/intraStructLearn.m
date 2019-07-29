% Learn Structure of BN from data and K2 algorithm
%
% Input:
%
% data: 2d array of (number of cases, temporal features)
% ns: array that depicts number of categories per variable
% max_fan_in: integer number of allowed edges between variables
% intraLength: integer number of variables
%
%1) Learn intra-strucure, using 1st time point static data (K2)
%
% Output:
% dag: intra-structure, 2d array (number of variables, number of variables) 
%
function dag = pomdpDBNObsModel(data, intraLength, max_fan_in, ns)

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

          order = 1:intraLength;
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	  dag = learn_struct_K2(data, ns, order, 'max_fan_in', max_fan_in, 'verbose', 'yes');	
          
end

