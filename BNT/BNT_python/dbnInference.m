function dataMarginals = dbnInference(cases, ss, T, ncases, bnet)
          %function that takes as input cases as cells and computes the marginal
          %probability of a node given evidence at each time point P(x_t|y0,y1,.._y_t)
          %returns a matrix of all marginal probabilites of of a binary hidden node
          % 
          % cases are the data, cases are cell structures that contain cell structures
          %for example each cell is a data point and each cell inside that cell
          %is the data for each node of the network over time.
	  % T is the horizon length
          % s is the slize size
          % n cases is the number of cases
          % bnet is the parameterized network

         onodes = [2:4];
         node_i = 1; %cancer node

         dataMarginals = zeros(ncases,T);

         for i=1:ncases
             for t=1:T
             evidence = cases{i}(1:ss,1:t);
            
	     for j=1:t
	         evidence{node_i,j} = [];
	     end
	  
	     engine = smoother_engine(jtree_2TBN_inf_engine(bnet));

	     [engine, ll] = enter_evidence(engine, evidence);
	    
	     marg = marginal_nodes(engine, node_i, t);
	     dataMarginals(i,t) = marg.T(2);
             end
         end

end
