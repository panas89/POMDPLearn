function bnet = createDBN(intraLength, intra, inter, ns, numNodes)
          %function that creates a dbn and then instantiates random CPT tables
         dnodes = 1:intraLength;
         bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes);

         % nodes initialization - randomized
         for i=1:numNodes
           bnet.CPD{i} = tabular_CPD(bnet, i,'CPT','rnd');
         end
end