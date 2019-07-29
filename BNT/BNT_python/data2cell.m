function cases = data2cell(data, ss, T, to_replace)
          % function that takes an input matrix and transforms into cell2mat
          %input to the BNT dbn learn, inference and structure learn library
          data(isnan(data)) = -1;

          sizeData = size(data);
          ncases = sizeData(1);
          ncolumns = sizeData(2);

          cases = cell(1, ncases);
          for i=1:ncases
            cases{i} = cell(ss,T);
            for j=1:ncolumns
              if data(i,j)==to_replace
                cases{i}{j} = [];
              else
                cases{i}{j} = data(i,j);
              end
            end
          end
end