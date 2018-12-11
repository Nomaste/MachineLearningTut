function y_altered = yrecode(y)
% This function takes a vector input and returns a matrix
% output, converting each scalar row into a corresponding 
% row vecotr of zeroes with '1' in the scalar value index.
% e.g [1; 0; 7] will output [1 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 1 0 0 0; ]

l = size(y,1);
y_altered = zeros(l,10);
i=1;
for i=1:l;
if (y(i)==0); 
y(i)=y(i)+10; 
end
y_altered(i,y(i)) = 1;
end

end