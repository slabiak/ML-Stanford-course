m=5000;
num_labels=10;
y2 = zeros(m,num_labels);
for h=1:m
  y2(h,y(h))=1;
endfor