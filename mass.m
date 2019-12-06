% load the model
model = mphload('beam'); %this is the mph file name without the .mph extension
model.study('std1').run;
model.sol('sol1').feature('s1').active(false);
MA = mphmatrix(model,'sol1','out',{'K','Kc','L','Lc','M','Null','Nullf','ud','uscale'});
