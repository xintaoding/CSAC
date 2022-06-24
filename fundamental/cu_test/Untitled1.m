d1=load('errorg2_Satellite.txt');
d2=load('errorg21_Satellite.txt');
d2=d2(d2<1000)
d3=sort(d1);
[h,p]=jbtest(d2',0.5)
hist(d2)