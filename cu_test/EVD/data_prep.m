clear all
%http://cmp.felk.cvut.cz/wbs/
main_name='vin';
dir_name='./tentatives/is_correct/';
tf=dir(dir_name);
k=1;
N=zeros(15,1);
tni=N;
for i=3:length(tf)
    ss=load(strcat(dir_name,tf(i).name));
    N(k)=length(ss);
    tni(k)=sum(ss);
    k=k+1;
end

f=load(strcat('./tentatives/frames/',main_name,'.txt'));
frames1=f(:,1:2)';
frames2=f(:,3:4)';
matches=[1:length(frames1);1:length(frames1)]-1;
save('frames1_vin.txt','frames1','-ASCII');
save('frames2_vin.txt','frames2','-ASCII');
save('matches_vin.txt','matches','-ASCII');
I1=imread('./1/vin.png');
I1=double(I1);
[m,n,k]=size(I1);
if k>1
I1=(I1(:,:,1)+I1(:,:,2)+I1(:,:,3))/3;
end
save('I1_vin.txt','I1','-ASCII');
I2=imread('./2/vin.png');
I2=double(I2);
[m,n,k]=size(I2);
if k>1
I2=(I2(:,:,1)+I2(:,:,2)+I2(:,:,3))/3;
end
save('I2_vin.txt','I2','-ASCII');