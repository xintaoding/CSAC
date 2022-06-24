strI1='USAC_t1_1';
strI2='USAC_t1_2';

I1=imread(strcat(strI1,'.jpg'));
I2=imread(strcat(strI2,'.jpg'));
I1=double(I1);
I2=double(I2);
[m,n,k]=size(I1);
if k>2
    I1=(I1(:,:,1)+I1(:,:,2)+I1(:,:,3))/3;
    I2=(I2(:,:,1)+I2(:,:,2)+I2(:,:,3))/3;
end

I1=I1-min(I1(:)) ;
I1=I1/max(I1(:)) ;
I2=I2-min(I2(:)) ;
I2=I2/max(I2(:)) ;
save(strcat('I1_',strI1(1:end-2),'.txt'),'I1','-ascii');
save(strcat('I2_',strI1(1:end-2),'.txt'),'I2','-ascii');
frames=load(strcat(strI1(1:end-1),'orig_pts.txt'));
frames1_USACgraf=frames(:,1:2)';
frames2_USACgraf=frames(:,3:4)';
save(strcat('frames1_',strI1(1:end-2),'.txt'),'frames1_USACgraf','-ascii');
save(strcat('frames2_',strI1(1:end-2),'.txt'),'frames2_USACgraf','-ascii');

[m,n]=size(frames);
matches_USACgraf=[1:m;1:m];
save(strcat('matches_',strI1(1:end-2),'.txt'),'matches_USACgraf','-ascii');
