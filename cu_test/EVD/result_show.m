%This program is mainly to show the correspondences
%
clear all
imgdir='.\EVD_dataset\'
im1=strcat(imgdir,'1\adam.png');
im2=strcat(imgdir,'2\adam.png');

fn='adam';
frames_matches='.\';
ind_results='.\csac_results\';
frames1=load(strcat(frames_matches,'frames1_',fn,'.txt'));
frames2=load(strcat(frames_matches,'frames2_',fn,'.txt'));
fp=load(strcat(ind_results,'ind_inliers_',fn,'.txt'))+1;%for c based method need to add 1, for matlab do not add 1.
%fp=fp(1:10);
%f1=frames1(:,fp>0)';
%f2=frames2(:,fp>0)';
f1=frames1(:,fp)';
f2=frames2(:,fp)';

outlier=0;
if outlier
fp_out=load(strcat(ind_results,'ind_outliers_',fn,'.txt'))+1;%for c based method need to add 1, for matlab do not add 1.
f1_out=frames1(:,fp_out)';
f2_out=frames2(:,fp_out)';
f_out=[f1_out f2_out];
end

f=[f1 f2];
I1=imread(im1);
I2=imread(im2);
[M1,N1,K1]=size(I1) ;
[M2,N2,K2]=size(I2) ;
if K1>K2
    I_temp(:,:,1)=I2;
    I_temp(:,:,2)=I2;
    I_temp(:,:,2)=I2;
    I2=I_temp;
end
if K1<K2
    I_temp(:,:,1)=I1;
    I_temp(:,:,2)=I1;
    I_temp(:,:,3)=I1;
    I1=I_temp;
end
[M1,N1,K1]=size(I1) ;
[M2,N2,K2]=size(I2) ;
stack='h' ;
switch stack
  case 'h'
    N3=N1+N2 ;
    M3=max(M1,M2) ;
    oj=N1 ;
    oi=0 ;
  case 'v'
    M3=M1+M2 ;
    N3=max(N1,N2) ;
    oj=0 ;
    oi=M1 ;    
  case 'd'
    M3=M1+M2 ;
    N3=N1+N2 ;
    oj=N1 ;
    oi=M1 ;
  case 'o'
    M3=max(M1,M2) ;
    N3=max(N1,N2) ;
    oj=0;
    oi=0;
end
I=zeros(M3,N3,K1) ;
I(1:M1,1:N1,:) = I1 ;

I(oi+(1:M2),oj+(1:N2),:) = I2 ;
set (gcf,'Position',[1,1,N1+N2,max(M1,M2)], 'color','w')
axes('Position', [0 0 1 1]) ;
xlim([1 M1+M2])
ylim([1 N1+N2])
if K1==3
    imshow(uint8(I))
else
    imagesc(I) ; colormap gray ; hold on ; axis image ; axis off ;
end
drawnow ;
x1=f(:,1);
y1=f(:,2);
x2=f(:,3)+oj;
y2=f(:,4);

x = [ x1  x2]' ;
y = [ y1  y2]' ;
h = line(x, y) ;
hold on
plot(x,y,'or','MarkerFaceColor','red','MarkerEdgeColor','r')
set(h,'LineWidth',2,'Color','g') ;

if outlier
    x1=f_out(:,1);
y1=f_out(:,2);
x2=f_out(:,3)+oj;
y2=f_out(:,4);

x = [ x1  x2]' ;
y = [ y1  y2]' ;
h = line(x, y) ;
hold on
plot(x,y,'or','MarkerFaceColor','red','MarkerEdgeColor','b')
set(h,'LineWidth',2,'Color','r') ;
end

F=getframe(gcf);
imwrite(F.cdata,strcat(ind_results,fn,'.png'))%%%Ïàµ±ÓÚ½ØÆÁ