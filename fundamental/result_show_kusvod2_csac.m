%This program involved the VLFeat090 in matlab enviroment
%After VLFeat processing, the RANSAC method is used to refine the matched results
name={'booksh'
'box'
'castle'
'corr'
'graff'
'head'
'kampa'
'Kyoto'
'leafs'
'plant'
'rotunda'
'shout'
'valbonne'
'wall'
'wash'
'zoom'};
datas_path='.\';

path_img='cu_test\kusvod2\';
k=0;

pathH='csac_results\';%====================================individual setting

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for i=1:images2_num
for i=1:length(name)
    I1_RGB=imread(strcat(path_img,name{i},'A.jpg'));
    I2_RGB=imread(strcat(path_img,name{i},'B.jpg'));
    nm=name{i};
    fm1=strcat(path_img,'frames1_',nm,'.txt')
    frames1=load(fm1);
    fm2=strcat(path_img,'frames2_',nm,'.txt')
    frames2=load(fm2);
    dir_ind=strcat(pathH,'csac_inliers_',nm,'.txt');
    if isempty(dir(dir_ind))
        continue;
    end
    matches_after_RANSAC=load(strcat(pathH,'csac_inliers_',nm,'.txt'))+1;
%    matches_after_RANSAC=matches_after_RANSAC';
    l=1;
    intersection=[];

    matches_after_RANSAC=[matches_after_RANSAC;matches_after_RANSAC];

    plotmatches(I1_RGB,I2_RGB,frames1(1:2,:),frames2(1:2,:),matches_after_RANSAC);
    F=getframe(gcf);
    imwrite(F.cdata,strcat(pathH,nm,'_csac.png'))%%%Ïàµ±ÓÚ½ØÆÁ
    close(figure(gcf))
end

