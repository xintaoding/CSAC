name={'booksh'
'box'
'castle'
'corr'
'graff'
'head'
'kampa'
'Kyoto'
%'leafs'
'plant'
'rotunda'
'shout'
'valbonne'
'wall'
'wash'
'zoom'};

n_pairs=length(name);
recall=zeros(n_pairs,1);
precision=zeros(n_pairs,1);
n_gt=zeros(n_pairs,1);
for i=1:n_pairs
    gt_inliers=load(strcat('cu_test/kusvod2/gt_inliers_c_',name{i},'.txt'));
    n_gt(i)=length(gt_inliers);

        str=strcat('./csac_results/','csac_inliers_',name{i},'.txt');
        if isempty(dir(str))
            continue;
        end
        method_inliers=load(strcat('./csac_results/','csac_inliers_',name{i},'.txt'));
        n_method=length(method_inliers);
        for k=1:n_method
            if sum(method_inliers(k)==gt_inliers)>0
                recall(i)=recall(i)+1;
            end
        end
        if n_method==0|n_gt==0
            precision(i)=0;
            recall(i)=0;
        else
            precision(i)=recall(i)/n_method;
            recall(i)=recall(i)/n_gt(i);
        end


end
[s_recall,idx]=sort(recall);
[s_precision,idx]=sort(precision);