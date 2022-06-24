
ExeFilePath='.\x64\Release\cu_test.exe';
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
for i=1:length(name)
Param1=[' ',strcat(name{i},'.txt')];%
%Param1=[' ',name{i}];%
Param2=[' ','15'];
Cmd=[ExeFilePath,Param1];
system(Cmd);
end
abssing=load(strcat('./csac_results/abs_',name{1},'.txt'));
abssun=zeros(length(name),length(abssing));
abssun(1,:)=abssing;
for i=1:length(name)
    abs_fn=strcat('./csac_results/abs_',name{i},'.txt');
    if isempty(dir(abs_fn))
        continue;
    end
    abssun(i,:)=load(abs_fn);
end