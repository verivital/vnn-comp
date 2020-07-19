function printResultsACASXu(n1,n2,N1,N2,property_id,method,numCores,safes_num,vts,total_vt,n_safe,n_unsafe,hard)
    s1 = sprintf("\n==================VERIFY PROPERTY P%d==================", property_id);
    if hard==1
        s2 = sprintf('results/P%d-%s_hard.txt',property_id, method);
    else
        s2 = sprintf('results/P%d-%s.txt',property_id, method);
    end
    fid = fopen(s2, 'wt');
    fprintf(s1);
    safes = cell(N1,N2);
    n_timeout=0;
    for i=n1:N1
        for j=n2:N2
            if safes_num(i,j) == 1
                safes{i, j} = 'UNSAT';
                n_safe = n_safe + 1;
            elseif safes_num(i,j) == 0
                safes{i, j} = 'SAT';
                n_unsafe = n_unsafe + 1;
            elseif safes_num(i,j) == 2
                safes{i, j} = 'UNK';
            elseif safes_num(i,j) == 3
                safes{i, j} = 'T_O';
                n_timeout = n_timeout+1;
            end
        end
    end
    total_vt=sum(sum(vts));
    Res = safes;
    VT = vts;
    if hard==1
        filename = 'results/P'+string(property_id)+'_'+string(method)+'_hard.mat';
    else
        filename = 'results/P'+string(property_id)+'_'+string(method)+'.mat';
    end
    save(filename, 'Res', 'VT');

    % print results
    fprintf('\n=====VERIFICATION RESULTS FOR PROPERTY P%d=====', property_id);
    fprintf(fid,'\n=====VERIFICATION RESULTS FOR PROPERTY P%d=====', property_id);
    fprintf('\n Reachability method: %s', method);
    fprintf(fid, '\n Reachability method: %s', method);
    fprintf('\n number of cores: %d', numCores);

    fprintf('\n Network     Safety     Verification Time');
    fprintf(fid,'\n Network     Safety     Verification Time');
    for i=1:N1
        for j=1:N2
            fprintf('\n  N%d_%d        %s          %.3f', i, j, safes{i, j}, vts(i, j));
            fprintf(fid, '\n  N%d_%d        %s          %.3f', i, j, safes{i, j}, vts(i, j));
        end
    end
    fprintf('\nTotal verification time: %.3f', total_vt);
    fprintf(fid, '\nTotal verification time: %.3f', total_vt);
    fprintf('\nNumber of UNSAT: %d/%d', n_safe, N1*N2);
    fprintf(fid,'\nNumber of UNSAT: %d/%d', n_safe, N1*N2);
    fprintf('\nNumber of SAT: %d/%d', n_unsafe, N1*N2);
    fprintf(fid, '\nNumber of SAT: %d/%d', n_unsafe, N1*N2);
    fprintf('\nNumber of TOUT: %d/%d', n_timeout, N1*N2);
    fprintf(fid, '\nNumber of TOUT: %d/%d', n_timeout, N1*N2);
    fprintf('\nNumber of UNKNOWN: %d/%d', N1*N2 - n_safe - n_unsafe - n_timeout, N1*N2);
    fprintf(fid,'\nNumber of unknown: %d/%d', N1*N2 - n_safe - n_unsafe - n_timeout, N1*N2);
    fprintf('\n=======================END=======================');
    fprintf(fid,'\n=======================END=======================');

end