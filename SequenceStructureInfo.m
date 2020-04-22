clearvars;clc

%% Compute position-dependent metadata for each sequence
% Ex: [[AA][BB]][AAA]
% Identity: 0 0 1 1 0 0 0
% Repeat0/Alter1: NaN 0 1 0 1 0 0
% ChunkNumber: 1 1 2 2 3 3 3 (/!\ "local" chunk, no hierarchy /!\)
% WithinChunkPosition: 1 2 1 2 1 2 3 (i.e. 1=chunk beginning) (/!\ "local" chunk, no hierarchy /!\)
% WithinChunkPositionReverse: 2 1 2 1 3 2 1 (i.e. 1=chunk ending) (/!\ "local" chunk, no hierarchy /!\)
% ChunkDepth: 2 2 2 2 1 1 1
% OpenedChunks: 2 1 2 0 1 1 0
% ChunkSize: 2 2 2 2 3 3 3
% ChunkBeginning: 1 0 1 0 1 0 0
% ChunkEnd: 0 1 0 1 0 0 1

data = [];
data.Sequence = {
'AAAAAAAAAAAAAAAA';
'ABABABABABABABAB';
'AABBAABBAABBAABB';
'AAAABBBBAAAABBBB';
'AABBABABAABBABAB';
'AAAABBBBAABBABAB';
'ABAAABBBBABBAAAB';};
% LoT_chunk_comp
data.SequenceBrackets = {
'[AAAAAAAAAAAAAAAA]';
'[ABABABABABABABAB]';
'[[AA][BB][AA][BB][AA][BB][AA][BB]]';
'[[AAAA][BBBB][AAAA][BBBB]]';
'[[[[AA][BB]][ABAB]][[[AA][BB]][ABAB]]]';
'[[[AAAA][BBBB]][[AA][BB]][ABAB]]';
'[[A][B][[AAA][BBBB][A][B][B][AAA][B]]';};

for nseq=1:numel(data.Sequence)
    
    seq  = data.Sequence{nseq};
    expr = data.SequenceBrackets{nseq};

    % ======== Identity ======== % // not really used later
    for jj=1:size(seq,2)
        if strcmp(seq(jj), 'A')
            data.Identity.(seq)(jj) = 0;
        elseif strcmp(seq(jj), 'B')
            data.Identity.(seq)(jj) = 1;
        end       
    end 
       
    % ======== Repeat0/Alter1 ======== %    
    data.RepeatAlter.(seq)(1) = NaN;
    for jj=2:size(seq,2)
        if strcmp(seq(jj), seq(jj-1))
            data.RepeatAlter.(seq)(jj) = 0;
        else
            data.RepeatAlter.(seq)(jj) = 1;
        end
    end
     
    % ======== ChunkNumber & WithinChunkPosition ======== %
    % "newexpr": keep only 'minimal' (non hierarchical) chunks
    tmp1 = strfind(expr, '['); torem1 = tmp1(find(diff(tmp1)==1));
    tmp2 = strfind(expr, ']'); torem2 = tmp2(find(diff(tmp2)==1));
    newexpr = expr; newexpr([torem1 torem2]) = [];
    % compute
    chkN = 0;
    chkP = 0;
    posReal = 1; 
    posExpr = 1;
    while posExpr <= size(newexpr,2)
        if strcmp(newexpr(posExpr), '[')
            chkN = chkN + 1;
            chkP = 0;
        elseif strcmp(newexpr(posExpr), ']')
        else
            data.ChunkNumber.(seq)(posReal) = chkN;
            chkP = chkP + 1;
            data.WithinChunkPosition.(seq)(posReal) = chkP;
            posReal = posReal + 1;   
        end
        posExpr = posExpr + 1;
    end
    
    % ======== ChunkSize (from ChunkNumber) ======== %
    % probably a simpler way!
    chkN = data.ChunkNumber.(seq);
    tmp = [];
    for val=unique(chkN)
        tmp(val) = numel(find(chkN == val));
    end
    chkS = []; vals = unique(chkN);
    for pos = 1:numel(chkN)
        chkS(pos) = tmp(find(chkN(pos) == vals));
    end
    data.ChunkSize.(seq) = chkS;
        
    % ======== WithinChunkPositionReverse ======== %
    chkstarts = find(data.WithinChunkPosition.(seq) == 1);
    chksizes = [diff(chkstarts)'; numel(seq)+1-chkstarts(end)];
    tmp = [];
    for jj=1:numel(chksizes)
        tmp = [tmp chksizes(jj):-1:1];
    end
    data.WithinChunkPositionReverse.(seq) = tmp;

    % ======== ChunkDepth ========= %
    chkD = 0;
    posReal = 1; 
    posExpr = 1;
    while posExpr <= size(expr,2)
        if strcmp(expr(posExpr), '[')
            chkD = chkD + 1;
        elseif strcmp(expr(posExpr), ']')
            chkD = chkD - 1;
        else
            data.ChunkDepth.(seq)(posReal) = chkD;
            posReal = posReal + 1;   
        end
        posExpr = posExpr + 1;
    end
    
    % ======== OpenedChunks ========= %
    chkO = 0;
    posReal = 1; 
    posExpr = 1;
    while posExpr <= size(expr,2)
        if strcmp(expr(posExpr), '[')
            chkO = chkO + 1;
        elseif strcmp(expr(posExpr), ']')
            chkO = chkO - 1;
            data.OpenedChunks.(seq)(posReal-1) = data.OpenedChunks.(seq)(posReal-1) - 1;
        else
            data.OpenedChunks.(seq)(posReal) = chkO;
            posReal = posReal + 1;   
        end
        posExpr = posExpr + 1;
    end
    
end

%% Create structures for each subject & variable (to be imported later in Python)

root_path = 'Z:\data\run_info';
subjects_list = {'sub01-pa_190002'; 'sub02-ch_180036'; 'sub03-mr_190273'; 'sub04-rf_190499'; 'sub05-cr_170417'; 'sub06-kc_160388';
                 'sub07-jm_100109'; 'sub08-cc_150418'; 'sub09-ag_170045'; 'sub10-gp_190568'; 'sub11-fr_190151'; 'sub12-lg_170436';
                 'sub13-lq_180242'; 'sub14-js_180232'; 'sub15-ev_070110'; 'sub16-ma_190185'; 'sub17-mt_170249'; 'sub18-eo_190576';
                 'sub19-mg_190180'};
             
for iS = 1 :numel(subjects_list)
    
    subject = subjects_list{iS};
    stim_data_folder = fullfile(root_path, subject);
    cd(stim_data_folder)
    
    Identity = [];
    RepeatAlter = [];
    ChunkNumber = [];
    WithinChunkPosition = [];
    WithinChunkPositionReverse = [];
    ChunkDepth = [];
    OpenedChunks = [];
    ChunkSize = [];
    for nfile = 1 : 14
        file = dir(['info_run' num2str(nfile) '.csv']);
        if ~isempty(file)
            file = fullfile(file.folder, file.name);
            disp(file)
            
            % Import the data
            opts = delimitedTextImportOptions('NumVariables', 2);
            opts.DataLines = [2, Inf];
            opts.Delimiter = ',';
            opts.VariableNames = ["Presented_sequence", "Position_Violation"];
            opts.VariableTypes = ["string", "double"];
            tbl = readtable(file, opts);
            Presented_sequence = tbl.Presented_sequence;
            Position_Violation = tbl.Position_Violation;
            clear opts tbl %% Clear temporary variables

            % Which pattern
            run_sequence = char(strrep(strrep(Presented_sequence(1),'0','A'),'1','B'));
            if strcmp(run_sequence(1), 'B')
                run_sequence = strrep(strrep(strrep(strrep(run_sequence, 'B', '0'),'A', '1'), '0', 'A'),'1', 'B');
            end
            % How many repetitions
            nrep = numel(Presented_sequence);
            
            % Store data for the run (from corresponding sequence in "data" structure)
            Identity(nfile, :) = repmat(char(Presented_sequence(1))-'0',1, nrep);
            RepeatAlter(nfile, :) = repmat(data.RepeatAlter.(run_sequence), 1, nrep);
            ChunkNumber(nfile, :) = repmat(data.ChunkNumber.(run_sequence), 1, nrep);
            WithinChunkPosition(nfile, :) = repmat(data.WithinChunkPosition.(run_sequence), 1, nrep);
            WithinChunkPositionReverse(nfile, :) = repmat(data.WithinChunkPositionReverse.(run_sequence), 1, nrep);
            ChunkDepth(nfile, :) = repmat(data.ChunkDepth.(run_sequence), 1, nrep);
            OpenedChunks(nfile, :) = repmat(data.OpenedChunks.(run_sequence), 1, nrep);
            ChunkSize(nfile, :) = repmat(data.ChunkSize.(run_sequence), 1, nrep);
        end
    end
    Alt = RepeatAlter(:); Alt(isnan(Alt)) = 0
    
    % Save
    save('Identity.mat', 'Identity')
    save('RepeatAlter.mat', 'RepeatAlter')
    save('ChunkNumber.mat', 'ChunkNumber')
    save('WithinChunkPosition.mat', 'WithinChunkPosition')
    save('WithinChunkPositionReverse.mat', 'WithinChunkPositionReverse')
    save('ChunkDepth.mat', 'ChunkDepth')
    save('OpenedChunks.mat', 'OpenedChunks')
    save('ChunkSize.mat', 'ChunkSize')
    
end