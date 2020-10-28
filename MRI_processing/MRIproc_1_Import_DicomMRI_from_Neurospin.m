clear all; close all; clc
addpath(pwd);

% Folders (& spm path)
if ispc
    w.serverdicomdir = '\\canif.intra.cea.fr\acquisition\database\Prisma_fit\'; % to copy dicom from
    w.datadir        = 'Z:\data\MRI\';
elseif isunix
    w.serverdicomdir = '/neurospin/acquisition/database/Prisma_fit/';            
    w.datadir        = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/'; 
%     spm_path         = '/i2bm/local/spm12-7219'; addpath(spm_path)
end

w.localdicomdir  = fullfile(w.datadir, 'orig_dicom', filesep);   % to copy dicom to
w.niftidir       = fullfile(w.datadir, 'orig_nifti', filesep);   % to convert dicom to

% Subject info (to find dicom folders, and add subjectID in output folders)
w.date      = {'20190716' '20190528' '20191002' '20191001' '20180601'...
               '20181009' '20170516' '20151207' '20191218' '20191010'...
               '20200130' '20180130' '20191016' '20191217' '20171130'...
               '20190704' '20180515' '20191112' '20191113'};
w.nip       = {'pa190002' 'ch180036' 'mr190273' 'rf190499' 'cr170417'...
               'kc160388' 'jm100109' 'cc150418' 'ag170045' 'gp190568'...
               'fr190151' 'lg170436' 'lq180242' 'js180232' 'ev070110'...
               'ma190185' 'mt170249' 'eo190576' 'mg190180'};
w.subjects  = { 'sub01'    'sub02'    'sub03'    'sub04'    'sub05'...
                'sub06'    'sub07'    'sub08'    'sub09'    'sub10'...
                'sub11'    'sub12'    'sub13'    'sub14'    'sub15'...
                'sub16'    'sub17'    'sub18'    'sub19'};
% spm('defaults','fmri');  
% spm_jobman('initcfg');

%% =================================== %%
%%% LOOP OVER SUBJECTS: RUN FUNCTIONS %%%
%%% ================================= %%%

% Subjects to process:
subs_to_do = 1:numel(w.subjects); %
% Subjects loop
for iS=subs_to_do
   
    %=======================================%
    %+++++++++++++++++++++++++++++++++++++++%
    Do_1_ImportNeurospinDicom(w,iS)
%     Do_2_ConvertDicomToNifti(w,iS)    // TO DO
    %+++++++++++++++++++++++++++++++++++++++%
    %=======================================%
end


%% Copy dicom folders to local disk
function Do_1_ImportNeurospinDicom(w,iS)

% fprintf([' \n \n']);       
% fprintf('=======================================================================\n');
% fprintf(['    ' w.subjects{iS} ': Importing dicom files from neurospin server...\n']);       
% fprintf('=======================================================================\n');
% tic 

flash_img = false;
sub_name = [w.subjects{iS} '-' w.nip{iS}(1:2) '_' w.nip{iS}(3:end)];

% output local dicom folder
sub_localdicomdir = fullfile(w.localdicomdir, sub_name);
if ~exist(sub_localdicomdir, 'dir'); mkdir(sub_localdicomdir); end
out_dir = fullfile(sub_localdicomdir, 'all_imgs');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% find subject dicom directory on the server
sub_serverdicomdir = dir([char(fullfile(w.serverdicomdir, w.date(iS))) filesep w.nip{iS} '*']); 
if isempty(sub_serverdicomdir)
    warning(['############ No folder found for subject ' w.subjects{iS} '-' w.nip{iS} ': "' [char(fullfile(w.serverdicomdir, w.date(iS))) filesep w.nip{iS} '*'] '" !!!']);
end

if ~isempty(sub_serverdicomdir)
    % if two folders for the same subject on the same date, keep one for later
    tmp = [];
    if size(sub_serverdicomdir,1)>1 % 
        tmp = sub_serverdicomdir (2,:);
        sub_serverdicomdir = sub_serverdicomdir (1,:);
    end
    sub_serverdicomdir = fullfile(sub_serverdicomdir.folder, sub_serverdicomdir.name);
    subdirs = dir(sub_serverdicomdir);
    T1foldIDX = find(contains({subdirs.name},'mprage'));
    FlashfoldIDX = find(contains({subdirs.name},'gre-5°-PDW'));

    % copy dicom files
    if ~isempty(T1foldIDX)
        T1fold = fullfile(subdirs(T1foldIDX).folder, subdirs(T1foldIDX).name);
        disp(['Importation of ' T1fold ' (' w.subjects{iS} ')...'])
        filenames=dir(T1fold);
        for ifile=3:length(filenames)
            copyfile(fullfile(filenames(ifile).folder, filenames(ifile).name),out_dir)
        end
    else
        disp(['T1 not found for subject ' sub_name])
    end
    if ~isempty(FlashfoldIDX)
        Flashfold = fullfile(subdirs(FlashfoldIDX).folder, subdirs(FlashfoldIDX).name);
        disp(['Importation of ' Flashfold ' (' w.subjects{iS} ')...'])
        filenames=dir(Flashfold);
        for ifile=3:length(filenames)
            copyfile(fullfile(filenames(ifile).folder, filenames(ifile).name),out_dir)
        end
    else
        disp(['Flash5° not found for subject ' sub_name])
    end

    % For the rare case where we need to collect data from a second folder of the same date
    if ~isempty(tmp)
        sub_serverdicomdir = fullfile(tmp.folder, tmp.name);
        subdirs = dir(sub_serverdicomdir);
        T1foldIDX = find(contains({subdirs.name},'mprage'));
        FlashfoldIDX = find(contains({subdirs.name},'gre-5°-PDW'));
        % copy dicom files
        if ~isempty(T1foldIDX)
            T1fold = fullfile(subdirs(T1foldIDX).folder, subdirs(T1foldIDX).name);
            disp(['Importation of ' T1fold ' (' w.subjects{iS} ')...'])
            filenames=dir(T1fold);
            for ifile=3:length(filenames)
                copyfile(fullfile(filenames(ifile).folder, filenames(ifile).name),out_dir)
            end
        else
            disp(['[2nd folder] T1 not found for subject ' sub_name])
        end
        if ~isempty(FlashfoldIDX)
            Flashfold = fullfile(subdirs(FlashfoldIDX).folder, subdirs(FlashfoldIDX).name);
            disp(['Importation of ' Flashfold ' (' w.subjects{iS} ')...'])
            filenames=dir(Flashfold);
            for ifile=3:length(filenames)
                copyfile(fullfile(filenames(ifile).folder, filenames(ifile).name),out_dir)
            end
        else
            disp(['[2nd folder] Flash5° not found for subject ' sub_name])
        end
    end
end
% fprintf('Importation of dicom files of %s done! (took %d minutes and %d seconds)\n\n', w.subjects{iS}, floor(toc/60), rem(round(toc),60))
    
end

function Do_2_ConvertDicomToNifti
% source_folder = 'Z:\data\MRI\orig_nifti\';
% cd('C:\MatlabTools\') % Move to dcm2niix location. For help: !dcm2niix -h
% 
% % Dicom folder
% sub_MRIdir = dir(fullfile(source_folder, 'sub14-js_180232'));
% sub_dicomdir = fullfile(sub_MRIdir(3).folder, sub_MRIdir(3).name);
% 
% % Run dcm2niix
% % command = ['dcm2niix -b y -f %n_%f -o ' sub_MRIdir(3).folder ' ' sub_dicomdir]; % -x y for cropped
% command = ['dcm2niix -b y -m y -f %n_%d -o ' sub_MRIdir(3).folder ' ' sub_dicomdir];
% system(command)


% -f : filename (%a=antenna (coil) name, %b=basename, %c=comments, %d=description, %e=echo number, %f=folder name, %i=ID of patient, %j=seriesInstanceUID, %k=studyInstanceUID, %m=manufacturer, %n=name of patient, %p=protocol, %r=instance number, %s=series number, %t=time, %u=acquisition number, %v=vendor, %x=study ID; %z=sequence name; default '%n_RUN_1_%f')
end