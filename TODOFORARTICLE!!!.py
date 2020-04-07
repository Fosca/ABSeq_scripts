# Si dans l'article nous devons produire une figure représentant les filtres EMS appliqués aux séquences standard, il faut absolument régler le problème suivant.

# Telle que la fonction dans EMS_funcs generate_EMS_all_sequences est écrite, il peut y avoir des epoques appartenant à un même essai standard dans les train et les test index pour un fold donné.
# On ne veut pas ça car ça revient à tester l'EMS sur des données sur lesquelles il a été entrainé.


# on va splitter en train et test sets séparemment les standard et viol de sorte à ce que les standards provenant d'un meme essai soit dans le même fold

epo_standard = epochs_balanced['ViolationOrNot == 0']
epo_viol = epochs_balanced['ViolationOrNot != 0']

inds = [int(epo_standard.metadata['RunNumber'].values[i] * 1000 + epo_standard.metadata['TrialNumber'].values[i] * 10)
        for i in range(len(epo_standard))]
inds_first = np.diff(inds)
inds_first = [1] + list(inds_first)
inds_first = np.where(inds_first)

epo_pour_folds = epo_standard[inds_first[0]]

y_tmp = [int(
    epo_pour_folds.metadata['SequenceID'].values[i] * 1000 + epo_pour_folds.metadata['StimPosition'].values[i] * 10 +
    epo_pour_folds.metadata['ViolationOrNot'].values[i]) for i in range(len(epo_pour_folds))]

folds_standard = {'train': [], 'test': []}
for train, test in StratifiedKFold(n_splits=4).split(epo_pour_folds.get_data(), y_tmp):
    folds_standard['train'] = train
    folds_standard['test'] = test