from ABseq_func import *
import config


subject = config.subjects_list[0]
epo = epoching_funcs.load_epochs_items(subject,cleaned=True)

data = epo["SequenceID == 2"]






