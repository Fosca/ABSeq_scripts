import sys
# import matplotlib; matplotlib.use('agg')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/ABSeq_scripts/')
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/TransitionProbModel/")
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/groupmne/")
sys.path.append("Z:/scripts/TransitionProbModel/")
sys.path.append("Z:/scripts/groupmne/")
sys.path.append("W:/LocalABseq_scripts/TransitionProbModel/")
sys.path.append("W:/LocalABseq_scripts/groupmne/")
sys.path.append("//Users/fosca/Documents/Fosca/Post_doc/Projects/ABSeq/scripts/TransitionProbModel/")
import MarkovModel_Python

path = "/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/Geom_Seq_scripts/Analysis_scripts/packages/"
list_packages = ['jr-tools','autoreject']

for package in list_packages:
    sys.path.append(path+package)