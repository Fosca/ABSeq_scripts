import argparse
import os

parser = argparse.ArgumentParser(description='This function will kill the jobs that have a number contained between -start and -end')
parser.add_argument('-start', default=None, type=int, help = 'The index of the job at the beginning')
parser.add_argument('-end', default=None, type=int, help = 'The index of the job at the end')

args = parser.parse_args()

for i in range(args.start+1,args.end+1):
    os.system('qdel %i'%i)