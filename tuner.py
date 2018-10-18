import os
import csv
import numpy as np
import subprocess
from collections import OrderedDict

import sys
import fileinput
import shutil
import argparse


def main(schaffers=False, katsuura=False):

    if not schaffers and not katsuura:
        # bentcigar parameters
        parameters_dict = OrderedDict([
            ('ISLANDS_NUMBER', np.linspace(1, 1000, 1, dtype='int')),
            ('POPULATION_SIZE', np.linspace(100, 100, 1, dtype='int')),
            ('CHILDREN_SIZE', np.linspace(100, 800, 1, dtype='int')),
            ('ARITHMETIC_XOVER_N_PARENTS', np.linspace(2, 8, 1, dtype='int')),
            ('MUTATION_PROBABILITY', np.linspace(0.083333, 0.96, 1)),
            ('RECOMB_PROBABILITY', np.linspace(0.9733333, 0.86, 1)),
            ('N_SURVIVORS', np.linspace(30, 100, 1, dtype='int')),
            ('TOURNAMENT_SIZE', np.linspace(20, 10, 1, dtype='int')),
            ('ARITHMETIC_RECOMB_ALPHA', np.linspace(0.11, 0.42, 1)),
            ('MUTATION_A', np.linspace(2.3888, 2.5, 1)),
            ('MUTATION_B', np.linspace(2.1666, 2.5, 1)),
            ('MUTATION_EPSILON', np.linspace(5.52773266332e-06, 5.52773266332e-06, 1)),
            ('MIGRATION_AFTER_EPOCHS', np.linspace(150, 150, 1, dtype='int')),
            ('ELITISM_TO_KEEP', np.linspace(1, 0, 1, dtype='int')),
            ('BLEND_CROSSOVER_ALPHA', np.linspace(0.5, 0.6, 1)),
        ])
    elif katsuura or schaffers:
        parameters_dict = OrderedDict([
            ('ISLANDS_NUMBER', np.linspace(1, 20, 1, dtype='int')),
            ('POPULATION_SIZE', np.linspace(100, 200, 5, dtype='int')),
            ('CHILDREN_SIZE', np.linspace(40, 100, 1, dtype='int')),
            ('ARITHMETIC_XOVER_N_PARENTS', np.linspace(2, 8, 1, dtype='int')),
            ('MUTATION_PROBABILITY', np.linspace(0.98, 0.56, 1)),
            ('RECOMB_PROBABILITY', np.linspace(0.853333, 0.86, 1)),
            ('N_SURVIVORS', np.linspace(100, 100, 1, dtype='int')),
            ('TOURNAMENT_SIZE', np.linspace(3, 10, 1, dtype='int')),
            ('ARITHMETIC_RECOMB_ALPHA', np.linspace(0.11, 0.42, 1)),
            ('MUTATION_A', np.linspace(2.388888, 2.5, 1)),
            ('MUTATION_B', np.linspace(2.1666, 2.5, 1)),
            ('MUTATION_EPSILON', np.linspace(5.52773266332e-06, 5.52773266332e-06, 1)),
            ('MIGRATION_AFTER_EPOCHS', np.linspace(10, 150, 10, dtype='int')),
            ('ELITISM_TO_KEEP', np.linspace(4, 20, 6, dtype='int')),
            ('BLEND_CROSSOVER_ALPHA', np.linspace(0.5, 0.5, 1)),
        ])

    with open('results.csv', 'w') as outfile:
        fieldnames = list(parameters_dict.keys())
        fieldnames.insert(0, 'SCORE')
        spamwriter = csv.writer(outfile, delimiter=';')
        spamwriter.writerow(fieldnames)

    for RECOMB_PROBABILITY in parameters_dict['RECOMB_PROBABILITY']:
        for BLEND_CROSSOVER_ALPHA in parameters_dict['BLEND_CROSSOVER_ALPHA']:
            for ELITISM_TO_KEEP in parameters_dict['ELITISM_TO_KEEP']:
                for MIGRATION_AFTER_EPOCHS in parameters_dict['MIGRATION_AFTER_EPOCHS']:
                    for ISLANDS_NUMBER in parameters_dict['ISLANDS_NUMBER']:
                        for MUTATION_EPSILON in parameters_dict['MUTATION_EPSILON']:
                            for MUTATION_A in parameters_dict['MUTATION_A']:
                                for MUTATION_B in parameters_dict['MUTATION_B']:
                                    for ARITHMETIC_RECOMB_ALPHA in parameters_dict['ARITHMETIC_RECOMB_ALPHA']:
                                        for TOURNAMENT_SIZE in parameters_dict['TOURNAMENT_SIZE']:
                                            for CHILDREN_SIZE in parameters_dict['CHILDREN_SIZE']:
                                                for POPULATION_SIZE in parameters_dict['POPULATION_SIZE']:
                                                    for ARITHMETIC_XOVER_N_PARENTS in parameters_dict['ARITHMETIC_XOVER_N_PARENTS']:
                                                        for MUTATION_PROBABILITY in parameters_dict['MUTATION_PROBABILITY']:
                                                            for N_SURVIVORS in parameters_dict['N_SURVIVORS']:
                                                                parameters = OrderedDict()
                                                                for key in parameters_dict.keys():
                                                                    parameters[key] = locals()[key]
                                                                    if N_SURVIVORS > POPULATION_SIZE:
                                                                        continue
                                                                edit_java_file(parameters)
                                                                compile_and_run_5_times(
                                                                    parameters, schaffers=schaffers, katsuura=katsuura)


def edit_java_file(parameters):
    shutil.copy('player18.template.java', 'player18.java')
    for k, v in parameters.items():
        for i, line in enumerate(fileinput.input('player18.java', inplace=1)):
            sys.stdout.write(line.replace(k + '_TUNERXXX', str(v)))


def compile_and_run_5_times(parameters, schaffers=False, katsuura=False):
    scores = []
    exceptioned = False
    halted_bad_result = False
    function = 'BentCigarFunction'
    if schaffers:
        function = 'SchaffersEvaluation'
    elif katsuura:
        function = 'KatsuuraEvaluation'
    for i in range(5):
        ret = subprocess.check_output('javac -cp contest.jar:commons-math3-3.6.1.jar player18.java ComputedGenotype.java Individual.java Island.java Matrix.java CholeskyDecomposition.java && jar cmf MainClass.txt submission.jar *class org/ && java -jar testrun.jar -submission=player18 -evaluation={} -seed={}'.format(function, i), shell=True)

        score = str(ret).split('Score: ')[1].split('\\n')[0]
        score = float(score)
        scores.append(score)
        print(score)
        if 'halt' in str(ret) and score < 1.0:
            exceptioned = True
            break

        if score < 4.0:
            halted_bad_result = True
            break

    average_score = sum(scores) / len(scores)
    print(average_score, len(scores))

    with open('results.csv', 'a') as outfile:
        spamwriter = csv.writer(outfile, delimiter=';')
        values_list = list(parameters.items())
        values_list = [v[1] for v in values_list]
        values_list.insert(0, average_score)
        if exceptioned:
            values_list.append('EXCEPTIONED')
        if halted_bad_result:
            values_list.append('HALTED_BAD_RESULT')
        spamwriter.writerow(values_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tuner run')
    parser.add_argument('--schaffers', action="store_true", default=False)
    parser.add_argument('--katsuura', action="store_true", default=False)

    args = parser.parse_args()
    main(args.schaffers, args.katsuura)
