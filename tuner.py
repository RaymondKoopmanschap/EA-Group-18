import os
import csv
import subprocess
from collections import OrderedDict

import sys
import fileinput
import shutil
import argparse
import random

random.seed(3)


def main(schaffers=False, katsuura=False):

    parameters_dict = OrderedDict([
        ('ISLANDS_NUMBER', (1, 20, 'int')),
        ('BLEND_CROSSOVER_ALPHA', (0.0, 1.0, 'float')),
        ('TOURNAMENT_SIZE', (2, 40, 'int')),
        ('POPULATION_SIZE', (10, 1000, 'int')),
    ])

    with open('results.csv', 'w') as outfile:
        fieldnames = list(parameters_dict.keys())
        fieldnames.insert(0, 'SCORE')
        spamwriter = csv.writer(outfile, delimiter=';')
        spamwriter.writerow(fieldnames)

    for i in range(0, 1000):
        parameters = {}
        parameters['ISLANDS_NUMBER'] = random.randint(1, 20)
        parameters['BLEND_CROSSOVER_ALPHA'] = random.random()
        parameters['TOURNAMENT_SIZE'] = random.randint(2, 40)
        parameters['POPULATION_SIZE'] = random.randint(10, 1000)


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
