from matplotlib import pyplot as plt
import numpy as np

run = 0

for method, filename in (('euclidean', 'results_euclidean_distance_19-10-2018-17-00.csv'),
                         ('our_mutation_distance', 'results_our_distance19-10-2018-17-00.csv'),
                         ('our_mutation_distance_optE_', 'results_our_distanceNIGHT19-10-2018-21-00.csv'),
                         ('euclidean_optE_', 'results_euclidean_distanceNIGHT19-10-2018-21-00.csv'),
                         ):

    with open(filename, "r") as f:
        test_number = 0
        best_fitnesses = np.array([])
        medians = np.array([])
        current_best_fitness_row = np.array([])
        current_medians_row = np.array([])
        for line in f:
            if line.startswith("Your code has"):
                if medians.shape[0] == 0:
                    medians = current_medians_row
                    best_fitnesses = current_best_fitness_row
                else:
                    medians = np.vstack((medians, current_medians_row))
                    best_fitnesses = np.vstack((best_fitnesses, current_best_fitness_row))
                current_best_fitness_row = np.array([])
                current_medians_row = np.array([])
                test_number += 1
            if line.startswith("This would") or line.startswith("Score") or line.startswith("Runtime"):
                continue
            if 'ALL' not in line:
                continue
            current_medians_row = np.append(current_medians_row, float(line.split(';')[3]))
            current_best_fitness_row = np.append(current_best_fitness_row, float(line.split(';')[2]))
            # print(line)

        print(medians.shape)
        print(best_fitnesses.shape)
        #np.savetxt("medians.csv", medians, delimiter=";")
        average_mean = medians.mean(axis=0)
        average_best_fitness_row = best_fitnesses.mean(axis=0)
        #plt.plot(average_mean)
        #plt.plot(average_best_fitness_row)
        plt.figure()
        plt.plot(average_mean)
        plt.savefig(method + 'mean_all_runs.png')

        plt.figure()
        plt.plot(average_best_fitness_row)
        plt.savefig(method + 'best_all_runs.png')

        plt.figure()
        for best_fitness_one_run in best_fitnesses:
            plt.plot(best_fitness_one_run)
        plt.savefig(method + 'best_fitnesses_all_runs.png')

        plt.figure()
        for mean_fitness_one_run in medians:
            plt.plot(mean_fitness_one_run)
        plt.savefig(method + 'mean_fitness_one_run.png')

        # plt.show()
        # np.savetxt("median_rows_euclidean.csv", average_mean, ";")
