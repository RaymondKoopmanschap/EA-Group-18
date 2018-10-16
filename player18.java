import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;

import java.util.*;
import java.lang.Double;
import java.lang.Math;
import java.io.IOException;
import java.io.FileInputStream;

public class player18 implements ContestSubmission {
    Random rnd_;
    ContestEvaluation evaluation_;
    private int evaluations_limit_;

    private boolean isSchaffers = false;
    
    List<Individual> population;
    public int evals;

    TreeMap<Double, ComputedGenotype> computed_genotypes = new TreeMap<Double, ComputedGenotype>();

    public player18() {
        System.setSecurityManager(null);
        rnd_ = new Random();
    }

    public static void main(String[] args) {
        System.out.println("Hello");
    }

    public void setSeed(long seed) {
        // Set seed of algortihms random process
        rnd_.setSeed(seed);
    }

    public void setEvaluation(ContestEvaluation evaluation) {
        // Set evaluation problem used in the run
        evaluation_ = evaluation;

        // Get evaluation properties
        Properties props = evaluation.getProperties();
        // Get evaluation limit
        evaluations_limit_ = Integer.parseInt(props.getProperty("Evaluations"));
        // Property keys depend on specific evaluation
        // E.g. double param = Double.parseDouble(props.getProperty("property_name"));
        boolean isMultimodal = Boolean.parseBoolean(props.getProperty("Multimodal"));
        boolean hasStructure = Boolean.parseBoolean(props.getProperty("Regular"));
        boolean isSeparable = Boolean.parseBoolean(props.getProperty("Separable"));

        // Do sth with property values, e.g. specify relevant settings of your algorithm
        //System.out.println("IsMultimodal: " + isMultimodal);
        //System.out.println("hasStructure: " + hasStructure);
        //System.out.println("isSeparable: " + isSeparable);
        if (!isMultimodal && !hasStructure && ! isSeparable) {
            //BentCigar
            //isSchaffers = true;
        } else if (isMultimodal && !hasStructure && ! isSeparable) {
            //Katsuura
            isSchaffers = true;
        } else if (isMultimodal && hasStructure && !isSeparable) {
            isSchaffers = true;
            //Schaffers
        }
    }


    public void run() {
        int POPULATION_SIZE = 100;
        int CHILDREN_SIZE = 10;
        double PARENTS_SIZE = 20;
        int ARITHMETIC_XOVER_N_PARENTS = 2;
        double MUTATION_PROBABILITY = 0.18333;
        int N_SURVIVORS = 30;
        int TOURNAMENT_SIZE = 2;
        double ARITHMETIC_RECOMB_ALPHA = 0.51;
        double MUTATION_A = 2.3888;
        double MUTATION_B = 2.1666;
        double MUTATION_EPSILON = 5.52773266332e-06;
        int MIGRATION_AFTER_EPOCHS = 150;
        double RECOMB_PROBABILITY = 0.9733333;

        double BLEND_CROSSOVER_ALPHA = 0.5;

        int ISLANDS_NUMBER = 1;
        int ELITISM_TO_KEEP = 1;

        List<Island> islands = InitializeIslands(ISLANDS_NUMBER, POPULATION_SIZE);

        // calculate fitness
        int epochs = 0;
        while (evals < evaluations_limit_) {
            epochs += 1;

            //TOURNAMENT_SIZE = Math.min(80, 130 - (int) (100* (1-0.95 * evals/evaluations_limit_)));
            if (evals > evaluations_limit_) {
                ELITISM_TO_KEEP = 1;
            }

            for (int island = 0; island < ISLANDS_NUMBER; island++) {
                population = islands.get(island).population;
                setFitnesses(population);

                sortPopulation(population);
                List<Individual> elite = new ArrayList<Individual>();
                for (int j = 0; j < ELITISM_TO_KEEP; j++) {
                    Individual one_elite = new Individual(rnd_, evaluation_, population.get(j).genotype);
                    one_elite.setNDeltas(population.get(j).n_deltas);
                    one_elite.setFitness(population.get(j).fitness);
                    elite.add(one_elite);
                }
                
                // Select parents
                List <Individual> parents = new ArrayList<Individual>();
                for (int i = 0; i < PARENTS_SIZE; i++) {
                    parents.addAll(TournamentSelection(population, TOURNAMENT_SIZE));
                }

                // produce children
                List <Individual> children = new ArrayList<Individual>();
                for (int i = 0; i < CHILDREN_SIZE; i++) {
                    double dice_roll = rnd_.nextDouble();
                    if (dice_roll > RECOMB_PROBABILITY) {
                        children.add(parents.get(rnd_.nextInt(parents.size())));
                        children.add(parents.get(rnd_.nextInt(parents.size())));
                        continue;
                    }
                
                    //children.addAll(WholeArithmeticRecombination(parents, ARITHMETIC_RECOMB_ALPHA));
                    children.addAll(BlendCrossover(parents, BLEND_CROSSOVER_ALPHA));
                    //children.addAll(BlendCrossoverWithCrowding(parents, BLEND_CROSSOVER_ALPHA));
                }

                // mutate children
                for (int i = 0; i < children.size(); i ++) {
                    double dice_roll = rnd_.nextDouble();
                    if (dice_roll < MUTATION_PROBABILITY) {
                        children.get(i).UncorrelatedMutationNStepSizes(MUTATION_EPSILON, MUTATION_A, MUTATION_B);
                        //children.get(i).CorrelatedMutation(MUTATION_EPSILON, MUTATION_A, MUTATION_B);
                        //children.get(i).CorrelatedMutation2(MUTATION_EPSILON, MUTATION_A, MUTATION_B);
                    }
                }
                setFitnesses(children);

                // Select survivors
                List <Individual> survivors = new ArrayList<Individual>();
                for (int i = 0; i < POPULATION_SIZE; i++) {
                    survivors.addAll(TournamentSelection(children, TOURNAMENT_SIZE));
                }
                // elitism
                survivors.addAll(elite);
                setFitnesses(survivors);
                sortPopulation(survivors);

                islands.get(island).incrementGeneration();
                population = survivors;
                islands.get(island).setPopulation(survivors);

                if (epochs % 250 == 0 && island == 1) {
                    //System.out.println(islands.get(island).last_recorded_fitness_changed + " " + islands.get(island).generations_without_fitness_change);
                    //System.out.println(islands.get(island).population.get(0).genotype.get(0) + " 0 " + islands.get(island).last_recorded_fitness_changed);
                    //System.out.println(islands.get(island).population.get(1).genotype.get(0) + " 1 " + island);
                    //System.out.println(islands.get(island).population.get(2).genotype.get(0) + " 3 " + island);
                }

                //System.out.println(islands.get(island).last_recorded_fitness_changed + " " + islands.get(island).generations_without_fitness_change);
                //System.out.println(islands.get(island).population.get(0).genotype + " 0 " + islands.get(island).population.get(0).fitness);
                System.out.println(islands.get(island).population.get(0).fitness);
                //System.out.println(islands.get(island).population.get(1).genotype.get(0) + " 1 " + island);
                //System.out.println(islands.get(island).population.get(2).genotype.get(0) + " 3 " + island);
                //
                /*
                if (epochs % MIGRATION_AFTER_EPOCHS == 0) {
                    IslandMigration(islands.get(island), islands);
                }
                */
                /*
                if (islands.get(island).generations_without_fitness_change > 20 && i % 3 == 0) {
                    //System.out.println("initialized new island");
                    //IslandMigration(islands.get(i), islands);
                    islands.set(i, InitializeRandomIsland(POPULATION_SIZE));
                }
                */
            }
        }
    }

    public List<Individual> InitializeRandomPopulation(int population_size) {
        List<Individual> temp_population = new ArrayList<Individual>(population_size);
        for (int j = 0; j < population_size; j++) {
            temp_population.add(new Individual(rnd_, evaluation_));
        }
        return temp_population;
    }

    public Island InitializeRandomIsland(int population_size) {
        return new Island(InitializeRandomPopulation(population_size));
    }

    public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }   
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public static double euclideanDistance(double[] vectorA, double[] vectorB) {
        double sum = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            sum += ((vectorA[i] - vectorB[i]) * (vectorA[i] - vectorB[i]));
        }   
        return Math.sqrt(sum);
    }


    public List<Individual> StohasticUniversalSampling(List<Individual> population) {
        Collections.sort(population, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });

        // TODO: fix it based on this answer
        // https://stackoverflow.com/questions/22749132/stochastic-universal-sampling-ga-in-python
        double s = 1.5;
        for (int i = 0; i < population.size(); i ++) {
            population.get(i).setRankingProbabilityLinear(
                    (2 - s) / population.size() +
                    (2 * (population.size() - 1 - i) * (s - 1)) / (population.size() * (population.size() - 1))
            );
        }

        List<Individual> parents = new ArrayList<Individual>();
        int total_number_of_parents = population.size();
        int produced_parents = 20;
        int current_member = 0;
        int i = 0;
        double r = rnd_.nextDouble();
        while (current_member < produced_parents) {
            while (r <= population.get(i).ranking_probability_linear) { 
                parents.add(population.get(i));
                r = r + 1.0 / total_number_of_parents;
                if (r > 1) {
                    r = r % 1;
                }
                current_member += 1;
            }
            i = i + 1;
        }
        return parents;
    }

    public List<Individual> TournamentSelection(List<Individual> population, int TOURNAMENT_SIZE) {

        List<Individual> parents = new ArrayList<Individual>();
        List<Integer> added_parents = new ArrayList<Integer>();
        int k = TOURNAMENT_SIZE;
        int produced_parents = population.size();

        List<Individual> competitors = new ArrayList<Individual>();

        for (int i = 0; i < k; i++) {
            competitors.add(population.get(rnd_.nextInt(population.size())));
        }
        List<Integer> competitorsResults = new ArrayList<Integer>(competitors.size());
        for (int i = 0; i < competitors.size(); i++) {
            competitorsResults.add(0);
        }

        for (int i = 0; i < competitors.size(); i++) {
            for (int j = 0; j < competitors.size(); j++) {
                if (i == j) {
                    continue;
                }
                if (competitors.get(i).getFitness() > competitors.get(j).getFitness()) {
                    competitorsResults.set(i, competitorsResults.get(i) + 1);
                } else if (competitors.get(i).getFitness() < competitors.get(j).getFitness()) {
                    competitorsResults.set(j, competitorsResults.get(j) + 1);
                }
            }
        }

        int best_competitor = 0;
        int best_competitor_value = 0;
        //System.out.println(" " + competitorsResults);
        for (int i = 0; i < competitorsResults.size(); i++) {
            //System.out.println(" " + competitorsResults + " " + competitorsResults.get(i) + " " + best_competitor_value + " i" + i);
            if (competitorsResults.get(i) >= best_competitor_value) {
                best_competitor_value = competitorsResults.get(i);
                best_competitor = i;
            }
        }
        //System.out.println(" " + competitorsResults + " " + best_competitor);
        return competitors.subList(best_competitor, best_competitor + 1);
    }

    public List<Individual> OverSelection(List<Individual> population) {

        List<Individual> parents = new ArrayList<Individual>();
        // Koza advice - for population size 1000 -> x = 32%
        //
        double x = 0.10;

        List<Individual> upperHalf = population.subList(0, (int)(x * population.size()));
        List<Individual> lowerHalf = population.subList((int)(x * population.size()), population.size());

        int produced_parents = population.size();
        int current_member = 0;
        while (current_member <= produced_parents) {
            double r = rnd_.nextDouble();
            if (r < 0.2) {
                parents.add(lowerHalf.get(rnd_.nextInt(lowerHalf.size())));
            } else {
                parents.add(upperHalf.get(rnd_.nextInt(upperHalf.size())));
            }
            current_member += 1;
        }

        Collections.sort(parents, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });
        return parents;
    }

    public List<Individual> OnePointCrossover(List<Individual> parents) {

        List<Individual> children = new ArrayList<Individual>(2);

        int firstIndividual = rnd_.nextInt(parents.size());
        int secondIndividual = rnd_.nextInt(parents.size());
        int crossoverPoint = rnd_.nextInt(rnd_.nextInt((7 - 0) + 1) + 1);
        List<Double> childGenotype1 = new ArrayList<Double>(10);
        List<Double> childGenotype2 = new ArrayList<Double>(10);
        List<Double> childNDeltas1 = new ArrayList<Double>(10);
        List<Double> childNDeltas2 = new ArrayList<Double>(10);
        childGenotype1.addAll(parents.get(firstIndividual).genotype.subList(0, crossoverPoint));
        childGenotype1.addAll(parents.get(secondIndividual).genotype.subList(crossoverPoint, 10));

        childGenotype2.addAll(parents.get(secondIndividual).genotype.subList(0, crossoverPoint));
        childGenotype2.addAll(parents.get(firstIndividual).genotype.subList(crossoverPoint, 10));

        //are you supposed to inherit deltas? I guess you are
        childNDeltas1.addAll(parents.get(firstIndividual).n_deltas.subList(0, crossoverPoint));
        childNDeltas1.addAll(parents.get(secondIndividual).n_deltas.subList(crossoverPoint, 10));

        //are you supposed to inherit deltas? I guess you are
        childNDeltas2.addAll(parents.get(secondIndividual).n_deltas.subList(0, crossoverPoint));
        childNDeltas2.addAll(parents.get(firstIndividual).n_deltas.subList(crossoverPoint, 10));
        Individual child_1 = new Individual(rnd_, evaluation_, childGenotype1);
        Individual child_2 = new Individual(rnd_, evaluation_, childGenotype2);

        child_1.setNDeltas(childNDeltas1);
        child_2.setNDeltas(childNDeltas2);

        children.add(child_1);
        children.add(child_2);
        return children;
    }

    public List<Individual> ArithmeticCrossover(List<Individual> parents, int number_of_parents) {
        List<Individual> children = new ArrayList<Individual>(2);
        List<Integer> arithmetic_parents = new ArrayList<Integer>(number_of_parents);

        for (int k = 0; k < number_of_parents; k++) {
            arithmetic_parents.add(rnd_.nextInt(parents.size()));
        }

        List<Double> childGenotype = new ArrayList<Double>(10);
        List<Double> childNDeltas = new ArrayList<Double>(10);
        for (int l = 0; l < 10; l++) {
            double sum = 0.0;
            double deltas_sum = 0.0;
            for (int m = 0; m < arithmetic_parents.size(); m++) {
                sum += parents.get(arithmetic_parents.get(m)).genotype.get(l);
                deltas_sum += parents.get(arithmetic_parents.get(m)).n_deltas.get(l);
            }
            childGenotype.add(sum/number_of_parents);
            childNDeltas.add(sum/number_of_parents);
        }

        Individual child = new Individual(rnd_, evaluation_, childGenotype);
        //child.setNDeltas(childNDeltas);
        children.add(child);
        return children;
    }

    public List<Individual> BlendCrossover(List<Individual> parents, double alpha) {
        // http://www.tomaszgwiazda.com/blendX.htm
        
        List<Individual> children = new ArrayList<Individual>(2);

        int parentIndex_1 = rnd_.nextInt(parents.size());
        int parentIndex_2 = rnd_.nextInt(parents.size());

        List<Double> childGenotype1 = new ArrayList<Double>(10);
        List<Double> childGenotype2 = new ArrayList<Double>(10);
        List<Double> childNDeltas1 = new ArrayList<Double>(10);
        List<Double> childNDeltas2 = new ArrayList<Double>(10);

        for (int i = 0; i < 10; i++) {
            double x_i = parents.get(parentIndex_1).genotype.get(i);
            double y_i = parents.get(parentIndex_2).genotype.get(i);

            double u;
            double gamma;
            double distance;

            double rangeMin;
            double rangeMax;

            distance = Math.abs(x_i - y_i);

            // howto get double in range:
            // https://stackoverflow.com/questions/3680637/generate-a-random-double-in-a-range
            //randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();

            rangeMin = Math.min(x_i, y_i) - alpha * distance;
            rangeMax = Math.min(x_i, y_i) + alpha * distance;
            u = rangeMin + (rangeMax - rangeMin) * rnd_.nextDouble();

            childGenotype1.add(u);

            u = rangeMin + (rangeMax - rangeMin) * rnd_.nextDouble();
            childGenotype2.add(u);

            /*
            childNDeltas1.add(
                    parents.get(parentIndex_1).n_deltas.get(i) * alpha +
                    parents.get(parentIndex_2).n_deltas.get(i) * (1 - alpha)
            );
            childNDeltas2.add(
                    parents.get(parentIndex_2).n_deltas.get(i) * alpha +
                    parents.get(parentIndex_1).n_deltas.get(i) * (1 - alpha)
            );
            */
        }

        Individual child_1 = new Individual(rnd_, evaluation_, childGenotype1);
        Individual child_2 = new Individual(rnd_, evaluation_, childGenotype2);

        //child_1.setNDeltas(childNDeltas1);
        //child_2.setNDeltas(childNDeltas2);

        children.add(child_1);
        children.add(child_2);
        return children;
    }

    public List<Individual> BlendCrossoverWithCrowding(List<Individual> parents, double alpha) {
        
        // http://www.tomaszgwiazda.com/blendX.htm
        List<Individual> children = new ArrayList<Individual>(2);

        int parentIndex_1 = rnd_.nextInt(parents.size());
        int parentIndex_2 = rnd_.nextInt(parents.size());

        Individual parent_1 = parents.get(parentIndex_1);
        Individual parent_2 = parents.get(parentIndex_2);

        List<Double> childGenotype1 = new ArrayList<Double>(10);
        List<Double> childGenotype2 = new ArrayList<Double>(10);
        List<Double> childNDeltas1 = new ArrayList<Double>(10);
        List<Double> childNDeltas2 = new ArrayList<Double>(10);

        for (int i = 0; i < 10; i++) {
            double x_i = parents.get(parentIndex_1).genotype.get(i);
            double y_i = parents.get(parentIndex_2).genotype.get(i);

            double u;
            double gamma;
            double distance;

            double rangeMin;
            double rangeMax;

            distance = Math.abs(x_i - y_i);

            // howto get double in range:
            // https://stackoverflow.com/questions/3680637/generate-a-random-double-in-a-range
            //randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();

            rangeMin = Math.min(x_i, y_i) - alpha * distance;
            rangeMax = Math.min(x_i, y_i) + alpha * distance;
            u = rangeMin + (rangeMax - rangeMin) * rnd_.nextDouble();

            childGenotype1.add(u);

            u = rangeMin + (rangeMax - rangeMin) * rnd_.nextDouble();
            childGenotype2.add(u);

            childNDeltas1.add(
                    parents.get(parentIndex_1).n_deltas.get(i) * alpha +
                    parents.get(parentIndex_2).n_deltas.get(i) * (1 - alpha)
            );
            childNDeltas2.add(
                    parents.get(parentIndex_2).n_deltas.get(i) * alpha +
                    parents.get(parentIndex_1).n_deltas.get(i) * (1 - alpha)
            );
        }

        Individual child_1 = new Individual(rnd_, evaluation_, childGenotype1);
        Individual child_2 = new Individual(rnd_, evaluation_, childGenotype2);

        /*
        List<Double> child1Deltas = new ArrayList<Double>(10);
        List<Double> child2Deltas = new ArrayList<Double>(10);
        for (int i = 0; i < 10; i++) {
            child1Deltas.add(1.0);
            child2Deltas.add(1.0);
        }
        */
        child_1.setNDeltas(childNDeltas1);
        child_2.setNDeltas(childNDeltas2);
        //children.add(child_1);
        //children.add(child_2);
        double euclideanDistance11 = euclideanDistance(
                parent_1.getGenotypeArray(), child_1.getGenotypeArray());
        double euclideanDistance12 = euclideanDistance(
                parent_1.getGenotypeArray(), child_2.getGenotypeArray());
        double euclideanDistance21 = euclideanDistance(
                parent_2.getGenotypeArray(), child_1.getGenotypeArray());
        double euclideanDistance22 = euclideanDistance(
                parent_2.getGenotypeArray(), child_2.getGenotypeArray());


        //System.out.println(child_1.genotype);
        //setFitnesses(children);
        child_1.setFitness((Double) evaluation_.evaluate(child_1.getGenotypeArray()));
        child_2.setFitness((Double) evaluation_.evaluate(child_2.getGenotypeArray()));
        evals += 2;

        if(parent_1.fitness == null) {
            parent_1.setFitness((Double) evaluation_.evaluate(parent_1.getGenotypeArray()));
            evals += 1;
        }
        if(parent_2.fitness == null) {
            parent_2.setFitness((Double) evaluation_.evaluate(parent_2.getGenotypeArray()));
            evals += 1;
        }

        if (euclideanDistance11 + euclideanDistance22 < euclideanDistance12 + euclideanDistance21) {
            if (child_1.fitness > parent_1.fitness) {
                System.out.println("child won1");
                children.add(child_1);
            } else {
                children.add(parent_1);
            }
            if (child_2.fitness > parent_2.fitness) {
                System.out.println("child won2");
                children.add(child_2);
            } else {
                children.add(parent_2);
            }
        } else {
            if (child_1.fitness > parent_2.fitness) {
                System.out.println("child won3");
                children.add(child_1);
            } else {
                children.add(parent_2);
            }
            if (child_2.fitness > parent_1.fitness) {
                System.out.println("child won4");
                children.add(child_2);
            } else {
                children.add(parent_1);
            }
        }

        return children;
    }

    public List<Individual> ArithmeticCrossoverWithCrossover(List<Individual> parents, int number_of_parents) {
        List<Individual> children = new ArrayList<Individual>(2);
        List<Integer> arithmetic_parents = new ArrayList<Integer>(number_of_parents);

        for (int k = 0; k < number_of_parents; k++) {
            arithmetic_parents.add(rnd_.nextInt(parents.size()));
        }

        List<Double> childGenotype = new ArrayList<Double>(10);
        List<Double> childNDeltas = new ArrayList<Double>(10);
        for (int l = 0; l < 10; l++) {
            double sum = 0.0;
            double deltas_sum = 0.0;
            for (int m = 0; m < arithmetic_parents.size(); m++) {
                sum += parents.get(arithmetic_parents.get(m)).genotype.get(l);
                deltas_sum += parents.get(arithmetic_parents.get(m)).n_deltas.get(l);
            }
            childGenotype.add(sum/number_of_parents);
            childNDeltas.add(sum/number_of_parents);
        }

        Individual child = new Individual(rnd_, evaluation_, childGenotype);
        //child.setNDeltas(childNDeltas);
        children.add(child);
        return children;
    }

    public List<Individual> WholeArithmeticRecombination(List<Individual> parents, double alpha) {
        // child1 = alpha * x + (1 - alpha) * y
        // child2 = alpha * y + (1 - alpha) * x

        
        
        List<Individual> children = new ArrayList<Individual>(2);

        int parentIndex_1 = rnd_.nextInt(parents.size());
        int parentIndex_2 = rnd_.nextInt(parents.size());

        List<Double> childGenotype1 = new ArrayList<Double>(10);
        List<Double> childGenotype2 = new ArrayList<Double>(10);
        List<Double> childNDeltas1 = new ArrayList<Double>(10);
        List<Double> childNDeltas2 = new ArrayList<Double>(10);

        for (int i = 0; i < 10; i++) {
            childGenotype1.add(
                    parents.get(parentIndex_1).genotype.get(i) * alpha +
                    parents.get(parentIndex_2).genotype.get(i) * (1 - alpha)
            );
            childGenotype2.add(
                    parents.get(parentIndex_2).genotype.get(i) * alpha +
                    parents.get(parentIndex_1).genotype.get(i) * (1 - alpha)
            );

            childNDeltas1.add(
                    parents.get(parentIndex_1).n_deltas.get(i) * alpha +
                    parents.get(parentIndex_2).n_deltas.get(i) * (1 - alpha)
            );
            childNDeltas2.add(
                    parents.get(parentIndex_2).n_deltas.get(i) * alpha +
                    parents.get(parentIndex_1).n_deltas.get(i) * (1 - alpha)
            );

        }

        Individual child_1 = new Individual(rnd_, evaluation_, childGenotype1);
        Individual child_2 = new Individual(rnd_, evaluation_, childGenotype2);

        //child_1.setNDeltas(childNDeltas1);
        //child_2.setNDeltas(childNDeltas2);

        children.add(child_1);
        children.add(child_2);
        return children;
    }

    public void IslandMigration(Island receiverIsland, List<Island> islands) {
        /* exchange island inhabitants */
        for (int x = 0; x < 10; x++) {
            int randomIsland_2 = rnd_.nextInt((islands.size() -1 - 0) + 1) + 0;
            if (receiverIsland.last_recorded_fitness_changed.equals(islands.get(randomIsland_2).last_recorded_fitness_changed)) {
                continue;
            }
            int randomIndividual_receiver = rnd_.nextInt(((receiverIsland.population.size()) -1 - 0) + 1) + 0;
            int randomIndividual_2 = rnd_.nextInt(((islands.get(randomIsland_2).population.size()) -1 - 0) + 1) + 0;

            receiverIsland.population.set(randomIndividual_receiver, islands.get(randomIsland_2).population.get(randomIndividual_2));
        }
    }

    public List<Individual> mi_plus_lambda_fitness_based(List<Individual> parents, List<Individual> children, int population_size) {
        List<Individual> survivors = new ArrayList<Individual>(0);
        Collections.sort(children, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });
        children.addAll(parents.subList(0, 30));
        List<Individual> new_children = StohasticUniversalSampling(children);
        Collections.sort(children, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });
        return children.subList(0, population_size);
    }


    public void setFitnesses(List<Individual> children) {
        double epsilon = 0.0000000000000003;
        for (Individual individual: children) {
            // Check fitness of unknown fuction
            if (individual.getFitness() == null) {
                /*
                boolean same_genotype_found = false;
                for (Map.Entry<Double, ComputedGenotype> e : computed_genotypes.subMap(individual.genotype.get(0) - epsilon, individual.genotype.get(0) + epsilon).entrySet()) {
                    if (twoGenotypesEqual(individual.genotype, e.getValue().genotype)) {
                        //System.out.println("skipped computing fitness for already known genotype");
                        individual.setFitness(e.getValue().fitness);
                        same_genotype_found = true;
                    }
                }
                if (same_genotype_found) {
                    continue;
                }
                */
                if (evals >= evaluations_limit_) {
                    return;
                }
                evals++;
                Double fitness = (Double) evaluation_.evaluate(individual.getGenotypeArray());
                /*
                computed_genotypes.put(
                        individual.genotype.get(0),
                        new ComputedGenotype(individual.genotype, fitness)
                );
                */
                individual.setFitness(fitness);
            }
        }
    }

    public boolean twoGenotypesEqual(List<Double> genotype1, List<Double> genotype2) {
        for (int i = 0; i < genotype1.size(); i ++) {
            if (!genotype1.get(i).equals(genotype2.get(i)))
                return false;
        }
        return true;
    }

    public void sortPopulation(List<Individual> population) {
        Collections.sort(population, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });
    }
    public List<Island> InitializeIslands(int number_of_islands, int population_size) {
        List<Island> islands = new ArrayList<Island>();
        for (int i = 0; i < number_of_islands; i++) {
            islands.add(InitializeRandomIsland(population_size));
        }
        for (int i = 0; i < number_of_islands; i++) {
            setFitnesses(islands.get(i).population);
        }
        return islands;
    }

    /*
    public List<Individual> DeterministicCrowding(List<Individual> population) {
        // deterministic crowding
        List<Individual> parents = new ArrayList<Individual>();
        for (int i = 0; i < population.size() * 0.01; i++) {
            int firstIndividual = rnd_.nextInt(population.size());
            int secondIndividual = rnd_.nextInt(population.size());
            Individual firstParent = population.get(firstIndividual);
            Individual secondParent = population.get(secondIndividual);
            if(firstParent.fitness == null) {
                firstParent.setFitness((Double) evaluation_.evaluate(firstParent.getGenotypeArray()));
            }
            if(secondParent.fitness == null) {
                secondParent.setFitness((Double) evaluation_.evaluate(secondParent.getGenotypeArray()));
            }

            int crossoverPoint = rnd_.nextInt(rnd_.nextInt((7 - 0) + 1) + 1);
            List<Double> childGenotype1 = new ArrayList<Double>(10);
            List<Double> childGenotype2 = new ArrayList<Double>(10);
            List<Double> childNDeltas1 = new ArrayList<Double>(10);
            List<Double> childNDeltas2 = new ArrayList<Double>(10);
            childGenotype1.addAll(parents.get(firstIndividual).genotype.subList(0, crossoverPoint));
            childGenotype1.addAll(parents.get(secondIndividual).genotype.subList(crossoverPoint, 10));

            childGenotype2.addAll(parents.get(secondIndividual).genotype.subList(0, crossoverPoint));
            childGenotype2.addAll(parents.get(firstIndividual).genotype.subList(crossoverPoint, 10));

            //are you supposed to inherit deltas? I guess you are
            childNDeltas1.addAll(parents.get(firstIndividual).n_deltas.subList(0, crossoverPoint));
            childNDeltas1.addAll(parents.get(secondIndividual).n_deltas.subList(crossoverPoint, 10));

            //are you supposed to inherit deltas? I guess you are
            childNDeltas2.addAll(parents.get(secondIndividual).n_deltas.subList(0, crossoverPoint));
            childNDeltas2.addAll(parents.get(firstIndividual).n_deltas.subList(crossoverPoint, 10));

            Individual child_1 = new Individual(rnd_, evaluation_, childGenotype1);
            Individual child_2 = new Individual(rnd_, evaluation_, childGenotype2);

            child_1.setNDeltas(childNDeltas1);
            child_2.setNDeltas(childNDeltas2);

            child_1.UncorrelatedMutationNStepSizes(0.00000001, 2.0, 2.0);
            child_2.UncorrelatedMutationNStepSizes(0.00000001, 2.0, 2.0);

            child_1.setFitness((Double) evaluation_.evaluate(child_1.getGenotypeArray()));
            child_2.setFitness((Double) evaluation_.evaluate(child_2.getGenotypeArray()));
            evals += 2;

            // calc distance between vectors in genotype space
            double euclideanDistance11 = euclideanDistance(
                    firstParent.getGenotypeArray(), child_1.getGenotypeArray());
            double euclideanDistance12 = euclideanDistance(
                    firstParent.getGenotypeArray(), child_2.getGenotypeArray());
            double euclideanDistance21 = euclideanDistance(
                    secondParent.getGenotypeArray(), child_1.getGenotypeArray());
            double euclideanDistance22 = euclideanDistance(
                    secondParent.getGenotypeArray(), child_2.getGenotypeArray());

            if (euclideanDistance11 + euclideanDistance22 < euclideanDistance12 + euclideanDistance21) {
                if (child_1.fitness > firstParent.fitness) {
                    firstParent.setGenotype(child_1.genotype);
                    firstParent.setNDeltas(child_1.n_deltas);
                    firstParent.setFitness(child_1.fitness);
                }
                if (child_2.fitness > secondParent.fitness) {
                    secondParent.setGenotype(child_2.genotype);
                    secondParent.setNDeltas(child_2.n_deltas);
                    secondParent.setFitness(child_2.fitness);
                }
            } else {
                if (child_1.fitness > secondParent.fitness) {
                    secondParent.setGenotype(child_1.genotype);
                    secondParent.setNDeltas(child_1.n_deltas);
                    secondParent.setFitness(child_1.fitness);
                }
                if (child_2.fitness > firstParent.fitness) {
                    firstParent.setGenotype(child_2.genotype);
                    firstParent.setNDeltas(child_2.n_deltas);
                    firstParent.setFitness(child_2.fitness);
                }
            }
        }
        return population;
    }
    */
}

