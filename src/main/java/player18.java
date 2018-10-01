import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;

import java.util.*;
import java.lang.Double;
import java.lang.Math;


public class player18 implements ContestSubmission {
    Random rnd_;
    ContestEvaluation evaluation_;
    private int evaluations_limit_;

    private boolean isSchaffers = false;
    
    List<Individual> population;

    public player18() {
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
            isSchaffers = true;
        } else if (isMultimodal && !hasStructure && ! isSeparable) {
            //Katsuura
            isSchaffers = true;
        } else if (isMultimodal && hasStructure && !isSeparable) {
            isSchaffers = true;
            //Schaffers
        }
    }



    public void run() {
        if (isSchaffers) {
            runSchaffers();
            return;
        }
        // Run your algorithm here
        //System.out.println("Evaluations limit: " + evaluations_limit_);

        int evals = 0;

        // init population
        int POPULATION_SIZE = 100;
        population = new ArrayList<Individual>(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            population.add(new Individual(rnd_));
        }


        // calculate fitness
        while (evals < evaluations_limit_) {

            // Select parents
            for (Individual individual: population) {
                // Check fitness of unknown fuction
                if (individual.getFitness() == null) {
                    if (evals >= evaluations_limit_) {
                        break;
                    }
                    evals++;
                    Double fitness = (Double) evaluation_.evaluate(individual.getGenotypeArray());
                    individual.setFitness(fitness);

                }
            }
            
            List<Individual> parents = new ArrayList<Individual>(population.subList(0, POPULATION_SIZE / 10));

            // recombination
            //double recombination_probability = 0.1;

            //one-point crossover
            for (int i = 0; i <  10; i++) {
                //one-point crossover
                // from 1 to 9
                int firstIndividual = rnd_.nextInt(parents.size());
                int secondIndividual = rnd_.nextInt(parents.size());

                int crossoverPoint = rnd_.nextInt(rnd_.nextInt((7 - 0) + 1) + 1);
                List<Double> childGenotype1 = new ArrayList<Double>(10);
                List<Double> childGenotype2 = new ArrayList<Double>(10);

                List<Double> childNDeltas1 = new ArrayList<Double>(10);
                List<Double> childNDeltas2 = new ArrayList<Double>(10);

                childGenotype1.addAll(parents.get(firstIndividual).genotype.subList(0, crossoverPoint));
                childGenotype1.addAll(parents.get(secondIndividual).genotype.subList(crossoverPoint, 10));

                //are you supposed to inherit deltas? I guess you are
                childNDeltas1.addAll(parents.get(firstIndividual).n_deltas.subList(0, crossoverPoint));
                childNDeltas1.addAll(parents.get(secondIndividual).n_deltas.subList(crossoverPoint, 10));

                childGenotype2.addAll(parents.get(secondIndividual).genotype.subList(0, crossoverPoint));
                childGenotype2.addAll(parents.get(firstIndividual).genotype.subList(crossoverPoint, 10));

                //are you supposed to inherit deltas? I guess you are
                childNDeltas2.addAll(parents.get(secondIndividual).n_deltas.subList(0, crossoverPoint));
                childNDeltas2.addAll(parents.get(firstIndividual).n_deltas.subList(crossoverPoint, 10));

                Individual child_1 = new Individual(rnd_, childGenotype1);
                Individual child_2 = new Individual(rnd_, childGenotype2);
                child_1.setNDeltas(childNDeltas1);
                child_2.setNDeltas(childNDeltas2);

                population.add(child_1);
                population.add(child_2);
            }

            for (int i = 0; i <  10; i++) { // creates 10
                //arithmetic crossover
                int number_of_parents = 3;
                List<Integer> arithmetic_parents = new ArrayList<Integer>(number_of_parents);

                for (int j = 0; j < number_of_parents; j++) {
                    arithmetic_parents.add(rnd_.nextInt(parents.size()));
                }

                List<Double> childGenotype = new ArrayList<Double>(10);
                List<Double> childNDeltas = new ArrayList<Double>(10);
                for (int j = 0; j < 10; j++) {
                    double sum = 0.0;
                    double deltas_sum = 0.0;
                    for (int k = 0; k < arithmetic_parents.size(); k++) {
                        sum += population.get(arithmetic_parents.get(k)).genotype.get(j);
                        deltas_sum += population.get(arithmetic_parents.get(k)).n_deltas.get(j);
                    }
                    childGenotype.add(sum/number_of_parents);
                    childNDeltas.add(sum/number_of_parents);
                }
                Individual child = new Individual(rnd_, childGenotype);
                population.add(child);

                // seems like a not very good idea to inherit it here
                //child.setNDeltas(childNDeltas);
            }

            
            // mutation
            double mutation_probability = 0.1;
            for (int i = 0; i < population.size(); i ++) {
                double dice_roll = rnd_.nextDouble();
                if (dice_roll < mutation_probability) {
                    //population.get(i).uniformMutation();
                    //population.get(i).nonUniformMutation();
                    //population.get(i).UncorrelatedMutationOneStepSize();
                    population.get(i).UncorrelatedMutationNStepSizes();
                }
                dice_roll = rnd_.nextDouble();
                if (dice_roll < 0.1) {
                    //population.get(i).uniformMutation();
                    population.get(i).nonUniformMutation();
                }
            }

            for (Individual individual: population) {
                // Check fitness of unknown fuction
                if (individual.getFitness() == null) {
                    if (evals >= evaluations_limit_) {
                        break;
                    }
                    evals++;
                    /*
                    double[] arra = {
                        -1.1481615452524465, 4.002046649371003, -0.43178744966559485, -3.4895948267289043, 0.44376272902370223, -1.8086296469778993, 1.1882329735675545, -0.7882132938335314, 1.5179851066816126, -0.27430498274451875
                       };
                    Double fitness = (Double) evaluation_.evaluate(arra);
                    */
                    
                    Double fitness = (Double) evaluation_.evaluate(individual.getGenotypeArray());
                    //System.out.println(fitness);
                    //System.out.println(Arrays.toString(individual.getGenotypeArray()));
                    individual.setFitness(fitness);
                }
            }
            // Select survivors
            Collections.sort(population, new Comparator<Individual>() {
                @Override
                public int compare(Individual i_1, Individual i_2) {
                    return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
                }
            });

            boolean age_based_selection = true;
            population = population.subList(0, 10);

            if (age_based_selection) {
                List<Individual> new_population = new ArrayList<Individual>(100);
                // age based
                for (int i=10; i <  population.size(); i++) {
                    population.get(i).age += 1;
                    if (population.get(i).age < 3) {
                        new_population.add(population.get(i));
                    }
                }
                population.addAll(new_population);
            } else {
                // TODO: add randomness
            }

            //System.out.println(population.get(0).fitness);
            //System.out.println(population.get(99).fitness);
        }

        Collections.sort(population, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });
        System.out.println(population.get(0));
    }

    public void runSchaffers() {
        // Run your algorithm here
        //System.out.println("Evaluations limit: " + evaluations_limit_);
        int ISLANDS = 5;
        int POPULATION_SIZE = 100;

        int evals = 0;
        List<Island> islands = new ArrayList<Island>();
        for (int i = 0; i < ISLANDS; i++) {
            islands.add(InitializeRandomIsland(POPULATION_SIZE));
        }

        int epochs = 0;
        while (evals < evaluations_limit_) {
            epochs += 1;
            for (int i = 0; i < ISLANDS; i++) {
                for (Individual individual: islands.get(i).population) {
                    if (individual.getFitness() == null) {
                        if (evals >= evaluations_limit_) {
                            break;
                        }
                        evals++;
                        Double fitness = (Double) evaluation_.evaluate(individual.getGenotypeArray());
                        individual.setFitness(fitness);
                    }
                }
            }
            for (int i = 0; i < ISLANDS; i++) {
                List<Individual> population = islands.get(i).population;

                Collections.sort(population, new Comparator<Individual>() {
                    @Override
                    public int compare(Individual i_1, Individual i_2) {
                        return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
                    }
                });

                List<Individual> parents = StohasticUniversalSampling(population);

                List<Individual> children = new ArrayList<Individual>();
                for (int j = 0; j < POPULATION_SIZE * 0.55; j++) {
                    children.addAll(WholeArithmeticRecombination(population));
                }
                /*
                for (int j = 0; j < POPULATION_SIZE  * 0.1; j++) { // creates 10
                    children.addAll(ArithmeticCrossover(population, 2));
                }
                */

                // mutation
                double mutation_probability = 0.1;
                for (int k = 0; k < children.size(); k ++) {
                    int j = rnd_.nextInt(children.size());
                    double dice_roll = rnd_.nextDouble();
                    if (dice_roll < mutation_probability) {
                        //children.get(j).uniformMutation();
                        //children.get(j).nonUniformMutation();
                        //children.get(j).UncorrelatedMutationOneStepSize();
                        children.get(j).UncorrelatedMutationNStepSizes();
                    }
                }

                for (Individual individual: children) {
                    if (individual.getFitness() == null) {
                        if (evals >= evaluations_limit_) {
                            break;
                        }
                        evals++;
                        Double fitness = (Double) evaluation_.evaluate(individual.getGenotypeArray());
                        individual.setFitness(fitness);
                    }
                }
                
                List<Individual> survivors = mi_plus_lambda_fitness_based(parents, children, POPULATION_SIZE);
                islands.get(i).setPopulation(survivors);
                islands.get(i).incrementGeneration();

                if (epochs % 10 == 0) {
                    System.out.println(islands.get(i).last_recorded_fitness_changed + " " + islands.get(i).generations_without_fitness_change);
                    System.out.println(islands.get(i).population.get(0).genotype.get(0) + " 0 " + islands.get(i).last_recorded_fitness_changed);
                    System.out.println(islands.get(i).population.get(1).genotype.get(0) + " 1 " + i);
                    System.out.println(islands.get(i).population.get(2).genotype.get(0) + " 3 " + i);
                }
                if (islands.get(i).generations_without_fitness_change > 50) {
                    //System.out.println("initialized new island");
                    IslandMigration(islands.get(i), islands);
                    //islands.set(i, InitializeRandomIsland(POPULATION_SIZE));
                }
                if (islands.get(i).generations_without_fitness_change > 10 && i % 3 == 0) {
                    //System.out.println("initialized new island");
                    //IslandMigration(islands.get(i), islands);
                    islands.set(i, InitializeRandomIsland(POPULATION_SIZE));
                }
            }
        }
    }

    public List<Individual> InitializeRandomPopulation(int population_size) {
        List<Individual> temp_population = new ArrayList<Individual>(population_size);
        for (int j = 0; j < population_size; j++) {
            temp_population.add(new Individual(rnd_));
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


    public List<Individual> StohasticUniversalSampling(List<Individual> population) {
        Collections.sort(population, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });

        double s = 1.5;
        for (int i = 0; i < population.size(); i ++) {
            population.get(i).setRankingProbabilityLinear(
                    (2 - s) / population.size() +
                    (2 * (population.size() - 1 - i) * (s - 1)) / (population.size() * (population.size() - 1))
            );
        }

        List<Individual> parents = new ArrayList<Individual>();
        int total_number_of_parents = population.size();
        int produced_parents = 50;
        int current_member = 0;
        int i = 0;
        while (current_member < produced_parents) {
            double r = rnd_.nextDouble();
            r =  r * (1.0 / total_number_of_parents);
            while (r <= population.get(i).ranking_probability_linear) { 
                parents.add(population.get(i));
                r = r + 1.0 / total_number_of_parents;
                current_member += 1;
            }
            i = i + 1;
        }
        return parents;
    }

    public List<Individual> TournamentSelection(List<Individual> population) {

        List<Individual> parents = new ArrayList<Individual>();
        List<Integer> added_parents = new ArrayList<Integer>();
        int k = 2;
        int produced_parents = population.size();

        for (int j = 0; j < population.size(); j++) {
            int firstIndividual = rnd_.nextInt(population.size());
            int secondIndividual = rnd_.nextInt(population.size());
            if (population.get(firstIndividual).fitness > population.get(secondIndividual).fitness) {
                if (added_parents.contains(firstIndividual)) {
                    continue;
                }
                parents.add(population.get(firstIndividual));
                added_parents.add(firstIndividual);
            } else {
                if (added_parents.contains(secondIndividual)) {
                    continue;
                }
                parents.add(population.get(secondIndividual));
                added_parents.add(secondIndividual);
            }
        }
        return parents;
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
        Individual child_1 = new Individual(rnd_, childGenotype1);
        Individual child_2 = new Individual(rnd_, childGenotype2);

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
        Individual child = new Individual(rnd_, childGenotype);
        //child.setNDeltas(childNDeltas);
        children.add(child);
        return children;
    }

    public List<Individual> WholeArithmeticRecombination(List<Individual> parents) {
        double alpha = 0.4;
        // child1 = alpha * x + (1 - alpha) * y
        // child2 = alpha * y + (1 - alpha) * x
        
        List<Individual> children = new ArrayList<Individual>(2);

        int parentIndex_1 = rnd_.nextInt(parents.size());
        int parentIndex_2 = rnd_.nextInt(parents.size());

        List<Double> childGenotype1 = new ArrayList<Double>(10);
        List<Double> childGenotype2 = new ArrayList<Double>(10);
        List<Double> childNDeltas1 = new ArrayList<Double>(10);
        List<Double> childNDeltas2 = new ArrayList<Double>(10);

        List<Double> childGenotype = new ArrayList<Double>(10);
        List<Double> childNDeltas = new ArrayList<Double>(10);
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

        Individual child_1 = new Individual(rnd_, childGenotype1);
        Individual child_2 = new Individual(rnd_, childGenotype2);

        child_1.setNDeltas(childNDeltas1);
        child_2.setNDeltas(childNDeltas2);

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
        //children.addAll(parents.subList(0, 30));
        List<Individual> new_children = StohasticUniversalSampling(children);
        //System.out.println(new_children.size());
        Collections.sort(children, new Comparator<Individual>() {
            @Override
            public int compare(Individual i_1, Individual i_2) {
                return - Double.compare(i_1.getFitnessToCompare(), i_2.getFitnessToCompare());
            }
        });
        return children.subList(0, population_size);
    }
}

/*
            // deterministic crowding - does not work
            parents = new ArrayList<Individual>(population.subList(0, (int) (population.size() * 0.5)));
            for (int i = 0; i < population.size() * 0.01; i++) {
                //one-point crossover
                // from 1 to 9
                int firstIndividual = rnd_.nextInt(parents.size());
                int secondIndividual = rnd_.nextInt(parents.size());
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

                Individual child_1 = new Individual(rnd_, childGenotype1);
                Individual child_2 = new Individual(rnd_, childGenotype2);

                child_1.setNDeltas(childNDeltas1);
                child_2.setNDeltas(childNDeltas2);

                child_1.UncorrelatedMutationNStepSizes();
                child_2.UncorrelatedMutationNStepSizes();
                //child_1.nonUniformMutation();
                //child_2.nonUniformMutation();
                //child_1.uniformMutation();
                //child_2.uniformMutation();

                child_1.setFitness((Double) evaluation_.evaluate(child_1.getGenotypeArray()));
                child_2.setFitness((Double) evaluation_.evaluate(child_2.getGenotypeArray()));
                evals += 2;

                if (evals > evaluations_limit_) 
                    return;

                // calc distance between vectors in genotype space
                double cosineSim11 = cosineSimilarity(
                        firstParent.getGenotypeArray(), child_1.getGenotypeArray());
                double cosineSim12 = cosineSimilarity(
                        firstParent.getGenotypeArray(), child_2.getGenotypeArray());
                double cosineSim21 = cosineSimilarity(
                        secondParent.getGenotypeArray(), child_1.getGenotypeArray());
                double cosineSim22 = cosineSimilarity(
                        secondParent.getGenotypeArray(), child_2.getGenotypeArray());
                //if (crossoverPoint < 5) {
                //if (i %2 == 0) {
                if (cosineSim11 + cosineSim22 > cosineSim21 + cosineSim12) {
                //if (Math.abs(child_1.fitness - firstParent.fitness) < Math.abs(child_1.fitness + secondParent.fitness)) {
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
            */


