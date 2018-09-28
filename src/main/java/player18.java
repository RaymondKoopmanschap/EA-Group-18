import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;

import java.util.*;
import java.lang.Double;


public class player18 implements ContestSubmission {
    Random rnd_;
    ContestEvaluation evaluation_;
    private int evaluations_limit_;

    private int POPULATION_SIZE = 100;
    
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
        } else if (isMultimodal && !hasStructure && ! isSeparable) {
            //Katsuura
        } else if (isMultimodal && hasStructure && !isSeparable) {
            //Schaffers
        }
    }



    public void run() {
        // Run your algorithm here
        //System.out.println("Evaluations limit: " + evaluations_limit_);

        int evals = 0;

        // init population
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
            boolean whole_population_reproduction = false;
            for (int i = 0; i <  10; i++) {
                //one-point crossover
                // from 1 to 9
                int firstIndividual = rnd_.nextInt(parents.size());
                int secondIndividual = rnd_.nextInt(parents.size());
                if (whole_population_reproduction) {
                    firstIndividual = rnd_.nextInt(population.size());
                    secondIndividual = rnd_.nextInt(population.size());
                }

                int crossoverPoint = rnd_.nextInt(rnd_.nextInt((7 - 0) + 1) + 1);
                List<Double> childGenotype1 = new ArrayList<Double>(10);
                List<Double> childGenotype2 = new ArrayList<Double>(10);
                if (whole_population_reproduction) {
                    childGenotype1.addAll(population.get(firstIndividual).genotype.subList(0, crossoverPoint));
                    childGenotype1.addAll(population.get(secondIndividual).genotype.subList(crossoverPoint, 10));

                    childGenotype2.addAll(population.get(secondIndividual).genotype.subList(0, crossoverPoint));
                    childGenotype2.addAll(population.get(firstIndividual).genotype.subList(crossoverPoint, 10));
                } else {
                    childGenotype1.addAll(parents.get(firstIndividual).genotype.subList(0, crossoverPoint));
                    childGenotype1.addAll(parents.get(secondIndividual).genotype.subList(crossoverPoint, 10));

                    childGenotype2.addAll(population.get(secondIndividual).genotype.subList(0, crossoverPoint));
                    childGenotype2.addAll(population.get(firstIndividual).genotype.subList(crossoverPoint, 10));
                }
                Individual child_1 = new Individual(rnd_, childGenotype1);
                Individual child_2 = new Individual(rnd_, childGenotype2);
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
                for (int j = 0; j < 10; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < arithmetic_parents.size(); k++) {
                        sum += population.get(arithmetic_parents.get(k)).genotype.get(j);
                    }
                    childGenotype.add(sum/number_of_parents);
                }
                Individual child = new Individual(rnd_, childGenotype);
                population.add(child);
            }

            
            // mutation
            double mutation_probability = 0.10;
            for (int i = 0; i < population.size(); i ++) {
                double dice_roll = rnd_.nextDouble();
                if (dice_roll < mutation_probability) {
                    //population.get(i).uniformMutation();
                    //population.get(i).nonUniformMutation();
                    //population.get(i).UncorrelatedMutationOneStepSize();
                    population.get(i).UncorrelatedMutationNStepSizes();
                }
                dice_roll = rnd_.nextDouble();
                if (dice_roll < 0.10) {
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
            // TODO: add randomness
            population = population.subList(0, POPULATION_SIZE);
            //System.out.println(population.get(0).fitness);
            //System.out.println(population.subList(0, 1));
        }
    }
}
