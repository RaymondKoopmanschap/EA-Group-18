import java.util.Random;
import java.util.Arrays;
import java.util.*;
import java.lang.Math;


public class Individual {
    int POPULATION_SIZE = 100;
    List<Double> genotype;
    Double fitness;
    Random rnd;

    Double delta;

    List<Double> n_deltas;

    @Override
    public String toString() {
        return fitness + "@" +
            this.genotype + 
            String.format("%.2f", delta) + "@" + String.format("%.2f", n_deltas.get(1));
    }

    public Individual(Random rnd, List<Double> genotype) {
        this.genotype = genotype;
        this.rnd = rnd;
        if (this.genotype.size() != 10) {
            for (int i = 0; i < 10; i++) {
                this.genotype.add(randomValueInDomain());
            }
        }

        this.delta = rnd.nextGaussian();
        this.n_deltas = new ArrayList<Double>(10);
        for (int i = 0; i < 10; i++) {
            this.n_deltas.add(rnd.nextGaussian());
        }
    }

    public Individual(Random rnd) {
        this(rnd, new ArrayList<Double>(10));
    }

    public double randomValueInDomain() {
        return - 5 + this.rnd.nextDouble() * (5 + 5);
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public Double getFitness() {
        return this.fitness;
    }

    public double[] getGenotypeArray() {
        double[] ret = new double[this.genotype.size()];
        for (int i = 0; i < 10; i ++) {
            ret[i] = this.genotype.get(i);
        }
        return ret;
    }

    public Double getFitnessToCompare() {
        //return this.fitness;
        if (this.fitness == null)
            return new Double(0.0);
        return this.fitness;
    }

    public void uniformMutation() {
        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        this.genotype.set(randomAllele, randomValueInDomain());
        this.fitness = null;
    }


    public void nonUniformMutation() {
        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        this.genotype.set(randomAllele, this.genotype.get(randomAllele) + rnd.nextGaussian());
        this.fitness = null;
    }

    public void UncorrelatedMutationOneStepSize() {
        double tau = 1 / Math.sqrt(POPULATION_SIZE);

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        this.delta = this.delta * Math.exp(tau * rnd.nextGaussian());
        //System.out.println(this.delta);
        this.genotype.set(randomAllele, this.genotype.get(randomAllele) + this.delta * rnd.nextGaussian());
    }

    public void UncorrelatedMutationNStepSizes() {
        double tau = 1 / (Math.sqrt(2.2 * Math.sqrt(POPULATION_SIZE)));
        double tau_prime = 1 / (Math.sqrt(2 * POPULATION_SIZE));

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;

        this.n_deltas.set(randomAllele,
                this.n_deltas.get(randomAllele) * Math.exp(tau_prime * rnd.nextGaussian() + 
                    tau * rnd.nextGaussian()));
        this.genotype.set(randomAllele, this.genotype.get(randomAllele) + this.n_deltas.get(randomAllele) * rnd.nextGaussian());
    }
}
