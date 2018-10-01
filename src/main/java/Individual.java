import java.util.Random;
import java.util.Arrays;
import java.util.*;
import java.lang.Math;


public class Individual {
    int POPULATION_SIZE = 100;
    List<Double> genotype;
    Double fitness;
    Random rnd;

    int age;

    Double ranking_probability_linear;
    Double ranking_probability_exp;

    Double delta;

    List<Double> n_deltas;

    @Override
    public String toString() {
        return " " + this.genotype + "@" + this.fitness;
        /*
        return fitness + "@" +
            //this.genotype + 
            String.format("%.2f", delta) + "@" + String.format("%.2f", n_deltas.get(1));
        */
    }

    public Individual(Random rnd, List<Double> genotype) {
        this.genotype = genotype;
        this.rnd = rnd;
        this.age = 0;
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
        /*
        this(rnd, Arrays.asList(
                    // best for bentcigar
                    -1.1481615452524466, 4.002046649371003, -0.43178744966559485, -3.4897643863689924, 0.44376272902370223, -1.8086296469778993, 1.1882329735675545, -0.7882132938335314, 1.5181745335044008, -0.42285650091772453
                    //-1.1481615452524465, 4.491617537557367, -0.43178744966559485, -3.12080777515226, 0.44376272902370223, -1.8086296469778993, 1.1882329735675545, -0.7882132938335314, 1.368928057796371, -0.42285650091772453
                    ));
        */
        this(rnd, new ArrayList<Double>(10));
        /*
        this(rnd, Arrays.asList(
                    ));
        /*
        this(rnd, Arrays.asList(
                    3.664747911894554, 2.5402683593118547, -1.530645323979111, 1.4463744845556485, 1.377491652390991, -1.8890949554704584, 3.4909693360640537, -2.3250308866867107, -0.3753226786821977, -2.0321133319786613
                    ));
        */
    }

    public double randomValueInDomain() {
        return - 5 + this.rnd.nextDouble() * (5 + 5);
    }

    public void setFitness(Double fitness) {
        this.fitness = fitness;
    }

    public void setGenotype(List<Double> genotype) {
        this.genotype = genotype;
    }

    public Double getFitness() {
        return this.fitness;
    }

    public void setNDeltas(List<Double> n_deltas) {
        this.n_deltas = n_deltas;
    }

    public void setRankingProbabilityLinear(Double ranking_probability_linear) {
        this.ranking_probability_linear = ranking_probability_linear;
    }

    public void setRankingProbabilityExp(Double ranking_probability_Exp) {
        this.ranking_probability_exp = ranking_probability_exp;
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
        double tau = 1  / (Math.sqrt(2 * Math.sqrt(POPULATION_SIZE)));
        double tau_prime = 1 / (Math.sqrt(2 * POPULATION_SIZE));

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;

        this.n_deltas.set(randomAllele,
                this.n_deltas.get(randomAllele) * Math.exp(tau_prime * rnd.nextGaussian() + 
                    tau * rnd.nextGaussian()));

        this.genotype.set(randomAllele, this.genotype.get(randomAllele) + this.n_deltas.get(randomAllele) * rnd.nextGaussian());
    }
}
