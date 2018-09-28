import java.util.Random;
import java.util.Arrays;
import java.util.*;

public class Individual {
    List<Double> genotype;
    Double fitness;
    Random rnd;

    public Individual(Random rnd, List<Double> genotype) {
        this.genotype = genotype;
        this.rnd = rnd;
        if (this.genotype.size() != 10) {
            for (int i = 0; i < 10; i++) {
                this.genotype.add(randomValueInDomain());
            }
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

    public void randomResetting() {
        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        this.genotype.set(randomAllele, randomValueInDomain());
        this.fitness = null;
    }
}
