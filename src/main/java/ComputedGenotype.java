import java.util.Random;
import java.util.Arrays;
import java.util.*;
import java.lang.Math;
import org.vu.contest.ContestEvaluation;


public class ComputedGenotype {
    List<Double> genotype;
    Double fitness;

    @Override
    public String toString() {
        return " " + this.genotype + "@" + this.fitness;
        /*
        return fitness + "@" +
            //this.genotype + 
            String.format("%.2f", delta) + "@" + String.format("%.2f", n_deltas.get(1));
        */
    }

    public ComputedGenotype(List<Double> genotype, Double fitness) {
        this.genotype = genotype;
        this.fitness = fitness;
    }
}
