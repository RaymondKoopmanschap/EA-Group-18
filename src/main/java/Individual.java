import java.util.Random;
import java.util.Arrays;
import java.util.*;
import java.lang.Math;
import org.vu.contest.ContestEvaluation;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
//import apache.commons.math3.distribution.*;


public class Individual {
    int POPULATION_SIZE = 100;
    List<Double> genotype;
    Double fitness;
    Random rnd;

    int age;

    Double ranking_probability_linear;
    Double ranking_probability_exp;

    ContestEvaluation evaluation;

    Double delta;

    List<Double> n_deltas;
    double[] n_alphas;
    double[][] covMatrix;

    @Override
    public String toString() {
        return " " + this.genotype + "@" + this.fitness;
        /*
        return fitness + "@" +
            //this.genotype + 
            String.format("%.2f", delta) + "@" + String.format("%.2f", n_deltas.get(1));
        */
    }

    public Individual(Random rnd, ContestEvaluation evaluation, List<Double> genotype) {
        this.genotype = genotype;
        this.rnd = rnd;
        this.age = 0;
        this.evaluation = evaluation;
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
        int dimensions = 10;
        int nAlpha = dimensions * (dimensions - 1) / 2;
        this.n_alphas = new double[nAlpha];
    }

    public Individual(Random rnd, ContestEvaluation evaluation) {
        // best for bentcigar
        /*
        this(rnd, evaluation, Arrays.asList(
                    //@9.999996891249292
                    //-1.0637434667691323, 4.002247195181288, -0.2222551655214499, -3.602248046618781, 0.11243458597276558, -1.9207687480867317, 1.2555547243618332, -0.7723689151136095, 1.3714606952113435, -0.38436238638777276
                    ));
                    */
        this(rnd, evaluation, new ArrayList<Double>(10));

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
        double epsilon = 0.00000001;
        double tau = 1 / Math.sqrt(POPULATION_SIZE);

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        this.delta = this.delta * Math.exp(tau * rnd.nextGaussian());
        if (this.delta < epsilon) {
            this.delta = epsilon; 
        }
        //System.out.println(this.delta);
        this.genotype.set(randomAllele, this.genotype.get(randomAllele) + this.delta * rnd.nextGaussian());
    }

    public void UncorrelatedMutationNStepSizes(double epsilon, double first_arg, double second_arg) {
        double tau = 1  / (Math.sqrt(first_arg * Math.sqrt(POPULATION_SIZE)));
        double tau_prime = 1 / (Math.sqrt(second_arg * POPULATION_SIZE));

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;


        double new_delta = this.n_deltas.get(randomAllele) * Math.exp(tau_prime * rnd.nextGaussian() + 
                    tau * rnd.nextGaussian());

        if (new_delta < epsilon) {
            new_delta = epsilon;
        }
        this.n_deltas.set(randomAllele,new_delta);

        this.genotype.set(randomAllele, this.genotype.get(randomAllele) + this.n_deltas.get(randomAllele) * rnd.nextGaussian());
    }

    public void CorrelatedMutation(double epsilon, double first_arg, double second_arg) {
        int dimensions = 10;
        double tau =  1  / (Math.sqrt(first_arg * Math.sqrt(dimensions)));
        double tau_prime =  1 / (Math.sqrt(second_arg * dimensions));

        double beta = 0.087; // 5 degrees

        double tau_gauss = tau_prime * rnd.nextGaussian();

        double[] dx = new double[dimensions];
        this.covMatrix = new double[dimensions][dimensions];

        // mutate sigmas
        for (int i = 0; i < dimensions; i++) {
            this.n_deltas.set(i, Math.max(0.00001, this.n_deltas.get(i) * Math.exp(tau_gauss + tau * rnd.nextGaussian())));
            //System.out.println(n_deltas.get(i));
        }

        int nAlpha = dimensions * (dimensions - 1) / 2;
        // mutate alphas
        for (int j = 0; j < nAlpha; j++) {
            this.n_alphas[j] += beta * rnd.nextGaussian();
            /*
            if (Math.abs(this.n_alphas[j]) > Math.PI) {
                this.n_alphas[j] -= 2 * Math.PI * Math.signum(this.n_alphas[j]);
            }
            */
        }
        System.out.println(Arrays.toString(n_alphas) + " <- alphas");
        //System.out.println(n_alphas[0]);

        // calculate covariance matrix
        calculateCovarianceMatrix(dimensions);

        double [] means = new double[dimensions];

        // get the samples from the multivariate normal distribution
        //System.out.println(Arrays.toString(n_alphas));
        //
        //System.out.println("aaaa");
        for (int i = 0; i < 10; i++) {
            //System.out.println(Arrays.toString(this.covMatrix[i]));
            for (int j = 0; j < 10; j++) {
                //System.out.printf( "   " +  "%+.2f", this.covMatrix[i][j]);
            }
            //System.out.println();
        }
        dx = new MultivariateNormalDistribution(means, covMatrix).sample(); // -> apache/commons/math/probability
        
        //dx = multivariateNormalDistribution(dimensions, rnd)[0];

        // mutate the genotype
        for (int i = 0; i < this.genotype.size(); i++) {
            this.genotype.set(i, keepInRange(this.genotype.get(i) + dx[i]));
            //System.out.println(dx[i]);
        }
    }

    private double keepInRange(double val) {
        return Math.min(5.0, Math.max(
                -5.0, val));
    }

    private double[][] multivariateNormalDistribution(int n, Random rnd_) {
        // covariance matrix
        Matrix covMatrix = new Matrix(this.covMatrix);
        // generate the L from the Cholesky Decomposition
        Matrix L = covMatrix.chol().getL();

        // draw samples from the normal gaussian
        double[] normSamples = new double[n];
        for (int i = 0; i < n; i++) {
            normSamples[i] = rnd_.nextGaussian();
        }

        // construct Matrix
        Matrix z = new Matrix(normSamples, 1);
        return L.times(z.transpose()).transpose().getArray();
    }

    private void calculateCovarianceMatrix(int dimensions) {
        // index used to traverse the alphas array
        int alphaIndex = 0;
        // calculate values on the diagonal and above it
        for (int i = 0; i < dimensions; i++) {
            this.covMatrix[i][i] = Math.pow(this.n_deltas.get(i), 2);
            for (int j = i + 1; j < dimensions; j++) {
                this.covMatrix[i][j] = 0.5 * (Math.pow(this.n_deltas.get(i), 2) -
                        Math.pow(this.n_deltas.get(j), 2)) * Math.tan(
                                2 * this.n_alphas[alphaIndex]);
            }
            alphaIndex++;
        }

        // calculate values under the diagonal
        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < i; j++) {
                this.covMatrix[i][j] = this.covMatrix[j][i];
            }
        }
    }

    /*
    public void CorrelatedMutation(double epsilon, double first_arg, double second_arg) {
        double tau = 1  / (Math.sqrt(first_arg * Math.sqrt(POPULATION_SIZE)));
        double tau_prime = 1 / (Math.sqrt(second_arg * POPULATION_SIZE));
        double beta = 5;

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;

        double new_delta = this.n_deltas.get(randomAllele) * Math.exp(tau_prime * rnd.nextGaussian() + 
                    tau * rnd.nextGaussian());

        double new_alpha = this.n_alphas.get(randomAllele) + beta * rnd.nextGaussian();

        if (new_delta < epsilon) {
            new_delta = epsilon;
        }
        this.n_deltas.set(randomAllele, new_delta);
        this.n_alphas.set(randomAllele, new_alpha);

        this.genotype.set(randomAllele, this.genotype.get(randomAllele) + this.n_deltas.get(randomAllele) * rnd.nextGaussian());
    }

    */
}
