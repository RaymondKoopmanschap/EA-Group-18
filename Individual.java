import java.util.Random;
import java.util.Arrays;
import java.util.*;
import java.lang.Math;
import org.vu.contest.ContestEvaluation;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.*;
import org.apache.commons.math3.linear.*;
//import apache.commons.math3.distribution.*;


public class Individual {
    int POPULATION_SIZE = 100;
    List<Double> genotype;
    Double fitness;
    Random rnd;

    int age;
    int tmp_population_index;

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
            this.n_deltas.add(1.0);
            //this.n_deltas.add(Math.exp( rnd.nextGaussian()
        }
        int dimensions = 10;
        int nAlpha = dimensions * (dimensions - 1) / 2;
        this.n_alphas = new double[nAlpha];
        calculateLMatrix(dimensions);
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

    public void setTempPopulationIndex(int index) {
        this.tmp_population_index = index;
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

    public void setNAlphas(double[] n_alphas) {
        this.n_alphas = n_alphas;
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

    public double[] getNDeltasArray() {
        double[] ret = new double[this.n_deltas.size()];
        for (int i = 0; i < 10; i ++) {
            ret[i] = this.n_deltas.get(i);
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
        this.fitness = null;
        double epsilon = 0.00000001;
        double tau = 1 / Math.sqrt(POPULATION_SIZE);

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        this.delta = this.delta * Math.exp(tau * rnd.nextGaussian());
        if (this.delta < epsilon) {
            this.delta = epsilon; 
        }
        //System.out.println(this.delta);
        this.genotype.set(randomAllele, keepInRange(this.genotype.get(randomAllele) + this.delta * rnd.nextGaussian()));
    }

    public void UncorrelatedMutationNStepSizes(double epsilon, double first_arg, double second_arg) {
        this.fitness = null;
        double tau = 1  / (Math.sqrt(first_arg * Math.sqrt(POPULATION_SIZE)));
        double tau_prime = 1 / (Math.sqrt(second_arg * POPULATION_SIZE));

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;


        double new_delta = this.n_deltas.get(randomAllele) * Math.exp(tau_prime * rnd.nextGaussian() + 
                    tau * rnd.nextGaussian());

        if (new_delta < epsilon) {
            new_delta = epsilon;
        }
        this.n_deltas.set(randomAllele,new_delta);

        this.genotype.set(randomAllele, keepInRange(this.genotype.get(randomAllele) + this.n_deltas.get(randomAllele) * rnd.nextGaussian()));
    }

    public void UncorrelatedMutationNStepSizesMutateAll(double epsilon, double first_arg, double second_arg) {
        this.fitness = null;
        double tau = 1  / (Math.sqrt(first_arg * Math.sqrt(POPULATION_SIZE)));
        double tau_prime = 1 / (Math.sqrt(second_arg * POPULATION_SIZE));

        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        for (int i = 0; i < 10; i++) {
            double new_delta = this.n_deltas.get(i) * Math.exp(tau_prime * rnd.nextGaussian() + 
                    tau * rnd.nextGaussian());
            if (new_delta < epsilon) {
                new_delta = epsilon;
            }
            this.n_deltas.set(i,new_delta);
            this.genotype.set(i, this.genotype.get(i) + this.n_deltas.get(i) * rnd.nextGaussian());
        }
    }

    public void CorrelatedMutation(double epsilon, double first_arg, double second_arg) {
        this.fitness = null;
        int dimensions = 10;
        double tau =  1  / (Math.sqrt(first_arg * Math.sqrt(dimensions)));
        double tau_prime =  1 / (Math.sqrt(second_arg * dimensions));

        double beta = 0.0873; // 5 degrees

        double tau_gauss = tau_prime * rnd.nextGaussian();

        double[] dx = new double[dimensions];
        this.covMatrix = new double[dimensions][dimensions];

        // mutate sigmas
        for (int i = 0; i < dimensions; i++) {
            this.n_deltas.set(i, this.n_deltas.get(i) * Math.exp(tau_gauss + tau * rnd.nextGaussian()));
            //System.out.println(n_deltas.get(i));
        }

        int nAlpha = dimensions * (dimensions - 1) / 2;
        // mutate alphas
        for (int j = 0; j < nAlpha; j++) {
            this.n_alphas[j] += beta * rnd.nextGaussian();
            if (Math.abs(this.n_alphas[j]) > Math.PI) {
                this.n_alphas[j] -= 2 * Math.PI * Math.signum(this.n_alphas[j]);
            }
        }
        //System.out.println(Arrays.toString(n_alphas) + " <- alphas");
        //System.out.println(this.n_deltas + " <- sigmas");
        //System.out.println(n_alphas[0]);

        // calculate covariance matrix
        calculateCovarianceMatrix2(dimensions);


        double [] means = new double[dimensions];
        dx = new MultivariateNormalDistribution(means, this.covMatrix).sample(); // -> apache/commons/math/probability
        
        //dx = multivariateNormalDistribution(dimensions, rnd)[0];

        // mutate the genotype
        int randomAllele = rnd.nextInt((9 - 0) + 1) + 0;
        this.genotype.set(randomAllele, keepInRange(this.genotype.get(randomAllele) + dx[randomAllele]));

        /*
        for (int i = 0; i < this.genotype.size(); i++) {
            this.genotype.set(i, keepInRange(this.genotype.get(i) + dx[i]));
            //System.out.println(dx[i]);
        }
        */
    }
    public void CorrelatedMutation2(double epsilon, double first_arg, double second_arg) {
        this.fitness = null;
        //Benjamin mutation do not use
        int dimensions = 10;
        double tau =  1  / (Math.sqrt(first_arg * Math.sqrt(dimensions)));
        double tau_prime =  1 / (Math.sqrt(second_arg * dimensions));

        double beta = 0.0873; // 5 degrees

        double tau_gauss = tau_prime * rnd.nextGaussian();

        this.covMatrix = new double[dimensions][dimensions];

        // mutate sigmas
        for (int i = 0; i < dimensions; i++) {
            this.n_deltas.set(i, this.n_deltas.get(i) * Math.exp(tau_gauss + tau * rnd.nextGaussian()));
            //System.out.println(n_deltas.get(i));
        }

        int nAlpha = dimensions * (dimensions - 1) / 2;
        // mutate alphas
        for (int j = 0; j < nAlpha; j++) {
            this.n_alphas[j] += beta * rnd.nextGaussian();
            if (Math.abs(this.n_alphas[j]) > Math.PI) {
                this.n_alphas[j] -= 2 * Math.PI * Math.signum(this.n_alphas[j]);
            }
        }
        //System.out.println(Arrays.toString(n_alphas) + " <- alphas");
        //System.out.println(this.n_deltas + " <- sigmas");
        //System.out.println(n_alphas[0]);

        // calculate covariance matrix
        //calculateCovarianceMatrix2(dimensions);

        Matrix LMatrix = calculateLMatrix(dimensions);

        double [] means = new double[dimensions];
        //dx = new MultivariateNormalDistribution(means, this.covMatrix).sample(); // -> apache/commons/math/probability
        
        //dx = multivariateNormalDistribution(dimensions, rnd)[0];

        // mutate the genotype
        int randomAxis = rnd.nextInt((9 - 0) + 1) + 0;
        double [][] epsilon_vector = new double[10][1];
        epsilon_vector[randomAxis][0] = rnd.nextGaussian();
        Matrix epsilonMatrix = new Matrix(epsilon_vector);
        //this.genotype.set(randomAllele, keepInRange(this.genotype.get(randomAllele) + dx[randomAllele]));
        Matrix dX = LMatrix.times(epsilonMatrix);
        for (int i = 0; i < this.genotype.size(); i++) {
            this.genotype.set(i, keepInRange(this.genotype.get(i) + dX.A[i][0]));
            //System.out.println(dx[i]);
        }
    }

    public double ourMutationDistance(Individual other) {
        int dimensions = 10;

        // this is done in previous step
        //this.calculateCovarianceMatrix2(dimensions);
        //
        //other.calculateCovarianceMatrix2(dimensions);

        double[][] combinedVarianceArray = add2DArrays(this.covMatrix, other.covMatrix);
        RealMatrix combinedVariance = MatrixUtils.createRealMatrix(combinedVarianceArray);
        EigenDecomposition eigenDecomp = new EigenDecomposition(combinedVariance);

        double[][] sqrtLambdaInvArray = new double[dimensions][dimensions];
        for (int i = 0; i < dimensions; i++){
            sqrtLambdaInvArray[i][i] = 1/Math.sqrt(eigenDecomp.getRealEigenvalues()[i]);
        }

        RealMatrix QReal = eigenDecomp.getV();
        Matrix Q = new Matrix(QReal.getData());
        Matrix sqrtLambdaInv = new Matrix(sqrtLambdaInvArray);
        Matrix differenceMatrix = listMatrixConv(this.genotype).minus(listMatrixConv(other.genotype));
        Matrix normalizedDistance = Q.times(sqrtLambdaInv.times(Q.transpose()))
                .times(differenceMatrix.transpose());
        double distance = (normalizedDistance.transpose()).times(normalizedDistance).A[0][0];
        return distance;
    }

    public static double[][] add2DArrays(double[][] matrix1, double[][] matrix2) {
        double[][] addedMatrix = new double[matrix1.length][matrix1[0].length];
        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                addedMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return addedMatrix;
    }
    public static Matrix listMatrixConv(List<Double> genotype){
        double[][] arrayMat = new double[1][genotype.size()];
        for (int i = 0; i < genotype.size(); i++){
            arrayMat[0][i] = genotype.get(i);
        }
        Matrix matrix = new Matrix(arrayMat);
        return matrix;
    }

    /*
    private double keepInRange(double val) {
        if (val < -5) {
            return -5;
        } else if (val > 5) {
            return 5;
        }
        return val;
    }
    */

    private double keepInRange(double val) {
        while (val > 5 || val < -5) {
            if (val > 5) {
                val -= 10;
            }
            else if (val < -5) {
                val += 10;
            }
        }
        return val;
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
                alphaIndex++;
            }
        }

        // calculate values under the diagonal
        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < i; j++) {
                this.covMatrix[i][j] = this.covMatrix[j][i];
            }
        }
    }

    private void calculateCovarianceMatrix2(int dimensions) {
        // index used to traverse the alphas array
        int alphaIndex = 0;
        // calculate values on the diagonal and above it
        double[][] rotation_matrix = new double[dimensions][dimensions];
        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < dimensions; j++) {
                if (i==j) {
                    rotation_matrix[i][j] = 1;
                } else {
                    rotation_matrix[i][j] = 0;
                }
            }
        }
        Matrix rotationalMatrixx = new Matrix(rotation_matrix);

        for (int i = 0; i < dimensions - 1; i++) {
            for (int j = i + 1; j < dimensions; j++) {
                double [][] new_rotational_matrix = new double[dimensions][dimensions];
                for (int b = 0; b < dimensions; b++) {
                    new_rotational_matrix[b][b] = 1;
                }
                new_rotational_matrix[i][i] = Math.cos(this.n_alphas[alphaIndex]);
                new_rotational_matrix[i][j] = -Math.sin(this.n_alphas[alphaIndex]);
                new_rotational_matrix[j][i] = Math.sin(this.n_alphas[alphaIndex]);
                new_rotational_matrix[j][j] = Math.cos(this.n_alphas[alphaIndex]);
                /*
                new_rotational_matrix[i][i] = Math.cos(0);
                new_rotational_matrix[i][j] = -Math.sin(0);
                new_rotational_matrix[j][i] = Math.sin(0);
                new_rotational_matrix[j][j] = Math.cos(0);
                */
                alphaIndex++;
                Matrix newRotationalMatrix = new Matrix(new_rotational_matrix);
                rotationalMatrixx = rotationalMatrixx.times(newRotationalMatrix);

                /*
                 * USE TO DISPLAY ROTATIONAL MATRICES
                for (int k = 0; k < 10; k++) {
                    //System.out.println(Arrays.toString(this.covMatrix[i]));
                    for (int l = 0; l < 10; l++) {
                        System.out.printf( "   " +  "%+.4f", rotationalMatrixx.A[k][l]);
                    }
                    System.out.println();
                }
                System.out.println("NEW MATRIX______________");
                 */
                /*
                */
            }
        }
        double[][] deltas_matrix = new double[dimensions][dimensions];
        for (int i = 0; i < dimensions; i++) {
            deltas_matrix[i][i] = this.n_deltas.get(i) * this.n_deltas.get(i);
            //deltas_matrix[i][i] = 1;
        }
        Matrix deltasMatrix = new Matrix(deltas_matrix);
        /*
        System.out.println("DELTAAAAS");
        for (int k = 0; k < 10; k++) {
            //System.out.println(Arrays.toString(this.covMatrix[i]));
            for (int l = 0; l < 10; l++) {
                System.out.printf( "   " +  "%+.4f", deltasMatrix.A[k][l]);
            }
            System.out.println();
        }
        System.out.println();
        */
        Matrix covarianceMatrix = rotationalMatrixx.times(deltasMatrix).times(rotationalMatrixx.transpose());
        /*
        System.out.println("ROTATIONAL MATRIX:");
        for (int k = 0; k < 10; k++) {
            //System.out.println(Arrays.toString(this.covMatrix[i]));
            for (int l = 0; l < 10; l++) {
                System.out.printf( "   " +  "%+.4f", rotationalMatrixx.A[k][l]);
            }
            System.out.println();
        }
        System.out.println();
        */

        this.covMatrix = covarianceMatrix.A;
        /*
        System.out.println("COVARIANCE MATRIX:");
        for (int k = 0; k < 10; k++) {
            //System.out.println(Arrays.toString(this.covMatrix[i]));
            for (int l = 0; l < 10; l++) {
                System.out.printf( "   " +  "%+.4f", covMatrix[k][l]);
            }
            System.out.println();
        }
        System.out.println();
        */
    }
    private Matrix calculateLMatrix(int dimensions) {
        // index used to traverse the alphas array
        int alphaIndex = 0;
        // calculate values on the diagonal and above it
        double[][] rotation_matrix = new double[dimensions][dimensions];
        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < dimensions; j++) {
                if (i==j) {
                    rotation_matrix[i][j] = 1;
                } else {
                    rotation_matrix[i][j] = 0;
                }
            }
        }
        Matrix rotationalMatrixx = new Matrix(rotation_matrix);

        for (int i = 0; i < dimensions - 1; i++) {
            for (int j = i + 1; j < dimensions; j++) {
                double [][] new_rotational_matrix = new double[dimensions][dimensions];
                for (int b = 0; b < dimensions; b++) {
                    new_rotational_matrix[b][b] = 1;
                }
                new_rotational_matrix[i][i] = Math.cos(this.n_alphas[alphaIndex]);
                new_rotational_matrix[i][j] = -Math.sin(this.n_alphas[alphaIndex]);
                new_rotational_matrix[j][i] = Math.sin(this.n_alphas[alphaIndex]);
                new_rotational_matrix[j][j] = Math.cos(this.n_alphas[alphaIndex]);
                alphaIndex++;
                Matrix newRotationalMatrix = new Matrix(new_rotational_matrix);
                rotationalMatrixx = rotationalMatrixx.times(newRotationalMatrix);

                /*
                 * USE TO DISPLAY ROTATIONAL MATRICES
                for (int k = 0; k < 10; k++) {
                    //System.out.println(Arrays.toString(this.covMatrix[i]));
                    for (int l = 0; l < 10; l++) {
                        System.out.printf( "   " +  "%+.4f", rotationalMatrixx.A[k][l]);
                    }
                    System.out.println();
                }
                System.out.println("NEW MATRIX______________");
                 */
                /*
                */
            }
        }
        double[][] deltas_matrix = new double[dimensions][dimensions];
        for (int i = 0; i < dimensions; i++) {
            deltas_matrix[i][i] = this.n_deltas.get(i); // by purpose not squared
        }
        Matrix deltasMatrix = new Matrix(deltas_matrix);
        /*
        System.out.println("DELTAAAAS");
        for (int k = 0; k < 10; k++) {
            //System.out.println(Arrays.toString(this.covMatrix[i]));
            for (int l = 0; l < 10; l++) {
                System.out.printf( "   " +  "%+.4f", deltasMatrix.A[k][l]);
            }
            System.out.println();
        }
        System.out.println();
        */
        Matrix covarianceMatrix = rotationalMatrixx.times(deltasMatrix).times(rotationalMatrixx.transpose());
        Matrix LMatrix = rotationalMatrixx.times(deltasMatrix);
        /*
        System.out.println("ROTATIONAL MATRIX:");
        for (int k = 0; k < 10; k++) {
            //System.out.println(Arrays.toString(this.covMatrix[i]));
            for (int l = 0; l < 10; l++) {
                System.out.printf( "   " +  "%+.4f", rotationalMatrixx.A[k][l]);
            }
            System.out.println();
        }
        System.out.println();
        */
        Matrix covMatrix = LMatrix.times(LMatrix.transpose());

        this.covMatrix = covarianceMatrix.A;
        /*
        System.out.println("COVARIANCE MATRIX:");
        for (int k = 0; k < 10; k++) {
            //System.out.println(Arrays.toString(this.covMatrix[i]));
            for (int l = 0; l < 10; l++) {
                System.out.printf( "   " +  "%+.4f", covMatrix[k][l]);
            }
            System.out.println();
        }
        System.out.println();
        */
        return LMatrix;
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
