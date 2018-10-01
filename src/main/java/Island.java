import java.util.Random;
import java.util.Arrays;
import java.util.*;
import java.lang.Math;


public class Island {
    List<Individual> population;
    int generations_without_fitness_change;
    Double last_recorded_fitness_changed;

    @Override
    public String toString() {
        return " " + this.generations_without_fitness_change;
        /*
        return fitness + "@" +
            //this.genotype + 
            String.format("%.2f", delta) + "@" + String.format("%.2f", n_deltas.get(1));
        */
    }

    public Island(List<Individual> population) {
        this.population = population;
        this.generations_without_fitness_change = 0;
        this.last_recorded_fitness_changed = 0.0;
    }
    
    public void setPopulation(List<Individual> population) {
        this.population = population;
    }

    public void incrementGeneration() {
        Double this_generation_best_fitness = 0.0;
        for (int i = 0; i < this.population.size(); i++) {
            if(population.get(i).fitness > this_generation_best_fitness) {
                this_generation_best_fitness = population.get(i).fitness;
            }
        }

        if (last_recorded_fitness_changed.equals(this_generation_best_fitness)) {
            generations_without_fitness_change += 1;
        } else {
            last_recorded_fitness_changed = this_generation_best_fitness;
            generations_without_fitness_change = 0;
        }
    }
}
