using IntegratedNestedLaplace
using DataFrames
using RDatasets
using Statistics

println("--- Example 04 Performance Test: Salamander Mating ---")

# 1. Load Data
df = dataset("survey", "salamander")
df.Cross = String.(df.Cross)
df.Female = String.(df.Female)
df.Male = String.(df.Male)

# 2. Run Twice to see Warm Performance
for i in 1:2
    println("\nRun $i:")
    start_time = time()
    
    res = inla(
        @formula(Mate ~ 1 + Cross + f(Female, IID) + f(Male, IID)), 
        df, 
        family=BernoulliLikelihood(),
        theta0=[1.0, 1.0]
    )
    
    end_time = time()
    println("  Time: ", round(end_time - start_time, digits=4), "s")
end
