
#### IF YOU NEED TO USE THIS DOC IN A NEW ENVIRONMENT:

cd("C:/Users/maxsu/DataAnalysis/julia_hgf") #cd to directory with the correct project.toml and manifest.toml
Pkg.activate(".") #activate environment in current directory (assuming we're in above directory)
Pkg.instantiate() #ensures all correct packages and dependencies in .tomls are installed
include("create_hgf.jl")  # Make sure this is the correct path to your script

## load libraries 
Pkg.add(url="https://github.com/ilabcode/HierarchicalGaussianFiltering.jl")

using HierarchicalGaussianFiltering
using ActionModels
using StatsPlots
using Distributions
using Turing

######## CREATE HGF ######
config = Dict()

#Defaults
spec_defaults = Dict(
    "n_bandits" => 3,

    ("xprob", "volatility") => -2,
    ("xprob", "drift") => 0,
    ("xprob", "autoconnection_strength") => 1,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,

    ("xvol", "volatility") => -2,
    ("xvol", "drift") => 0,
    ("xvol", "autoconnection_strength") => 1,
    ("xvol", "initial_mean") => 0,
    ("xvol", "initial_precision") => 1,

    ("xbin", "xprob", "coupling_strength") => 1,
    ("xprob", "xvol", "coupling_strength") => 1,

    "update_type" => EnhancedUpdate(),
    "save_history" => true,
)

#Merge to overwrite defaults
config = merge(spec_defaults, config)

#Initialize list of nodes
nodes = HierarchicalGaussianFiltering.AbstractNodeInfo[]
edges = Dict{Tuple{String, String}, HierarchicalGaussianFiltering.CouplingType}()
grouped_xprob_volatility = []
grouped_xprob_drift = []
grouped_xprob_autoconnection_strength = []
grouped_xprob_initial_mean = []
grouped_xprob_initial_precision = []
grouped_xbin_xprob_coupling_strength = []
grouped_xprob_xvol_coupling_strength = []

#For each bandit
for i = 1:config["n_bandits"]

    #Add input node
    push!(nodes, BinaryInput("u$i"))

    #Add binary node
    push!(nodes, BinaryState("xbin$i"))

    #Add probability node
    push!(
        nodes,
        ContinuousState(
            name = "xprob$i",
            volatility = config[("xprob", "volatility")],
            drift = config[("xprob", "drift")],
            autoconnection_strength = config[("xprob", "autoconnection_strength")],
            initial_mean = config[("xprob", "initial_mean")],
            initial_precision = config[("xprob", "initial_precision")],
        ),
    )

    #Group the parameters for each binary HGF
    push!(grouped_xprob_volatility, ("xprob$i", "volatility"))
    push!(grouped_xprob_drift, ("xprob$i", "drift"))
    push!(grouped_xprob_autoconnection_strength, ("xprob$i", "autoconnection_strength"))
    push!(grouped_xprob_initial_mean, ("xprob$i", "initial_mean"))
    push!(grouped_xprob_initial_precision, ("xprob$i", "initial_precision"))
    push!(grouped_xbin_xprob_coupling_strength, ("xbin$i", "xprob$i", "coupling_strength"))
    push!(grouped_xprob_xvol_coupling_strength, ("xprob$i", "xvol", "coupling_strength"))

    #Add edges
    push!(edges, ("u$i", "xbin$i") => ObservationCoupling())
    push!(
        edges,
        ("xbin$i", "xprob$i") =>
            ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")]),
    )
    push!(
        edges,
        ("xprob$i", "xvol") =>
            VolatilityCoupling(config[("xprob", "xvol", "coupling_strength")]),
    )

end

#Add the shared volatility parent
push!(
    nodes,
    ContinuousState(
        name = "xvol",
        volatility = config[("xvol", "volatility")],
        drift = config[("xvol", "drift")],
        autoconnection_strength = config[("xvol", "autoconnection_strength")],
        initial_mean = config[("xvol", "initial_mean")],
        initial_precision = config[("xvol", "initial_precision")],
    ),
)

parameter_groups = [
    ParameterGroup("xprob_volatility",
        grouped_xprob_volatility,
        config[("xvol", "volatility")],
    ),
    ParameterGroup("xprob_drift",
        grouped_xprob_drift,
        config[("xvol", "drift")],
    ),
    ParameterGroup("xprob_autoconnection_strength",
        grouped_xprob_autoconnection_strength,
        config[("xvol", "autoconnection_strength")],
    ),
    ParameterGroup("xprob_initial_mean",
        grouped_xprob_initial_mean,
        config[("xvol", "initial_mean")],
    ),
    ParameterGroup("xprob_initial_precision",
        grouped_xprob_initial_precision,
        config[("xvol", "initial_precision")],
    ),
    ParameterGroup("xbin_xprob_coupling_strength",
        grouped_xbin_xprob_coupling_strength,
        config[("xbin", "xprob", "coupling_strength")],
    ),
    ParameterGroup("xprob_xvol_coupling_strength",
        grouped_xprob_xvol_coupling_strength,
        config[("xprob", "xvol", "coupling_strength")],
    ),
]

#Initialize the HGF
hgf = init_hgf(
    nodes = nodes,
    edges = edges,
    parameter_groups = parameter_groups,
    verbose = false,
    node_defaults = NodeDefaults(update_type = config["update_type"]),
    save_history = config["save_history"],
)



##### TRY OUT HGF #####

## READ DATA ##
#One vector per timepoint.
#Consiting of a vector with a value for each bandit
#Which can be 0, 1 or missing (i.e. no observaiton of that bandit)
inputs = [
    [missing, 1, missing],
    [missing, 0, missing],
    [missing, missing, 1],
    [missing, missing, 1],
]

#See current parameter values
get_parameters(hgf)

parameters = Dict("xprob_volatility" => -2,
                "xprob_initial_mean" => 0,)

#Change parameter values
set_parameters!(hgf, parameters)
reset!(hgf)

#Give inputs to the HGF
give_inputs!(hgf, inputs)

#Plot belief trajectory for the HGF
plot_trajectory(hgf, "xprob3")





### CREATE AGENT ###

#Softmax function
function softmax(x::AbstractVector, temp::Real)
    exp_values = exp.(x / temp)
    return exp_values / sum(exp_values)
end

#Action model function
function choose_bandit(agent::Agent, input::Any)

    ### UPDATE HGF ###
    #Unpack the input into which badnit has been observed, and what the observation was
    observed_bandit, observation = input

    #Extrat the HGF
    hgf = agent.substruct

    #Create empty vector of observations
    hgf_input = Vector{Union{Int, Missing}}(missing, length(hgf.input_nodes))

    #Change the missing to the atual observation for the bandit that was observed
    hgf_input[observed_bandit] = observation

    #Pass the observation to the HGF
    update_hgf!(hgf, hgf_input)


    ### PICK A BANDIT ###   
    #Get the predicted probabilites for reqards for each of the bandits
    predicted_probabilities = [hgf.ordered_nodes.input_nodes[i].edges.observation_parents[1].states.prediction_mean for i in 1:length(hgf.input_nodes)]

    #Get the temperature parameter
    β = agent.parameters["softmax_temperature"]

    #Run them through the softmax
    action_proabilities = softmax(predicted_probabilities, β)

    #Return a Categorical probability distribution
    action_distribution = Categorical(action_proabilities)

    return action_distribution
end

#Add the temeprature parmaeter for the action model
parameters = Dict("softmax_temperature" => 1)

#create the agent
agent = init_agent(
    choose_bandit, 
    substruct = hgf,
    parameters = parameters
)


### PLAY WITH THE AGENT ###

#See the parameters in the agent
get_parameters(agent)

#Set parameters
set_parameters!(agent, Dict("softmax_temperature" => 0.1))
reset!(agent)

#"real inputs"
inputs = [
    [1, 1],
    [2, 0],
    [3, 1],
    [1, 0],
    [2, 1],
]

#Run forward to simulate actions
simulated_actions = give_inputs!(agent,inputs)

#Plot belief trajectories
plot_trajectory(agent, "xbin2")


#### SIMULARTING TO SEE WAHT HAPPENS WITH DIFFERENT PARAMETER SETTINGS

true_probs = [0.2, 0.2, 0.8]

next_input = (3,1)

for i = 1:5
    action = single_input!(agent, next_input)

    new_observation = rand(Bernoulli(true_probs[action]))

    next_input = (action, new_observation)
end



### PARAMETER ESTIMATION FOR SINGLE PARTICIPANT ###


# priors = Dict(
#     "xprob_volatility" => Normal(0, 1),
#     #"softmax_temperature" => truncated(Normal(0, 1),lower = 0),
# )

# results = fit_model(agent, priors, inputs, simulated_actions)



### PARAMETER ESTIMATION FOR FULL DATASET ###

# results = fit_model(
#     agent, 
#     priors, 
#     data;
#     independent_group_cols = [:ID],
#     input_cols = [:prev_chosen_bandit, :reward],
#     action_cols = [:chosen_bandit],
#     n_cores = 4,
#     n_chains = 2,
#     n_iterations = 1000
# )

