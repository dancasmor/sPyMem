
import matplotlib.pyplot as plt
from sPyMem.hippocampus_bioinspired_dg_ca1 import hippocampus_bioinspired_dg_ca1
import spynnaker8 as sim
import math
import os

# Parameters:
# + Number of directions of the memory
cueSize = 5
# + Size of the content of the memory in bits/neuron
contSize = 10

# + Time step of the simulation
timeStep = 1.0

# + Experiment:
experiment = 1
#   1) Learn and recall
#   2) Learn, recall and relearning with forget
#   3) A mix of several operations at max frequency (4 ms space between activations of CA3)
#       - 7 ms after learning
#       - 5 ms after recall
if experiment == 1:
    # + Spikes of the input layer
    inputSpikesCue = [[0, 1, 2, 10], [], []]
    inputSpikesCont = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [], [], [], [], [], [], [0, 1, 2]]
    # + Duration of the simulation
    simTime = 20
    # + Name of the experiment, i.e., name of the generated figure
    experiment_name = "1_learn_recall"
elif experiment == 2:
    inputSpikesCue = [[0, 1, 2, 10, 20, 21, 22, 30], [], []]
    inputSpikesCont = [[0, 1, 2], [0, 1, 2], [0, 1, 2, 20, 21, 22], [20, 21, 22], [], [], [], [], [20, 21, 22], [0, 1, 2]]
    simTime = 40
    experiment_name = "2_learn_forget_relearn"
elif experiment == 3:
    # op -> t=0[0,2]=[7,8,9], t=7[2]=[6,7,8], t=14[0,1]=[5,6,7], t=21[0,2], t=26[2], t=31[2]=[0,1,5,6,7],
    #       t=38[0,1]=[0,1,8,9], t=45[2], t=50[0,1]
    inputSpikesCue = [[0, 1, 2, 14, 15, 16, 21, 38, 39, 40, 50], [14, 15, 16, 38, 39, 40, 50], [0, 1, 2, 7, 8, 9, 21, 26, 31, 32, 33, 45]]
    inputSpikesCont = [[31, 32, 33, 38, 39, 40], [31, 32, 33, 38, 39, 40], [], [], [], [14, 15, 16, 31, 32, 33],
                       [7, 8, 9, 14, 15, 16, 31, 32, 33], [0, 1, 2, 7, 8, 9, 14, 15, 16, 31, 32, 33], [0, 1, 2, 7, 8, 9, 38, 39, 40],
                       [0, 1, 2, 38, 39, 40]]
    simTime = 60
    experiment_name = "3_combined_operations"
else:
    inputSpikesCue = [[], [], []]
    inputSpikesCont = [[], [], [], [], [], [], [], [], [], []]
    simTime = 1
    experiment_name = "none"
inputSpikes = inputSpikesCue + inputSpikesCont


def test():
    # + Number of neurons in input layer: the number of bits neccesary to represent the number of directions
    #       in binary + the size of patterns
    numInputLayerNeurons = math.ceil(math.log2(cueSize)) + contSize

    # Setup simulation
    sim.setup(timeStep)

    # Create network
    # Input layer
    ILayer = sim.Population(numInputLayerNeurons, sim.SpikeSourceArray(spike_times=inputSpikes), label="ILayer")
    # Output layer: fire a spike when receive a spike
    neuronParameters = {"cm": 0.27, "i_offset": 0.0, "tau_m": 3.0, "tau_refrac": 1.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                        "v_reset": -60.0, "v_rest": -60.0, "v_thresh": -57.5}
    OLayer = sim.Population(numInputLayerNeurons, sim.IF_curr_exp(**neuronParameters), label="OLayer")
    OLayer.set(v=-60)
    # Create memory
    memory = hippocampus_bioinspired_dg_ca1.Memory(cueSize, contSize, sim)
    memory.connect_in(ILayer)
    memory.connect_out(OLayer)

    # Record spikes from output layer
    memory.DG.DGLayer.record(["spikes"])
    memory.CA1.CA1Layer.record(["spikes"])
    memory.CA3cueLayer.record(["spikes"])
    memory.CA3contLayer.record(["spikes"])
    OLayer.record(["spikes"])

    # Begin simulation
    sim.run(simTime)

    # Get spike information of the output layer
    # Get spike information
    #   - DG
    dgSpikes = memory.DG.DGLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesDG = []
    for neuron in dgSpikes:
        formatSpikesDG.append(neuron.as_array().tolist())
    #   - CA1
    ca1putSpikes = memory.CA1.CA1Layer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA1 = []
    for neuron in ca1putSpikes:
        formatSpikesCA1.append(neuron.as_array().tolist())
    #   - CA3cue
    cueSpikes = memory.CA3cueLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCue = []
    for neuron in cueSpikes:
        formatSpikesCue.append(neuron.as_array().tolist())
    #   - CA3cont
    contSpikes = memory.CA3contLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCont = []
    for neuron in contSpikes:
        formatSpikesCont.append(neuron.as_array().tolist())
    #   - OUT
    outputSpikes = OLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesOut = []
    for neuron in outputSpikes:
        formatSpikesOut.append(neuron.as_array().tolist())

    # End simulation
    sim.end()

    # Plot results
    spikes_plot([inputSpikes, formatSpikesDG, formatSpikesCue, formatSpikesCont, formatSpikesCA1, formatSpikesOut],
                ["IN", "DG", "Cue", "Cont", "CA1", "OUT"],
                ["o", "o", "o", "o", "o", "o"], ["green", "darkviolet", "orange", "goldenrod", "blue", "red"],
                ["IN", "DG", "CA3cue", "CA3cont", "CA1", "OUT"],
                "Hipocampal memory population spikes", "results/", experiment_name, False, True)

    print("Finished!")


# Plot the spike information
def spikes_plot(spikes, popNames, pointTypes, colors, labels, title, outFilePath, baseFilename, plot, write):
    plt.figure(figsize=(20, 16))

    # Add point for each neuron of each population that fire, take y labels and x labels
    populationsXValues = []
    populationsYValues = []
    globalIndex = 0
    listYticks = []
    listXticks = []
    for indexPop, populationSpikes in enumerate(spikes):
        xvalues = []
        yvalues = []
        # Assign y value (population index) and y label
        for indexNeuron, spikesSingleNeuron in enumerate(populationSpikes):
            listYticks.append(popNames[indexPop] + str(indexNeuron))
            xvalues = xvalues + spikesSingleNeuron
            yvalues = yvalues + [indexNeuron + globalIndex for i in spikesSingleNeuron]
        globalIndex = globalIndex + len(populationSpikes)
        # Add to the populations values list
        populationsXValues.append(xvalues)
        populationsYValues.append(yvalues)
        # Add xvalues to labels
        listXticks = list(set(listXticks + xvalues))
    maxXvalue = max(listXticks)
    minXvalue = min(listXticks)

    # Lines for each points
    for indexPop in range(len(spikes)):
        plt.vlines(populationsXValues[indexPop], ymin=-1, ymax=populationsYValues[indexPop], color=colors[indexPop],
                   alpha=0.1)
        plt.hlines(populationsYValues[indexPop], xmin=-1, xmax=populationsXValues[indexPop], color=colors[indexPop],
                   alpha=0.1)

    # Add spikes to scatterplot
    for indexPop in range(len(spikes)):
        plt.plot(populationsXValues[indexPop], populationsYValues[indexPop], pointTypes[indexPop],
                 color=colors[indexPop], label=labels[indexPop], markersize=10)

    # Metadata
    plt.xlabel("Simulation time (ms)", fontsize=20)
    plt.ylabel("Neuron spikes", fontsize=20)
    plt.title(title, fontsize=20)
    plt.ylim([-1, globalIndex])
    plt.xlim(-1 + minXvalue, maxXvalue + 1)
    plt.yticks(range(len(listYticks)), listYticks, fontsize=20)
    plt.legend(fontsize=20)

    # Divide xticks list in pair or odd position
    listXticks.sort()
    listXticksOdd = [int(tick) for index, tick in enumerate(listXticks) if not (index % 2 == 0)]
    listXticksPair = [int(tick) for index, tick in enumerate(listXticks) if index % 2 == 0]
    # Write them with alternate distance
    ax = plt.gca()
    ax.set_xticklabels(listXticksOdd, minor=True)
    ax.set_xticks(listXticksOdd, minor=True)
    ax.set_xticklabels(listXticksPair, minor=False)
    ax.set_xticks(listXticksPair, minor=False)
    ax.tick_params(axis='x', which='minor', pad=35)
    ax.tick_params(axis='x', which='both', labelsize=18, rotation=90)

    # Save and/or plot
    if write:
        # Check if folder exist, if not, create it
        if not os.path.isdir(outFilePath):
            os.mkdir(outFilePath)
        plt.savefig(outFilePath + baseFilename + ".png")
    if plot:
        plt.show()


if __name__ == "__main__":
    test()
