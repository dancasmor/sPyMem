.. _test_and_applications:

Tests and applications
======================

This sections shows the tests and applications done for each hippocampus memory model to illustrate the use of its through this package.


Tests
-----

The tests are simple and serve as a proof of concept of the model, as well as to have a reference on how to work with it: inserting it into a larger network, making input/output connections, parameter tuning, taking input/output data, ...

* Basic tests for each model can be found in its `sPyMem Github <https://github.com/dancasmor/sPyMem>`_.


Applications
------------

The applications allow the performance of the memory model to be tested in real time, in real or greater complexity situations than those shown in the tests, and even embedded in other larger systems.

* `Real-time spike-based hippocampus memory model for image storage <https://github.com/dancasmor/Real-time-spike-based-hippocampus-memory-model-for-image-storage>`_: application that allows to perform learning and recall operations on 5x5 pixel black and white images on the `hippocampus_with_forgetting <hippocampus_with_forgetting.html>`_ model in real time. A graphical interface is included to allow the user to create the images and perform the appropriate operations on the memory as well as to reconstruct images based on the spiking output of the network. It also includes a second GUI that allows to visualise the spiking activity of the whole network for each time step of the simulation. This visualisation will be both graphical, by means of a diagram of the memory model at the population level, and detailed with the exact specification of which neurons in which populations generate the spikes.

* `Real-time spike-based hippocampus memory model and Posterior Parietal Cortex model for environment pseudo-mapping and navigation <https://github.com/dancasmor/Bio-inspired-spike-based-Hippocampus-and-Posterior-Parietal-Cortex-robotic-system-for-pseudo-mapping>`_: application that makes use of the `hippocampus_with_forgetting <hippocampus_with_forgetting.html>`_ model connected to a bio-inspired model of the Posterior Parietal Cortex for robotic navigation in grid-like environments and mapping of the traversed region. This application not only provides execution on virtual environments with simulated data inputs, but also the code to run it on a real Arduino-based robotic platform. In this application you can not only see the memory usage in real time, but also how to sweep to check the stored content, record the weights learned by STDP from one simulation and load them into a different simulation.

(In progress to publish new applications...)
