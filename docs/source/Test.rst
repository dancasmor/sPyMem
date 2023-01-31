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

* `Real-time image storage and spike activity visualisation <https://github.com/dancasmor/Real-time-spike-based-hippocampus-memory-model-for-image-storage>`_ : Application for learning and recall (from a fragment of it) of images in real time and a tool for visualising the trace of the spiking memory activity during previous operation. It employs the `hippocampus_with_forgetting <hippocampus_with_forgetting.html>`_ model and notably makes use of the real-time model, as well as new forms of dynamic visualisation of the spiking activity of the network.

(In progress to publish news applications...)
