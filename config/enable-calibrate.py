import lsst.meas.multifit
import lsst.meas.extensions.multiShapelet

root.calibrate.measurement.algorithms.names |= ["multishapelet.psf", "cmodel"]
root.calibrate.measurement.slots.modelFlux = "cmodel.flux"
