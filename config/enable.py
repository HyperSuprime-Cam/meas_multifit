# Enable CModel measurements
# The 'root' should be a SourceMeasurementConfig.

import lsst.meas.multifit
import lsst.meas.extensions.multiShapelet

root.algorithms.names |= ["multishapelet.psf", "cmodel"]
root.slots.modelFlux = "cmodel.flux"
