# Disable CModel measurements
# The 'root' should be a SourceMeasurementConfig.

root.algorithms.names.discard("cmodel")
root.algorithms.names.discard("multishapelet.psf")
root.slots.modelFlux = None
