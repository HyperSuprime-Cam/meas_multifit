#!/usr/bin/env python
import lsst.daf.persistence
import lsst.afw.geom
import lsst.afw.table
import lsst.afw.detection
import lsst.afw.image
import lsst.meas.multifit
import os
import argparse
import numpy


class CModelDebugger(object):

    def __init__(self, config, exposure, footprint, psf, center, moments, approxFlux=-1.0, record=None,
                 forced=False):
        self.config = config
        self.exposure = exposure
        self.footprint = footprint
        self.psf = psf
        self.center = center
        self.moments = moments
        self.approxFlux = approxFlux
        self.record = record
        self.forced = forced

    @classmethod
    def loadForced(cls, sourceID, dataRef=None, catalog="deepCoadd_forced_src", exposure="deepCoadd",
                   reference="deepCoadd_ref", config="deep_forcedPhotCoadd_config"):
        self = cls.load(sourceID, dataRef, catalog, exposure, config)
        if not isinstance(reference, lsst.afw.table.SourceCatalog):
            reference = dataRef.get(reference, immediate=True)
        self.reference = lsst.meas.multifit.CModelResult()
        for name in ("initial", "exp", "dev"):
            stage = getattr(self.reference, name)
            stage.setFlag(lsst.meas.multifit.CModelStageResult.FAILED,
                          reference.get("cmodel.%s.flux.failed" % name))
            stage.nonlinear = reference.get("cmodel.%s.nonlinear" % name)
            stage.fixed = reference.get("cmodel.%s.fixed" % name)
        self.reference.setFlag(lsst.meas.multifit.CModelResult.FAILED, reference.get("cmodel.flux.failed"))
        self.forced = True

    @classmethod
    def load(cls, sourceID, dataRef=None, catalog="deepCoadd_meas", exposure="deepCoadd",
             config="measureCoaddSources_config"):
        if not isinstance(catalog, lsst.afw.table.SourceCatalog):
            catalog = dataRef.get(catalog, immediate=True)
        record = catalog.find(sourceID)
        if not isinstance(exposure, lsst.afw.image.ExposureF):
            exposure = dataRef.get(exposure, immediate=True)
        if not isinstance(config, lsst.pex.config.Config):
            config = dataRef.get(config, immediate=True)
        if not isinstance(config, lsst.meas.algorithms.SourceMeasurementConfig):
            config = config.measurement
        image = lsst.afw.image.ImageF(os.path.join(config.algorithms["cmodel"].diagnostics.root,
                                                   "%s.fits" % sourceID))
        exposure = lsst.afw.image.ExposureF(exposure, image.getBBox(lsst.afw.image.PARENT),
                                            lsst.afw.image.PARENT, True)
        exposure.getMaskedImage().getImage().getArray()[:,:] = image.getArray()
        psfModel = lsst.meas.extensions.multiShapelet.FitPsfModel(
            config.algorithms["multishapelet.psf"].makeControl(),
            record
        )
        psf = psfModel.asMultiShapelet()
        return cls(config.algorithms["cmodel"], exposure, record.getFootprint(), psf,
                   record.getCentroid(), record.getShape(), record.getPsfFlux(), record=record)

    def run(self):
        algorithm = lsst.meas.multifit.CModelAlgorithm(self.config.makeControl())
        if self.forced:
            self.result.algorithm.applyForced(self.exposure, self.footprint, self.psf, self.center,
                                              self.reference, self.approxFlux)
        else:
            self.result = algorithm.apply(self.exposure, self.footprint, self.psf, self.center,
                                          self.moments, self.approxFlux)
        return self.result

    def evaluateModels(self):
        nPix = self.result.finalFitRegion.getArea()
        def makeArrayStruct(weighted):
            s = lsst.pipe.base.Struct(
                cmodel=lsst.pipe.base.Struct(),
                exp=lsst.pipe.base.Struct(),
                dev=lsst.pipe.base.Struct()
            )
            s.cmodel.matrix = numpy.zeros((2, nPix), dtype=numpy.float32).transpose()
            s.exp.matrix = s.cmodel.matrix[:,:1]
            s.dev.matrix = s.cmodel.matrix[:,1:]
            s.cmodel.amplitudes = numpy.array([1-self.result.fracDev, self.result.fracDev],
                                              dtype=numpy.float32)
            s.cmodel.amplitudes *= self.result.fitSysToMeasSys.flux * self.result.flux
            s.exp.amplitudes = self.result.exp.amplitudes.astype(numpy.float32)
            s.dev.amplitudes = self.result.dev.amplitudes.astype(numpy.float32)
            self.result.exp.likelihood.computeModelMatrix(s.exp.matrix, self.result.exp.nonlinear, weighted)
            self.result.dev.likelihood.computeModelMatrix(s.dev.matrix, self.result.dev.nonlinear, weighted)
            def finishStruct(struct):
                struct.model = numpy.dot(struct.matrix, struct.amplitudes)
                struct.image = lsst.afw.image.ImageF(self.exposure.getBBox(lsst.afw.image.PARENT))
                lsst.afw.detection.expandArray(self.result.finalFitRegion, struct.model,
                                               struct.image.getArray(),
                                               struct.image.getXY0())
            finishStruct(s.cmodel)
            finishStruct(s.exp)
            finishStruct(s.dev)
            return s
        self.weighted = makeArrayStruct(True)
        self.unweighted = makeArrayStruct(False)
        self.weighted.data = lsst.pipe.base.Struct(
            vector=self.result.exp.likelihood.getData(),
            image=lsst.afw.image.ImageF(self.exposure.getBBox(lsst.afw.image.PARENT))
        )
        lsst.afw.detection.expandArray(self.result.finalFitRegion, self.weighted.data.vector,
                                       self.weighted.data.image.getArray(),
                                       self.weighted.data.image.getXY0())
        self.unweighted.data = lsst.pipe.base.Struct(
            vector=numpy.zeros(nPix, dtype=numpy.float32),
            image=self.exposure.getMaskedImage().getImage()
        )
        lsst.afw.detection.flattenArray(
            self.result.finalFitRegion,
            self.unweighted.data.image.getArray(),
            self.unweighted.data.vector,
            self.exposure.getXY0()
        )
        self.variance = lsst.pipe.base.Struct(
            image=self.exposure.getMaskedImage().getVariance(),
            vector=numpy.zeros(nPix, dtype=numpy.float32)
        )
        lsst.afw.detection.flattenArray(
            self.result.finalFitRegion,
            self.variance.image.getArray(),
            self.variance.vector,
            self.exposure.getXY0()
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Rerun CModel on a source with diagnostic outputs already saved "
                     "(usually run in gdb with a breakpoint already set).")
    )
    parser.add_argument('path', type=str, help="data repository path, including rerun")
    parser.add_argument('tract', type=int, help="tract for data ID")
    parser.add_argument('patch', type=str, help="patch tuple for data ID")
    parser.add_argument('filter', type=str, help="filter for data ID")
    parser.add_argument('id', type=int, help="source ID")
    parser.add_argument('--forced', action='store_true', default=False, help="run in forced-photometry mode")
    args = parser.parse_args()
    butler = lsst.daf.persistence.Butler(args.path)
    dataRef = butler.dataRef('deepCoadd_meas', tract=args.tract, patch=args.patch, filter=args.filter)
    if args.forced:
        dbg = CModelDebugger.loadForced(sourceID=args.id, dataRef=dataRef)
    else:
        dbg = CModelDebugger.load(sourceID=args.id, dataRef=dataRef)
    dbg.run()
