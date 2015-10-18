// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#include <cstdlib>

#include "boost/filesystem/path.hpp"
#include "boost/make_shared.hpp"

#include "ndarray/eigen.h"

#include "lsst/afw/detection/FootprintSet.h"
#include "lsst/afw/detection/FootprintArray.cc"
#include "lsst/afw/math/LeastSquares.h"
#include "lsst/meas/extensions/multiShapelet/FitPsf.h"
#include "lsst/meas/multifit/TruncatedGaussian.h"
#include "lsst/meas/multifit/CModel.h"
#include "lsst/meas/multifit/MultiModel.h"

namespace lsst { namespace meas { namespace multifit {


//-------------------- Utility code -------------------------------------------------------------------------

namespace {

Pixel computeFluxInFootprint(
    afw::image::Image<Pixel> const & image,
    afw::detection::Footprint const & footprint
) {
    ndarray::Array<Pixel,1,1> flat = flattenArray(footprint, image.getArray(), image.getXY0());
    // We're only using the flux to provide a scale for the problem that eases some numerical problems,
    // so for objects with SNR < 1, it's probably better to use the RMS than the flux, since the latter
    // can be negative.
    Pixel a = flat.asEigen<Eigen::ArrayXpr>().sum();
    Pixel b = std::sqrt(flat.asEigen<Eigen::ArrayXpr>().square().sum());
    return std::max(a, b);
}

} // anonymous

//-------------------- Control Objects ----------------------------------------------------------------------

PTR(Model) CModelStageControl::getModel() const {
    return Model::make(getProfile().getBasis(nComponents, maxRadius), Model::FIXED_CENTER);
}

PTR(Prior) CModelStageControl::getPrior() const {
    if (priorSource == "NONE") {
        return PTR(Prior)();
    } else if (priorSource == "FILE") {
        char const * pkgDir = std::getenv("MEAS_MULTIFIT_DIR");
        if (!pkgDir) {
            throw LSST_EXCEPT(
                pex::exceptions::IoErrorException,
                "MEAS_MULTIFIT_DIR environment variable not defined; cannot find persisted Priors"
            );
        }
        boost::filesystem::path priorPath
            = boost::filesystem::path(pkgDir)
            / boost::filesystem::path("data")
            / boost::filesystem::path(priorName + ".fits");
        PTR(Mixture) mixture = Mixture::readFits(priorPath.string());
        return boost::make_shared<MixturePrior>(mixture, "single-ellipse");
    } else if (priorSource == "CONFIG") {
        return boost::make_shared<SoftenedLinearPrior>(priorConfig);
    } else {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterException,
            "priorSource must be one of 'NONE', 'FILE', or 'CONFIG'"
        );
    }
}

PTR(algorithms::AlgorithmControl) CModelControl::_clone() const {
    return boost::make_shared<CModelControl>(*this);
}

PTR(algorithms::Algorithm) CModelControl::_makeAlgorithm(
    afw::table::Schema & schema,
    PTR(daf::base::PropertyList) const & metadata,
    algorithms::AlgorithmMap const & others,
    bool isForced
) const {
    return boost::make_shared<CModelAlgorithm>(*this, boost::ref(schema), others, isForced);
}

// ------------------- Result Objects -----------------------------------------------------------------------

CModelStageResult::CModelStageResult() :
    flux(std::numeric_limits<Scalar>::quiet_NaN()),
    fluxSigma(std::numeric_limits<Scalar>::quiet_NaN()),
    fluxInner(std::numeric_limits<Scalar>::quiet_NaN()),
    objective(std::numeric_limits<Scalar>::quiet_NaN()),
    ellipse(std::numeric_limits<Scalar>::quiet_NaN(), std::numeric_limits<Scalar>::quiet_NaN(),
            std::numeric_limits<Scalar>::quiet_NaN(), false)
{
    flags[FAILED] = true;
}

CModelResult::CModelResult() :
    flux(std::numeric_limits<Scalar>::quiet_NaN()),
    fluxSigma(std::numeric_limits<Scalar>::quiet_NaN()),
    fluxInner(std::numeric_limits<Scalar>::quiet_NaN()),
    fracDev(std::numeric_limits<Scalar>::quiet_NaN()),
    objective(std::numeric_limits<Scalar>::quiet_NaN())
{
    flags[FAILED] = true;
}


// ------------------- Key Objects for transferring to/from afw::table Records ------------------------------

namespace {

struct CModelStageKeys {

    // this constructor is used to allocate output fields in both forced and non-forced mode
    CModelStageKeys(
        Model const & model,
        afw::table::Schema & schema,
        std::string const & prefix,
        std::string const & stage,
        bool isForced,
        CModelStageControl const & ctrl
    ) :
        flux(afw::table::addFluxFields(schema, prefix + ".flux", "flux from the " + stage + " fit")),
        fluxInner(schema.addField<Scalar>(prefix + ".flux.inner",
                                          "flux within the fit region, with no extrapolation"))
    {
        if (!isForced) {
            ellipse = schema.addField<afw::table::Moments<Scalar> >(
                prefix + ".ellipse", "effective radius ellipse from the " + stage + " fit"
            );
            objective = schema.addField<Scalar>(
                prefix + ".objective", "-ln(likelihood*prior) at best-fit point for the " + stage + " fit"
            );
            nonlinear = schema.addField<afw::table::Array<Scalar> >(
                prefix + ".nonlinear", "nonlinear parameters for the " + stage + " fit",
                model.getNonlinearDim()
            );
            fixed = schema.addField<afw::table::Array<Scalar> >(
                prefix + ".fixed", "fixed parameters for the " + stage + " fit",
                model.getFixedDim()
            );
            flags[CModelStageResult::TR_SMALL] = schema.addField<afw::table::Flag>(
                prefix + ".flags.trSmall",
                "the optimizer converged because the trust radius became too small; this is a less-secure "
                "result than when the gradient is below the threshold, but usually not a problem"
            );
            flags[CModelStageResult::MAX_ITERATIONS] = schema.addField<afw::table::Flag>(
                prefix + ".flags.maxIter",
                "the optimizer hit the maximum number of iterations and did not converge"
            );
            if (ctrl.doRecordHistory) {
                nIter = schema.addField<int>(prefix + ".nIter", "Number of total iterations in stage");
            }
            if (ctrl.doRecordTime) {
                time = schema.addField<Scalar>(prefix + ".time", "Time spent in stage", "seconds");
            }
        } else {
            flags[CModelStageResult::BAD_REFERENCE] = schema.addField<afw::table::Flag>(
                prefix + ".flags.badReference",
                "The original fit in the reference catalog failed."
            );
        }
        flags[CModelStageResult::NUMERIC_ERROR] = schema.addField<afw::table::Flag>(
            prefix + ".flags.numericError",
            "numerical underflow or overflow in model evaluation; usually this means the prior was "
            "insufficient to regularize the fit, or all pixel values were zero."
        );
        flags[CModelStageResult::FAILED] = flux.flag; // these flags refer to the same underlying field
    }

    // this constructor is used to get needed keys from the reference schema in forced mode
    CModelStageKeys(
        Model const & model,
        afw::table::Schema const & schema,
        std::string const & prefix
    ) :
        flux(schema[prefix + ".flux"], schema[prefix + ".flux.err"], schema[prefix + ".flux.flags"]),
        nonlinear(schema[prefix + ".nonlinear"]),
        fixed(schema[prefix + ".fixed"])
    {
        flags[CModelStageResult::FAILED] = flux.flag; // these flags refer to the same underlying field
        LSST_THROW_IF_NE(
            model.getNonlinearDim(), nonlinear.getSize(),
            pex::exceptions::LengthErrorException,
            "Configured model nonlinear dimension (%d) does not match reference schema (%d)"
        );
        LSST_THROW_IF_NE(
            model.getFixedDim(), fixed.getSize(),
            pex::exceptions::LengthErrorException,
            "Configured model fixed dimension (%d) does not match reference schema (%d)"
        );
    }

    void copyResultToRecord(CModelStageResult const & result, afw::table::BaseRecord & record) {
        record.set(flux.meas, result.flux);
        record.set(flux.err, result.fluxSigma);
        record.set(flux.flag, result.getFlag(CModelStageResult::FAILED));
        record.set(fluxInner, result.fluxInner);
        if (objective.isValid()) {
            record.set(objective, result.objective);
        }
        if (ellipse.isValid()) {
            record.set(ellipse, result.ellipse);
        }
        if (nonlinear.isValid() && !result.nonlinear.isEmpty()) {
            record.set(nonlinear, result.nonlinear);
        }
        if (fixed.isValid() && !result.fixed.isEmpty()) {
            record.set(fixed, result.fixed);
        }
        if (nIter.isValid()) {
            record.set(nIter, result.history.size());
        }
        if (time.isValid()) {
            record.set(time, result.time);
        }
        for (int b = 0; b < CModelStageResult::N_FLAGS; ++b) {
            if (flags[b].isValid()) {
                record.set(flags[b], result.flags[b]);
            }
        }
    }

    CModelStageResult copyRecordToResult(afw::table::BaseRecord const & record) const {
        // this is only used when reading reference records, so we only transfer the fields we need for that
        CModelStageResult result;
        result.setFlag(CModelStageResult::FAILED, record.get(flags[CModelStageResult::FAILED]));
        result.nonlinear = record.get(nonlinear);
        result.fixed = record.get(fixed);
        return result;
    }

    bool hasDetailedErrorFlagSet(afw::table::BaseRecord const & record) const {
        for (int i = 2; i < CModelStageResult::N_FLAGS; ++i) {
            if (flags[i].isValid() && record.get(flags[i])) return true;
        }
        return false;
    }

    bool checkBadReferenceFlag(afw::table::BaseRecord & record) const {
        if (flags[CModelStageResult::BAD_REFERENCE].isValid()) {
            if (record.get(flags[CModelStageResult::BAD_REFERENCE])) {
                record.set(flags[CModelStageResult::FAILED], true);
                return true;
            }
        }
        return false;
    }

    afw::table::KeyTuple<afw::table::Flux> flux;
    afw::table::Key<Scalar> fluxInner;
    afw::table::Key<afw::table::Moments<Scalar> > ellipse;
    afw::table::Key<Scalar> objective;
    afw::table::Key<afw::table::Flag> flags[CModelStageResult::N_FLAGS];
    afw::table::Key<afw::table::Array<Scalar> > nonlinear;
    afw::table::Key<afw::table::Array<Scalar> > fixed;
    afw::table::Key<Scalar> time;
    afw::table::Key<int> nIter;
};

struct CModelKeys {

    // this constructor is used to allocate output fields in both forced and non-forced mode
    CModelKeys(
        Model const & initialModel, Model const & expModel, Model const & devModel,
        afw::table::Schema & schema,
        std::string const & prefix,
        bool isForced,
        CModelControl const & ctrl
    ) :
        initial(initialModel, schema, prefix + ".initial", "initial", isForced, ctrl.initial),
        exp(expModel, schema, prefix + ".exp", "exponential", isForced, ctrl.exp),
        dev(devModel, schema, prefix + ".dev", "de Vaucouleur", isForced, ctrl.dev),
        center(schema.addField<afw::table::Point<Scalar> >(
                   // The fact that the center passed to all the algorithms isn't saved by the measurement
                   // framework is a bug that will be addressed in the next version of the framework.
                   // For now, we save it ourselves so we can reproduce the conditions in the framework
                   // exactly.
                   prefix + ".center", "center position used in CModel fit", "pixels"
               )),
        flux(afw::table::addFluxFields(schema, prefix + ".flux", "flux from the final cmodel fit")),
        fluxInner(schema.addField<Scalar>(prefix + ".flux.inner",
                                          "flux within the fit region, with no extrapolation")),
        fracDev(schema.addField<Scalar>(prefix + ".fracDev", "fraction of flux in de Vaucouleur component")),
        objective(schema.addField<Scalar>(prefix + ".objective", "-ln(likelihood) (chi^2) in cmodel fit"))
    {
        try {
            kronRadius = schema["flux.kron.radius"];
        } catch (pex::exceptions::NotFoundException &) {
            // we'll fall back to other options if Kron radius is not available.
        }
        flags[CModelResult::FAILED] = flux.flag; // these keys refer to the same underlying field
        flags[CModelResult::REGION_MAX_AREA] = schema.addField<afw::table::Flag>(
            prefix + ".flags.region.maxArea",
            "number of pixels in fit region exceeded the region.maxArea value."
        );
        flags[CModelResult::REGION_MAX_BAD_PIXEL_FRACTION] = schema.addField<afw::table::Flag>(
            prefix + ".flags.region.maxBadPixelFraction",
            "the fraction of bad/clipped pixels in the fit region exceeded region.maxBadPixelFraction"
        );
        if (!isForced) {
            flags[CModelResult::REGION_USED_FOOTPRINT_AREA] = schema.addField<afw::table::Flag>(
                prefix + ".flags.region.usedFootprintArea",
                "the pixel region for the initial fit was defined by the area of the Footprint"
            );
            flags[CModelResult::REGION_USED_PSF_AREA] = schema.addField<afw::table::Flag>(
                prefix + ".flags.region.usedPsfArea",
                "the pixel region for the initial fit was set to a fixed factor of the PSF area"
            );
            flags[CModelResult::REGION_USED_INITIAL_ELLIPSE_MIN] = schema.addField<afw::table::Flag>(
                prefix + ".flags.region.usedInitialEllipseMin",
                "the pixel region for the final fit was set to the lower bound defined by the initial fit"
            );
            flags[CModelResult::REGION_USED_INITIAL_ELLIPSE_MAX] = schema.addField<afw::table::Flag>(
                prefix + ".flags.region.usedInitialEllipseMax",
                "the pixel region for the final fit was set to the upper bound defined by the initial fit"
            );
            flags[CModelResult::NO_SHAPE] = schema.addField<afw::table::Flag>(
                prefix + ".flags.noShape",
                "the shape slot needed to initialize the parameters failed or was not defined"
            );
            flags[CModelResult::SMALL_SHAPE] = schema.addField<afw::table::Flag>(
                prefix + ".flags.smallShape",
                (boost::format(
                    "initial parameter guess resulted in negative radius; used minimum of %f pixels instead."
                ) % ctrl.minInitialRadius).str()
            );
            ellipse = schema.addField<afw::table::Moments<Scalar> >(
                prefix + ".ellipse", "fracDev-weighted average of exp.ellipse and dev.ellipse"
            );
            initialFitRegion = schema.addField< afw::table::Moments<double> >(
                prefix + ".region.initial.ellipse",
                "ellipse used to set the pixel region for the initial fit (before applying bad pixel mask)"
            );
            finalFitRegion = schema.addField< afw::table::Moments<double> >(
                prefix + ".region.final.ellipse",
                "ellipse used to set the pixel region for the final fit (before applying bad pixel mask)"
            );
        } else {
            flags[CModelResult::BAD_REFERENCE] = schema.addField<afw::table::Flag>(
                prefix + ".flags.badReference",
                "The original fit in the reference catalog failed."
            );
        }
        flags[CModelResult::NO_PSF] = schema.addField<afw::table::Flag>(
            prefix + ".flags.noPsf",
            "the multishapelet fit to the PSF model did not succeed"
        );
        flags[CModelResult::NO_WCS] = schema.addField<afw::table::Flag>(
            prefix + ".flags.noWcs",
            "input exposure has no world coordinate system information"
        );
        flags[CModelResult::NO_CALIB] = schema.addField<afw::table::Flag>(
            prefix + ".flags.noCalib",
            "input exposure has no photometric calibration information"
        );
        flags[CModelResult::BAD_CENTROID] = schema.addField<afw::table::Flag>(
            prefix + ".flags.badCentroid",
            "input centroid was not within the fit region (probably because it's not within the Footprint)"
        );
    }

    // this constructor is used to get needed keys from the reference schema in forced mode
    CModelKeys(
        Model const & initialModel, Model const & expModel, Model const & devModel,
        afw::table::Schema const & schema,
        std::string const & prefix
    ) :
        initial(initialModel, schema, prefix + ".initial"),
        exp(expModel, schema, prefix + ".exp"),
        dev(devModel, schema, prefix + ".dev"),
        center(schema[prefix + ".center"]),
        initialFitRegion(schema[prefix + ".region.initial.ellipse"]),
        finalFitRegion(schema[prefix + ".region.final.ellipse"])
    {
        flags[CModelStageResult::FAILED] = schema[prefix + ".flux.flags"];
    }

    void copyResultToRecord(CModelResult const & result, afw::table::BaseRecord & record) {
        initial.copyResultToRecord(result.initial, record);
        exp.copyResultToRecord(result.exp, record);
        dev.copyResultToRecord(result.dev, record);
        if (ellipse.isValid()) {
            double u = 1.0 - result.fracDev;
            double v = result.fracDev;
            record.set(ellipse.getIxx(), u*result.exp.ellipse.getIxx() + v*result.dev.ellipse.getIxx());
            record.set(ellipse.getIyy(), u*result.exp.ellipse.getIyy() + v*result.dev.ellipse.getIyy());
            record.set(ellipse.getIxy(), u*result.exp.ellipse.getIxy() + v*result.dev.ellipse.getIxy());
        }
        record.set(flux.meas, result.flux);
        record.set(flux.err, result.fluxSigma);
        record.set(fluxInner, result.fluxInner);
        record.set(fracDev, result.fracDev);
        record.set(objective, result.objective);
        if (initialFitRegion.isValid()) {
            record.set(initialFitRegion, result.initialFitRegion);
        }
        if (finalFitRegion.isValid()) {
            record.set(finalFitRegion, result.finalFitRegion);
        }
        for (int b = 0; b < CModelResult::N_FLAGS; ++b) {
            if (flags[b].isValid()) {
                record.set(flags[b], result.flags[b]);
            }
        }
    }

    CModelResult copyRecordToResult(afw::table::BaseRecord const & record) const {
        // this is only used when reading reference records, so we only transfer the fields we need for that
        CModelResult result;
        result.initial = initial.copyRecordToResult(record);
        result.exp = exp.copyRecordToResult(record);
        result.dev = dev.copyRecordToResult(record);
        result.initialFitRegion = record.get(initialFitRegion);
        result.finalFitRegion = record.get(finalFitRegion);
        result.setFlag(CModelResult::FAILED, record.get(flags[CModelResult::FAILED]));
        return result;
    }

    bool hasDetailedErrorFlagSet(afw::table::BaseRecord const & record) const {
        for (int i = 1; i < CModelResult::N_FLAGS; ++i) {
            if (flags[i].isValid() && record.get(flags[i])) return true;
        }
        if (initial.hasDetailedErrorFlagSet(record)) return true;
        if (exp.hasDetailedErrorFlagSet(record)) return true;
        if (dev.hasDetailedErrorFlagSet(record)) return true;
        return false;
    }

    void checkBadReferenceFlag(afw::table::BaseRecord & record) const {
        if (flags[CModelResult::BAD_REFERENCE].isValid()) {
            // if any of the per-stage BAD_REFERENCE flags is set, the main one should be.
            record.set(
                flags[CModelResult::BAD_REFERENCE],
                record.get(flags[CModelResult::BAD_REFERENCE]) ||
                initial.checkBadReferenceFlag(record) ||
                exp.checkBadReferenceFlag(record) ||
                dev.checkBadReferenceFlag(record)
            );
            // if the main BAD_REFERENCE flag is set, the FAILED flag should be as well.
            if (record.get(flags[CModelResult::BAD_REFERENCE])) {
                record.set(flags[CModelResult::FAILED], true);
            }
        }
    }

    CModelStageKeys initial;
    CModelStageKeys exp;
    CModelStageKeys dev;
    afw::table::Key<afw::table::Point<Scalar> > center;
    afw::table::KeyTuple<afw::table::Flux> flux;
    afw::table::Key<Scalar> fluxInner;
    afw::table::Key<Scalar> fracDev;
    afw::table::Key<Scalar> objective;
    afw::table::Key<afw::table::Moments<Scalar> > initialFitRegion;
    afw::table::Key<afw::table::Moments<Scalar> > finalFitRegion;
    afw::table::Key<afw::table::Moments<Scalar> > ellipse;
    afw::table::Key<afw::table::Flag> flags[CModelResult::N_FLAGS];
    afw::table::Key<float> kronRadius; // input (for fit region determination)
};

} // anonymous

// ------------------- CModelStageData: per-object data we pass around together a lot -----------------------

namespace {

struct CModelStageData {
    afw::geom::Point2D measSysCenter;
    PTR(afw::coord::Coord) position;
    UnitSystem measSys;
    UnitSystem fitSys;
    LocalUnitTransform fitSysToMeasSys;
    ndarray::Array<Scalar,1,1> parameters;
    ndarray::Array<Scalar,1,1> nonlinear;
    ndarray::Array<Scalar,1,1> amplitudes;
    ndarray::Array<Scalar,1,1> fixed;
    shapelet::MultiShapeletFunction psf;

    CModelStageData(
        afw::image::Exposure<Pixel> const & exposure,
        Scalar approxFlux, afw::geom::Point2D const & center,
        shapelet::MultiShapeletFunction const & psf_,
        Model const & model
    ) :
        measSysCenter(center), position(exposure.getWcs()->pixelToSky(center)),
        measSys(exposure), fitSys(*position, exposure.getCalib()->getMagnitude(approxFlux)),
        fitSysToMeasSys(*position, fitSys, measSys),
        parameters(ndarray::allocate(model.getNonlinearDim() + model.getAmplitudeDim())),
        nonlinear(parameters[ndarray::view(0, model.getNonlinearDim())]),
        amplitudes(parameters[ndarray::view(model.getNonlinearDim(), parameters.getSize<0>())]),
        fixed(ndarray::allocate(model.getFixedDim())),
        psf(psf_)
    {}

    CModelStageData changeModel(Model const & model) const {
        // If we allow centroids to vary in some stages and not others, this will resize the parameter
        // arrays and update them accordingly.  For now we just assert that dimensions haven't changed
        // and do a deep-copy.
        // In theory, we should also assert that the ellipse parametrizations haven't changed, but that
        // assert would be too much work to be worthwhile.
        assert(model.getNonlinearDim() == nonlinear.getSize<0>());
        assert(model.getAmplitudeDim() == amplitudes.getSize<0>());
        assert(model.getFixedDim() == fixed.getSize<0>());
        CModelStageData r(*this);
        r.parameters = ndarray::copy(parameters);
        r.nonlinear = r.parameters[ndarray::view(0, model.getNonlinearDim())];
        r.amplitudes = r.parameters[ndarray::view(model.getNonlinearDim(), parameters.getSize<0>())];
        // don't need to deep-copy fixed parameters because they're, well, fixed
        return r;
    }

};

} // anonymous

// ------------------- Private Implementation objects -------------------------------------------------------

namespace {

ndarray::Array<Pixel,2,-1> makeModelMatrix(
    Likelihood const & likelihood,
    ndarray::Array<Scalar const,1,1> const & nonlinear
) {
    ndarray::Array<Pixel,2,2> modelMatrixT
        = ndarray::allocate(likelihood.getAmplitudeDim(), likelihood.getDataDim());
    ndarray::Array<Pixel,2,-1> modelMatrix = modelMatrixT.transpose();
    likelihood.computeModelMatrix(modelMatrix, nonlinear, false);
    return modelMatrix;
}


struct WeightSums {

    WeightSums(
        ndarray::Array<Pixel const,2,-1> const & modelMatrix,
        ndarray::Array<Pixel const,1,1> const & data,
        ndarray::Array<Pixel const,1,1> const & variance
    ) : fluxInner(0.0), fluxVar(0.0), norm(0.0)
    {
        assert(modelMatrix.getSize<1>() == 1);
        run(modelMatrix.transpose()[0].asEigen<Eigen::ArrayXpr>(),
            data.asEigen<Eigen::ArrayXpr>(),
            variance.asEigen<Eigen::ArrayXpr>());
    }

    WeightSums(
        ndarray::Array<Pixel const,1,1> const & model,
        ndarray::Array<Pixel const,1,1> const & data,
        ndarray::Array<Pixel const,1,1> const & variance
    ) : fluxInner(0.0), fluxVar(0.0), norm(0.0)
    {
        run(model.asEigen<Eigen::ArrayXpr>(),
            data.asEigen<Eigen::ArrayXpr>(),
            variance.asEigen<Eigen::ArrayXpr>());
    }

    void run(
        ndarray::EigenView<Pixel const,1,1,Eigen::ArrayXpr> const & model,
        ndarray::EigenView<Pixel const,1,1,Eigen::ArrayXpr> const & data,
        ndarray::EigenView<Pixel const,1,1,Eigen::ArrayXpr> const & variance
    ) {
        double w = model.sum();
        double wd = (model*data).sum();
        double ww = model.square().sum();
        double wwv = (model.square()*variance).sum();
        norm = w/ww;
        fluxInner = wd*norm;
        fluxVar = wwv*norm*norm;
    }

    double fluxInner;
    double fluxVar;
    double norm;
};


class CModelStageImpl {
public:
    shapelet::RadialProfile const * profile;
    PTR(Model) model;
    PTR(Prior) prior;
    mutable Model::EllipseVector ellipses;
    PTR(afw::table::BaseTable) historyTable;
    PTR(OptimizerHistoryRecorder) historyRecorder;

    explicit CModelStageImpl(CModelStageControl const & ctrl) :
        profile(&ctrl.getProfile()),
        model(ctrl.getModel()),
        prior(ctrl.getPrior()),
        ellipses(model->makeEllipseVector())
    {
        if (ctrl.doRecordHistory) {
            afw::table::Schema historySchema;
            historyRecorder.reset(new OptimizerHistoryRecorder(historySchema, model, true));
            historyTable = afw::table::BaseTable::make(historySchema);
        }
    }

    CModelStageResult makeResult() const {
        CModelStageResult result;
        result.model = model;
        result.prior = prior;
        return result;
    }

    void fillResult(
        CModelStageResult & result,
        CModelStageData const & data,
        WeightSums const & sums
    ) const {
        // these are shallow assignments
        result.nonlinear = data.nonlinear;
        result.amplitudes = data.amplitudes;
        result.fixed = data.fixed;
        // flux is just the amplitude converted from fitSys to measSys
        result.flux = data.amplitudes[0] * data.fitSysToMeasSys.flux;
        result.fluxInner = sums.fluxInner;
        result.fluxSigma = std::sqrt(sums.fluxVar)*result.flux/result.fluxInner;
        // to compute the ellipse, we need to first read the nonlinear parameters into the workspace
        // ellipse vector, then transform from fitSys to measSys.
        model->writeEllipses(data.nonlinear.begin(), data.fixed.begin(), ellipses.begin());
        result.ellipse = ellipses.front().getCore().transform(data.fitSysToMeasSys.geometric.getLinear());
    }

    void fit(
        CModelStageControl const & ctrl, CModelStageResult & result, CModelStageData const & data,
        afw::image::Exposure<Pixel> const & exposure, afw::detection::Footprint const & footprint
    ) const {
        long long startTime = 0;
        if (ctrl.doRecordTime) {
            startTime = daf::base::DateTime::now().nsecs();
        }
        result.likelihood = boost::make_shared<UnitTransformedLikelihood>(
            model, data.fixed, data.fitSys, *data.position,
            exposure, footprint, data.psf, UnitTransformedLikelihoodControl(ctrl.usePixelWeights)
        );
        PTR(OptimizerObjective) objective = OptimizerObjective::makeFromLikelihood(result.likelihood, prior);
        result.objfunc = objective;
        Optimizer optimizer(objective, data.parameters, ctrl.optimizer);
        try {
            if (ctrl.doRecordHistory) {
                result.history = afw::table::BaseCatalog(historyTable);
                optimizer.run(*historyRecorder, result.history);
            } else {
                optimizer.run();
            }
        } catch (std::overflow_error &) {
            result.setFlag(CModelStageResult::NUMERIC_ERROR, true);
        } catch (std::underflow_error &) {
            result.setFlag(CModelStageResult::NUMERIC_ERROR, true);
        } catch (pex::exceptions::UnderflowErrorException &) {
            result.setFlag(CModelStageResult::NUMERIC_ERROR, true);
        } catch (pex::exceptions::OverflowErrorException &) {
            result.setFlag(CModelStageResult::NUMERIC_ERROR, true);
        }

        // Use the optimizer state to set flags.  There's more information in the state than we
        // report in the result, but it's only useful for debugging, and for that the user should
        // look at the history by running outside of plugin mode.
        int state = optimizer.getState();
        if (state & Optimizer::FAILED) {
            result.setFlag(CModelStageResult::FAILED, true);
            if (state & Optimizer::FAILED_MAX_ITERATIONS) {
                result.setFlag(CModelStageResult::MAX_ITERATIONS, true);
            } else if (state & Optimizer::FAILED_NAN) {
                result.setFlag(CModelStageResult::NUMERIC_ERROR, true);
            }
        } else {
            result.setFlag(CModelStageResult::FAILED, false);
            if (state & Optimizer::CONVERGED_TR_SMALL) {
                result.setFlag(CModelStageResult::TR_SMALL, true);
            }
        }

        result.objective = optimizer.getObjectiveValue();

        // Set the output parameter vectors.  We deep-assign to the data object to split nonlinear and
        // amplitudes, then shallow-assign these to the result object.
        data.parameters.deep() = optimizer.getParameters(); // sets nonlinear and amplitudes - they are views

        // This flux uncertainty is computed holding all the nonlinear parameters fixed, and treating
        // the best-fit model as a continuous aperture.  That's likely what we'd want for colors, but it
        // underestimates the statistical uncertainty on the total flux (though that's probably dominated by
        // systematic errors anyway).
        ndarray::Array<Pixel,2,-1> modelMatrix = makeModelMatrix(*result.likelihood, data.nonlinear);
        WeightSums sums(
            modelMatrix,
            result.likelihood->getUnweightedData(),
            result.likelihood->getVariance()
        );

        // If we're using per-pixel variances, we need to do another linear fit without them, since
        // using per-pixel variances there can cause magnitude-dependent biases in the flux.
        // (We're not sure if using per-pixel variances in the nonlinear fit can do that).
        if (ctrl.usePixelWeights) {
            afw::math::LeastSquares lstsq = afw::math::LeastSquares::fromDesignMatrix(
                modelMatrix,
                result.likelihood->getUnweightedData()
            );
            data.amplitudes.deep() = lstsq.getSolution();
        }

        // Set parameter vectors, flux values, ellipse on result.
        fillResult(result, data, sums);

        if (ctrl.doRecordTime) {
            result.time = (daf::base::DateTime::now().nsecs() - startTime)/1E9;
        }
    }

    void fitLinear(
        CModelStageControl const & ctrl, CModelStageResult & result, CModelStageData const & data,
        afw::image::Exposure<Pixel> const & exposure, afw::detection::Footprint const & footprint
    ) const {
        result.likelihood = boost::make_shared<UnitTransformedLikelihood>(
            model, data.fixed, data.fitSys, *data.position,
            exposure, footprint, data.psf, UnitTransformedLikelihoodControl(ctrl.usePixelWeights)
        );
        ndarray::Array<Pixel,2,-1> modelMatrix = makeModelMatrix(*result.likelihood, data.nonlinear);
        afw::math::LeastSquares lstsq = afw::math::LeastSquares::fromDesignMatrix(
            modelMatrix,
            result.likelihood->getUnweightedData()
        );
        data.amplitudes.deep() = lstsq.getSolution();
        result.objective
            = 0.5*(
                result.likelihood->getUnweightedData().asEigen().cast<Scalar>()
                - modelMatrix.asEigen().cast<Scalar>() * lstsq.getSolution().asEigen()
            ).squaredNorm();

        WeightSums sums(modelMatrix, result.likelihood->getUnweightedData(), result.likelihood->getVariance());

        fillResult(result, data, sums);
        result.setFlag(CModelStageResult::FAILED, false);
    }

};

} // anonymous


class CModelAlgorithm::Impl {
public:

    explicit Impl(CModelControl const & ctrl) :
        initial(ctrl.initial), exp(ctrl.exp), dev(ctrl.dev)
    {
        // construct linear combination model
        ModelVector components(2);
        components[0] = exp.model;
        components[1] = dev.model;
        Model::NameVector prefixes(2);
        prefixes[0] = "exp";
        prefixes[1] = "dev";
        model = boost::make_shared<MultiModel>(components, prefixes);
    }

    CModelStageImpl initial;
    CModelStageImpl exp;
    CModelStageImpl dev;
    PTR(Model) model;
    PTR(CModelKeys) keys;
    PTR(CModelKeys) refKeys;
    PTR(extensions::multiShapelet::FitPsfControl const) fitPsfCtrl;

    CModelResult makeResult() const {
        CModelResult result;
        result.initial = initial.makeResult();
        result.exp = exp.makeResult();
        result.dev = dev.makeResult();
        return result;
    }

    void fitLinear(
        CModelControl const & ctrl, CModelResult & result,
        CModelStageData const & expData, CModelStageData const & devData,
        afw::image::Exposure<Pixel> const & exposure, afw::detection::Footprint const & footprint
    ) const {
        // concatenate exp and dev parameter arrays to make parameter arrays for combined model
        ndarray::Array<Scalar,1,1> nonlinear = ndarray::allocate(model->getNonlinearDim());
        nonlinear[ndarray::view(0, exp.model->getNonlinearDim())] = expData.nonlinear;
        nonlinear[ndarray::view(exp.model->getNonlinearDim(), model->getNonlinearDim())] = devData.nonlinear;
        ndarray::Array<Scalar,1,1> fixed = ndarray::allocate(model->getFixedDim());
        fixed[ndarray::view(0, exp.model->getFixedDim())] = expData.fixed;
        fixed[ndarray::view(exp.model->getFixedDim(), model->getFixedDim())] = devData.fixed;

        UnitTransformedLikelihood likelihood(
            model, fixed, expData.fitSys, *expData.position,
            exposure, footprint, expData.psf, UnitTransformedLikelihoodControl(false)
        );
        ndarray::Array<Pixel,2,-1> modelMatrix = makeModelMatrix(likelihood, nonlinear);
        Vector gradient = -(modelMatrix.asEigen().adjoint() *
            likelihood.getUnweightedData().asEigen()).cast<Scalar>();
        Matrix hessian = Matrix::Zero(likelihood.getAmplitudeDim(), likelihood.getAmplitudeDim());
        hessian.selfadjointView<Eigen::Lower>().rankUpdate(modelMatrix.asEigen().adjoint().cast<Scalar>());
        Scalar q0 = 0.5*likelihood.getUnweightedData().asEigen().squaredNorm();

        // Use truncated Gaussian to compute the maximum-likelihood amplitudes with the constraint
        // that all amplitude must be >= 0
        TruncatedGaussian tg = TruncatedGaussian::fromSeriesParameters(q0, gradient, hessian);
        Vector amplitudes = tg.maximize();
        result.flux = expData.fitSysToMeasSys.flux * amplitudes.sum();

        // To compute the error on the flux, we treat the best-fit composite profile as a continuous
        // aperture and compute the uncertainty on that aperture flux.
        // That means this is an underestimate of the true uncertainty, but it's the sort that kind of
        // makes sense for colors, and it's consistent with the fact that we're also ignoring the
        // uncertainty in the nonlinear parameters.  It also makes this uncertainty equivalent to the
        // PSF flux uncertainty and the single-component exp or dev uncertainty when fitting point
        // sources, which is convenient, even if it's not statistically correct.
        // Doing a better job would involve taking into account that we have positivity constraints
        // on the two components, which means the actual uncertainty is neither Gaussian nor symmetric,
        // which is a lot harder to compute and a lot harder to use.
        ndarray::Array<Pixel,1,1> model = ndarray::allocate(likelihood.getDataDim());
        model.asEigen() = modelMatrix.asEigen() * amplitudes.cast<Pixel>();
        WeightSums sums(model, likelihood.getUnweightedData(), likelihood.getVariance());
        result.fluxInner = sums.fluxInner;
        result.fluxSigma = std::sqrt(sums.fluxVar)*result.flux/result.fluxInner;
        result.setFlag(CModelResult::FAILED, false);
        result.fracDev = amplitudes[1] / amplitudes.sum();
        result.objective = tg.evaluateLog()(amplitudes);
    }

    void guessParametersFromMoments(
        CModelControl const & ctrl, CModelStageData & data,
        afw::geom::ellipses::Quadrupole const & moments,
        CModelResult & result
    ) const {
        afw::geom::ellipses::Ellipse psfEllipse = data.psf.evaluate().computeMoments();
        // Deconvolve the moments ellipse, with a floor to keep the result from
        // having moments <= 0
        Scalar const mir2 = ctrl.minInitialRadius * ctrl.minInitialRadius;
        Scalar ixx = mir2, iyy = mir2, ixy = 0.0;
        try {
            afw::geom::ellipses::Quadrupole psfMoments(psfEllipse.getCore());
            ixx = std::max(moments.getIxx() - psfMoments.getIxx(), mir2);
            iyy = std::max(moments.getIyy() - psfMoments.getIyy(), mir2);
            ixy = moments.getIxy() - psfMoments.getIxy();
        } catch (pex::exceptions::InvalidParameterException &) {
            // let ixx, iyy, ixy stay at initial minimum values
            result.setFlag(CModelResult::SMALL_SHAPE, true); // set this now, unset it on success later
        }
        if (ixx*iyy < ixy*ixy) {
            ixy = 0.0;
            result.setFlag(CModelResult::SMALL_SHAPE, true); // set this now, unset it on success later
        }
        afw::geom::ellipses::Quadrupole deconvolvedMoments(ixx, iyy, ixy, false);
        try {
            deconvolvedMoments.normalize();
        } catch (pex::exceptions::InvalidParameterException &) {
            deconvolvedMoments = afw::geom::ellipses::Quadrupole(mir2, mir2, 0.0);
            result.setFlag(CModelResult::SMALL_SHAPE, true);
        }
        afw::geom::ellipses::Ellipse deconvolvedEllipse(
            deconvolvedMoments,
            afw::geom::Point2D(data.measSysCenter - psfEllipse.getCenter())
        );
        // Convert ellipse from moments to half-light using the ratio for this profile
        deconvolvedEllipse.getCore().scale(1.0 / initial.profile->getMomentsRadiusFactor());
        // Transform the deconvolved ellipse from MeasSys to FitSys
        deconvolvedEllipse.transform(data.fitSysToMeasSys.geometric.invert()).inPlace();
        // Convert to the ellipse parametrization used by the Model (assigning to an ellipse converts
        // between parametrizations)
        assert(initial.ellipses.size() == 1u); // should be true of all Models that come from RadialProfiles
        initial.ellipses.front() = deconvolvedEllipse;

        // Read the ellipse into the nonlinear and fixed parameters.
        initial.model->readEllipses(initial.ellipses.begin(), data.nonlinear.begin(), data.fixed.begin());

        // Set the initial amplitude (a.k.a. flux) to 1: recall that in FitSys, this is approximately correct
        assert(data.amplitudes.getSize<0>() == 1); // should be true of all Models from RadialProfiles
        data.amplitudes[0] = 1.0;

        // Ensure the initial parameters are compatible with the prior
        if (initial.prior && initial.prior->evaluate(data.nonlinear, data.amplitudes) == 0.0) {
            initial.ellipses.front().setCore(afw::geom::ellipses::Quadrupole(mir2, mir2, 0.0));
            initial.model->readEllipses(initial.ellipses.begin(), data.nonlinear.begin(), data.fixed.begin());
            if (initial.prior->evaluate(data.nonlinear, data.amplitudes) == 0.0) {
                throw LSST_EXCEPT(
                    pex::exceptions::LogicErrorException,
                    "minInitialRadius is incompatible with prior"
                );
            }
        }
    }

    // Check that if we've set the general error flag, we have a detailed flag to explain it.
    // If we don't, log a warning.
    void checkFlagDetails(afw::table::SourceRecord & record) const {
        // The BAD_REFERENCE flag should always imply general failure, even if we attempted to
        // proceed (because the results should not be trusted).  But we set general failure to true
        // at the beginning so it's set if an unexpected exception is thrown, and then we unset it
        // when the optimizer succeeds, so we have to make sure BAD_REFERENCE implies FAILED
        // here.
        // We also guarantee that the per-stage BAD_REFERENCE flags also imply the main one.
        keys->checkBadReferenceFlag(record);
        // Check for unflagged NaNs.  Warn if we see any so we can fix the underlying problem, and
        // then flag them anyway.
        if (lsst::utils::isnan(record.get(keys->flux.meas))
            && !record.get(keys->flags[CModelResult::FAILED])
        ) {
            pex::logging::Log::getDefaultLog().log(
                pex::logging::Log::WARN,
                (boost::format(
                    "Unflagged NaN detected for source %s; please report this as a bug in CModel"
                ) % record.getId()).str()
            );
            // Now that we've warned about it, go ahead and set the flag.
            record.set(keys->flags[CModelResult::FAILED], true);
        }
        // Look for cases where we've set the general failure flag with no specific error flag.
        if (!record.get(keys->flags[CModelResult::FAILED])) return;
        if (keys->hasDetailedErrorFlagSet(record)) return;
        pex::logging::Log::getDefaultLog().log(
            pex::logging::Log::WARN,
            (boost::format(
                "Error unexplained by flags detected for source %s; please report this as a bug in CModel"
            ) % record.getId()).str()
        );
    }

};

// ------------------- CModelAlgorithm itself ---------------------------------------------------------------

CModelAlgorithm::CModelAlgorithm(
    Control const & ctrl,
    afw::table::Schema & schema,
    algorithms::AlgorithmMap const & others,
    bool isForced
) : algorithms::Algorithm(ctrl), _impl(new Impl(ctrl))
{
    _impl->keys = boost::make_shared<CModelKeys>(
        *_impl->initial.model, *_impl->exp.model, *_impl->dev.model,
        boost::ref(schema), ctrl.name, isForced, ctrl
    );
    // Ideally we'd like to initalize refKeys here too when isForced==true, but we aren't passed the
    // refSchema here, so instead we'll construct that on first use.  This will be fixed in the next
    // version of the measurement framework that's in progress on the LSST side.

    algorithms::AlgorithmMap::const_iterator i = others.find(ctrl.psfName);
    if (i != others.end()) {
        // Not finding the PSF is now a non-fatal error at this point, because in Jose's use case, we
        // don't need it here.  We'll throw later if it's missing.
        _impl->fitPsfCtrl = boost::dynamic_pointer_cast<extensions::multiShapelet::FitPsfControl const>(
            i->second->getControl().clone()
        );
    }
}

CModelAlgorithm::CModelAlgorithm(Control const & ctrl) :
    algorithms::Algorithm(ctrl), _impl(new Impl(ctrl))
{}

CModelAlgorithm::Result CModelAlgorithm::apply(
    afw::image::Exposure<Pixel> const & exposure,
    shapelet::MultiShapeletFunction const & psf,
    afw::geom::Point2D const & center,
    afw::geom::ellipses::Quadrupole const & moments,
    Scalar approxFlux,
    Scalar kronRadius,
    int footprintArea
) const {
    Result result = _impl->makeResult();
    _applyImpl(result, exposure, psf, center, moments, approxFlux, kronRadius, footprintArea);
    return result;
}


void CModelAlgorithm::_applyImpl(
    Result & result,
    afw::image::Exposure<Pixel> const & exposure,
    shapelet::MultiShapeletFunction const & psf,
    afw::geom::Point2D const & center,
    afw::geom::ellipses::Quadrupole const & moments,
    Scalar approxFlux,
    Scalar kronRadius,
    int footprintArea
) const {

    afw::geom::ellipses::Quadrupole psfMoments = psf.evaluate().computeMoments().getCore();

    PixelFitRegion region(getControl().region, moments, psfMoments, kronRadius, footprintArea);
    result.initialFitRegion = region.ellipse;
    region.applyMask(*exposure.getMaskedImage().getMask(), center);
    result.setFlag(CModelResult::REGION_MAX_AREA, region.maxArea);
    result.setFlag(CModelResult::REGION_MAX_BAD_PIXEL_FRACTION, region.maxBadPixelFraction);
    result.setFlag(CModelResult::REGION_USED_FOOTPRINT_AREA, region.usedFootprintArea);
    result.setFlag(CModelResult::REGION_USED_PSF_AREA, region.usedPsfArea);
    if (!region.footprint) return;

    // Negative approxFlux means we should come up with an estimate ourselves.
    // This is only used to avoid scaling problems in the optimizer, so it doesn't have to be very good.
    if (!(approxFlux > 0.0)) {
        approxFlux = computeFluxInFootprint(*exposure.getMaskedImage().getImage(), *region.footprint);
        if (!(approxFlux > 0.0)) {
            // This is only be possible if the object has all data pixels set to zero or
            // if there are unmasked NaNs in the fit region.
            result.initial.setFlag(CModelStageResult::NUMERIC_ERROR, true);
            result.initial.setFlag(CModelStageResult::FAILED, true);
            result.exp.setFlag(CModelStageResult::NUMERIC_ERROR, true);
            result.exp.setFlag(CModelStageResult::FAILED, true);
            result.dev.setFlag(CModelStageResult::NUMERIC_ERROR, true);
            result.dev.setFlag(CModelStageResult::FAILED, true);
            result.setFlag(CModelResult::FAILED, true);
            return;
        }
    }

    // Set up coordinate systems and empty parameter vectors
    CModelStageData initialData(exposure, approxFlux, center, psf, *_impl->initial.model);
    result.fitSysToMeasSys = initialData.fitSysToMeasSys;

    // Initialize the parameter vectors by doing deconvolving the moments
    _impl->guessParametersFromMoments(getControl(), initialData, moments, result);

    // Do the initial fit
    // TODO: use only 0th-order terms in psf
    _impl->initial.fit(getControl().initial, result.initial, initialData, exposure, *region.footprint);
    if (result.initial.getFlag(CModelStageResult::FAILED)) return;

    // Include a multiple of the initial-fit ellipse in the footprint, re-do clipping
    result.initial.model->writeEllipses(initialData.nonlinear.begin(), initialData.fixed.begin(),
                                        _impl->initial.ellipses.begin());
    _impl->initial.ellipses.front().transform(initialData.fitSysToMeasSys.geometric).inPlace();
    region.applyEllipse(_impl->initial.ellipses.front().getCore(), psfMoments);
    result.finalFitRegion = region.ellipse;
    region.applyMask(*exposure.getMaskedImage().getMask(), center);
    // It's okay to "override" these flags, because we'd have already returned early if they were set above.
    result.setFlag(CModelResult::REGION_MAX_AREA, region.maxArea);
    result.setFlag(CModelResult::REGION_MAX_BAD_PIXEL_FRACTION, region.maxBadPixelFraction);
    result.setFlag(CModelResult::REGION_USED_INITIAL_ELLIPSE_MIN, region.usedMinEllipse);
    result.setFlag(CModelResult::REGION_USED_INITIAL_ELLIPSE_MAX, region.usedMaxEllipse);
    if (!region.footprint) return;

    // Do the exponential fit
    CModelStageData expData = initialData.changeModel(*_impl->exp.model);
    _impl->exp.fit(getControl().exp, result.exp, expData, exposure, *region.footprint);

    // Do the de Vaucouleur fit
    CModelStageData devData = initialData.changeModel(*_impl->dev.model);
    _impl->dev.fit(getControl().dev, result.dev, devData, exposure, *region.footprint);

    if (result.exp.getFlag(CModelStageResult::FAILED) ||result.dev.getFlag(CModelStageResult::FAILED))
        return;

    // Do the linear combination fit
    try {
        _impl->fitLinear(getControl(), result, expData, devData, exposure, *region.footprint);
    } catch (...) {
        result.setFlag(CModelResult::FAILED, true);
        throw;
    }
}

CModelAlgorithm::Result CModelAlgorithm::applyForced(
    afw::image::Exposure<Pixel> const & exposure,
    shapelet::MultiShapeletFunction const & psf,
    afw::geom::Point2D const & center,
    CModelResult const & reference,
    Scalar approxFlux
) const {
    Result result = _impl->makeResult();
    _applyForcedImpl(result, exposure, psf, center, reference, approxFlux);
    return result;
}

void CModelAlgorithm::writeResultToRecord(
    Result const & result,
    afw::table::BaseRecord & record
) const {
    if (!_impl->keys) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Algorithm was not initialized with a schema; cannot copy to record"
        );
    }
    _impl->keys->copyResultToRecord(result, record);
}

void CModelAlgorithm::_applyForcedImpl(
    Result & result,
    afw::image::Exposure<Pixel> const & exposure,
    shapelet::MultiShapeletFunction const & psf,
    afw::geom::Point2D const & center,
    CModelResult const & reference,
    Scalar approxFlux
) const {

    if (reference.getFlag(CModelResult::FAILED)) {
        result.setFlag(CModelResult::BAD_REFERENCE, true);
        result.setFlag(CModelResult::FAILED, true);
    }

    // n.b. we're using the fit region from the reference without transforming
    // it to the forced photometry coordinate system.  That should be fine on coadds,
    // but not when doing forced photometry on individual visits.
    // We also use the final fit region from the reference here, even for the initial
    // fit, and then do not update it.  We expect this to be better than the initial fit
    // region, even though it makes the initial fit regions less consistent between
    // regular and forced measurement.
    PixelFitRegion region(getControl().region, reference.finalFitRegion);
    region.applyMask(*exposure.getMaskedImage().getMask(), center);
    result.setFlag(CModelResult::REGION_MAX_AREA, region.maxArea);
    result.setFlag(CModelResult::REGION_MAX_BAD_PIXEL_FRACTION, region.maxBadPixelFraction);
    if (!region.footprint) return;

    // Negative approxFlux means we should come up with an estimate ourselves.
    // This is only used to avoid scaling problems in the optimizer, so it doesn't have to be very good.
    if (!(approxFlux > 0.0)) {
        approxFlux = computeFluxInFootprint(*exposure.getMaskedImage().getImage(), *region.footprint);
        if (!(approxFlux > 0.0)) {
            // This is only be possible if the object has all data pixels set to zero or
            // if there are unmasked NaNs in the fit region.
            result.initial.setFlag(CModelStageResult::NUMERIC_ERROR, true);
            result.initial.setFlag(CModelStageResult::FAILED, true);
            result.exp.setFlag(CModelStageResult::NUMERIC_ERROR, true);
            result.exp.setFlag(CModelStageResult::FAILED, true);
            result.dev.setFlag(CModelStageResult::NUMERIC_ERROR, true);
            result.dev.setFlag(CModelStageResult::FAILED, true);
            result.setFlag(CModelResult::FAILED, true);
            return;
        }
    }

    // Set up coordinate systems and empty parameter vectors
    CModelStageData initialData(exposure, approxFlux, center, psf, *_impl->initial.model);
    result.fitSysToMeasSys = initialData.fitSysToMeasSys;

    // Initialize the parameter vectors from the reference values.  Because these are
    // in fitSys units, we don't need to transform them, as fitSys (or at least its
    // Wcs) should be the same in both forced mode and non-forced mode.
    initialData.nonlinear.deep() = reference.initial.nonlinear;
    initialData.fixed.deep() = reference.initial.fixed;

    // Do the initial fit (amplitudes only)
    if (!reference.initial.getFlag(CModelStageResult::FAILED)) {
        _impl->initial.fitLinear(getControl().initial, result.initial, initialData,
                                 exposure, *region.footprint);
    } else {
        result.initial.setFlag(CModelStageResult::BAD_REFERENCE, true);
        result.initial.setFlag(CModelStageResult::FAILED, true);
    }

    // Do the exponential fit (amplitudes only)
    CModelStageData expData = initialData.changeModel(*_impl->exp.model);
    if (!reference.exp.getFlag(CModelStageResult::FAILED)) {
        expData.nonlinear.deep() = reference.exp.nonlinear;
        expData.fixed.deep() = reference.exp.fixed;
        _impl->exp.fitLinear(getControl().exp, result.exp, expData, exposure, *region.footprint);
    } else {
        result.exp.setFlag(CModelStageResult::BAD_REFERENCE, true);
        result.exp.setFlag(CModelStageResult::FAILED, true);
    }

    // Do the de Vaucouleur fit (amplitudes only)
    CModelStageData devData = initialData.changeModel(*_impl->dev.model);
    if (!reference.dev.getFlag(CModelStageResult::FAILED)) {
        devData.nonlinear.deep() = reference.dev.nonlinear;
        devData.fixed.deep() = reference.dev.fixed;
        _impl->dev.fitLinear(getControl().dev, result.dev, devData, exposure, *region.footprint);
    } else {
        result.dev.setFlag(CModelStageResult::BAD_REFERENCE, true);
        result.dev.setFlag(CModelStageResult::FAILED, true);
    }

    if (result.exp.getFlag(CModelStageResult::FAILED) ||result.dev.getFlag(CModelStageResult::FAILED))
        return;

    // Do the linear combination fit
    try {
        _impl->fitLinear(getControl(), result, expData, devData, exposure, *region.footprint);
    } catch (...) {
        result.setFlag(CModelResult::FAILED, true);
        throw;
    }
}

template <typename PixelT>
shapelet::MultiShapeletFunction CModelAlgorithm::_processInputs(
    afw::table::SourceRecord & source,
    afw::image::Exposure<PixelT> const & exposure
) const {
    // Set all failure flags so that's the result if we throw.
    source.set(_impl->keys->flags[Result::FAILED], true);
    source.set(_impl->keys->initial.flags[CModelStageResult::FAILED], true);
    source.set(_impl->keys->exp.flags[CModelStageResult::FAILED], true);
    source.set(_impl->keys->dev.flags[CModelStageResult::FAILED], true);
    if (!_impl->keys) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Algorithm was not initialized with a schema; cannot run in plugin mode"
        );
    }
    if (!exposure.getWcs()) {
        source.set(_impl->keys->flags[Result::NO_WCS], true);
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeErrorException,
            "Exposure has no Wcs"
        );
    }
    if (!exposure.getCalib() || exposure.getCalib()->getFluxMag0().first == 0.0) {
        source.set(_impl->keys->flags[Result::NO_CALIB], true);
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeErrorException,
            "Exposure has no valid Calib"
        );
    }
    if (!exposure.getPsf()) {
        source.set(_impl->keys->flags[Result::NO_PSF], true);
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeErrorException,
            "Exposure has no Psf"
        );
    }
    if (!_impl->fitPsfCtrl) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Schema passed to constructor did not have FitPsf fields; "
            "a MultiShapeletFunction PSF must be passed to apply()."
        );
    }
    extensions::multiShapelet::FitPsfModel psfModel(*_impl->fitPsfCtrl, source);
    if (psfModel.hasFailed() || !(psfModel.ellipse.getArea() > 0.0)) {
        source.set(_impl->keys->flags[Result::NO_PSF], true);
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeErrorException,
            "Multishapelet PSF approximation failed or was not run"
        );
    }
    return psfModel.asMultiShapelet();
}

template <typename PixelT>
void CModelAlgorithm::_apply(
    afw::table::SourceRecord & source,
    afw::image::Exposure<PixelT> const & exposure,
    afw::geom::Point2D const & center
) const {
    Result result = _impl->makeResult();
    // Record the center we used in the fit
    source.set(_impl->keys->center, center);
    // Read the shapelet approximation to the PSF, load/verify other inputs from the SourceRecord
    shapelet::MultiShapeletFunction psf = _processInputs(source, exposure);
    afw::geom::ellipses::Quadrupole moments;
    if (!source.getTable()->getShapeKey().isValid() ||
        (source.getTable()->getShapeFlagKey().isValid() && source.getShapeFlag())) {
        if (getControl().fallbackInitialMomentsPsfFactor > 0.0) {
            result.setFlag(Result::NO_SHAPE, true);
            moments = psf.evaluate().computeMoments().getCore();
            moments.scale(getControl().fallbackInitialMomentsPsfFactor);
        } else {
            source.set(_impl->keys->flags[Result::NO_SHAPE], true);
            throw LSST_EXCEPT(
                pex::exceptions::RuntimeErrorException,
                "Shape slot algorithm failed or was not run, and fallbackInitialMomentsPsfFactor < 0"
            );
        }
    } else {
        moments = source.getShape();
    }
    // If PsfFlux has been run, use that for approx flux; otherwise we'll compute it ourselves.
    Scalar approxFlux = -1.0;
    if (source.getTable()->getPsfFluxKey().isValid() && !source.getPsfFluxFlag()) {
        approxFlux = source.getPsfFlux();
    }
    // If KronFlux has been run, use the Kron radius to initialize the fit region.
    Scalar kronRadius = -1.0;
    if (_impl->keys->kronRadius.isValid() && source.get(_impl->keys->kronRadius) > 0) {
        kronRadius = source.get(_impl->keys->kronRadius);
    }
    try {
        _applyImpl(result, exposure, psf, center, moments, approxFlux, kronRadius,
                   source.getFootprint()->getArea());
    } catch (...) {
        _impl->keys->copyResultToRecord(result, source);
        _impl->checkFlagDetails(source);
        throw;
    }
    _impl->keys->copyResultToRecord(result, source);
    _impl->checkFlagDetails(source);
}

template <typename PixelT>
void CModelAlgorithm::_applyForced(
    afw::table::SourceRecord & source,
    afw::image::Exposure<PixelT> const & exposure,
    afw::geom::Point2D const & center,
    afw::table::SourceRecord const & reference,
    afw::geom::AffineTransform const & refToMeas
) const {
    Result result = _impl->makeResult();
    assert(source.getFootprint()->getArea());
    // Record the center we used in the fit
    source.set(_impl->keys->center, center);
    // Read the shapelet approximation to the PSF, load/verify other inputs from the SourceRecord
    shapelet::MultiShapeletFunction psf = _processInputs(source, exposure);
    if (!_impl->refKeys) { // ideally we'd do this in the ctor, but we can't so we do it on first use
        _impl->refKeys.reset(
            new CModelKeys(
                *_impl->initial.model, *_impl->exp.model, *_impl->dev.model,
                reference.getSchema(), getControl().name
            )
        );
    }
    // If PsfFlux has been run, use that for approx flux; otherwise we'll compute it ourselves.
    Scalar approxFlux = -1.0;
    if (source.getTable()->getPsfFluxKey().isValid() && !source.getPsfFluxFlag()) {
        approxFlux = source.getPsfFlux();
    }
    try {
        Result refResult = _impl->refKeys->copyRecordToResult(reference);
        _applyForcedImpl(result, exposure, psf, center, refResult, approxFlux);
    } catch (...) {
        _impl->keys->copyResultToRecord(result, source);
        _impl->checkFlagDetails(source);
        throw;
    }
    _impl->keys->copyResultToRecord(result, source);
    _impl->checkFlagDetails(source);
}

LSST_MEAS_ALGORITHM_PRIVATE_IMPLEMENTATION(CModelAlgorithm);

}}} // namespace lsst::meas::multifit
