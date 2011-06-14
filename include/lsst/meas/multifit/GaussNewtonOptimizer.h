// -*- LSST-C++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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

#ifndef LSST_MEAS_MULTIFIT_GaussNewtonOptimizer
#define LSST_MEAS_MULTIFIT_GaussNewtonOptimizer

#include "lsst/meas/multifit/GaussianDistribution.h"
#include "lsst/meas/multifit/BaseEvaluator.h"

namespace lsst {
namespace meas {
namespace multifit {

class GaussNewtonOptimizer {
public:
    GaussNewtonOptimizer(){}

    GaussianDistribution::Ptr solve(
        BaseEvaluator::Ptr const & evaluator,
        double const fTol=1.e-8, double const gTol=1.e-8, 
        double const minStep=1.e-8, 
        int const maxIter=200, 
        double const tau=1.e-3, 
        bool const retryWithSvd=false
    );
    lsst::ndarray::Array<double const, 2, 2> getParameterPoints() const;
    bool didConverge() const {return _didConverge;}
private:
    std::list<lsst::ndarray::Array<double, 1, 1> > _parameterPoints;
    bool _didConverge;
};

}}} //end namespace lsst::meas::multifit

#endif //end #ifndef LSST_MEAS_MULTIFIT_GaussNewtonOptimizer