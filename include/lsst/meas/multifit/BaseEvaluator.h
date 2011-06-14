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

#ifndef LSST_MEAS_MULTIFIT_BaseEvaluator
#define LSST_MEAS_MULTIFIT_BaseEvaluator

#include "lsst/ndarray.h"
#include "lsst/meas/multifit/constants.h"
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

namespace lsst { namespace meas { namespace multifit {

class Evaluation;

/**
 *  @brief An abstract base class that combines a data vector and model(s),
 *         organizing the model as a parameterized matrix multiplied by a vector
 *         of linear coefficients.
 *
 *  BaseEvaluator subclasses should be immutable.
 */
class BaseEvaluator {
public:

    typedef boost::shared_ptr<BaseEvaluator> Ptr;

    /// @brief Size of data vector (number of rows of matrix).
    int getDataSize() const { return _dataVector.getSize<0>(); }

    /// @brief Size of coefficient vector (number of colums of matrix).
    int getCoefficientSize() const { return _coefficientSize; }

    /// @brief Number of parameters.
    int getParameterSize() const { return _parameterSize; }

    /// @brief Number of coefficient inequality constraints.
    int getConstraintSize() const { return _constraintVector.getSize<0>(); }

    /**
     *  @brief Clip the given parameter vector to the valid range and return a penalty
     *         that scales with how far the parameter vector was beyond the constraints.
     */
    virtual double clipToBounds(lsst::ndarray::Array<double,1,1> const & parameters) const = 0;

    /**
     *  @brief Data vector.
     *
     *  If the data vector is weighted (divided by sigma) the evaluted model matrix should be as well.
     */
    lsst::ndarray::Array<Pixel const,1,1> getDataVector() const { return _dataVector; }

    /**
     *  @brief Inequality constraint matrix.
     *
     *  This is the matrix @f$A@f$ in the constraint @f$Ax \ge b@f$.
     */
    lsst::ndarray::Array<Pixel const,2,2> getConstraintMatrix() const { return _constraintMatrix; }

    /**
     *  @brief Inequality constraint matrix.
     *
     *  This is the vector @f$b@f$ in the constraint @f$Ax \ge b@f$.
     */
    lsst::ndarray::Array<Pixel const,1,1> getConstraintVector() const { return _constraintVector; }

    /**
     *  @brief Evaluate the matrix with the given parameters.
     *
     *  @param[out] matrix  An array to fill with shape (getDataSize(), getCoefficientSize()).
     *  @param[in]  param   An array of parameters with size getParameterSize().
     *
     *  If the data vector is weighted, the output matrix should be as well (each row should be divided
     *  by the corresponding pixel sigma value).
     */
    void evaluateModelMatrix(
        lsst::ndarray::Array<Pixel,2,2> const & matrix,
        lsst::ndarray::Array<Pixel const,1,1> const & param
    ) const;

    /**
     *  @brief Evaluate the derivative of the matrix with respect to the parameters.
     *
     *  The first dimension of the array is the parameter dimension.
     *
     *  The model matrix is guaranteed to be the result of _evaluateModelMatrix with the same
     *  parameter vector.
     */
    virtual void evaluateModelMatrixDerivative(
        ndarray::Array<Pixel,3,3> const & modelMatrixDerivative,
        ndarray::Array<Pixel const,1,1> const & param
    ) const;

    void writeInitialParameters(lsst::ndarray::Array<Pixel,1,1> const & param) const;

    virtual ~BaseEvaluator() {}

protected:

    friend class Evaluation;

    BaseEvaluator(int dataSize, int coefficientSize, int parameterSize) :
        _coefficientSize(coefficientSize),
        _parameterSize(parameterSize),
        _dataVector(ndarray::allocate(dataSize)),
        _constraintVector(),
        _constraintMatrix()
    {}

    BaseEvaluator(
        ndarray::Array<Pixel,1,1> const & data, int coefficientSize, int parameterSize
    ) :
        _coefficientSize(coefficientSize),
        _parameterSize(parameterSize),
        _dataVector(data),
        _constraintVector(),
        _constraintMatrix()
    {}

    BaseEvaluator(BaseEvaluator const & other) :
        _coefficientSize(other._coefficientSize), 
        _parameterSize(other._parameterSize), 
        _dataVector(other._dataVector),
        _constraintVector(other._constraintVector),
        _constraintMatrix(other._constraintMatrix)
    {}

    virtual void _evaluateModelMatrix(
        ndarray::Array<Pixel,2,2> const & matrix,
        ndarray::Array<Pixel const,1,1> const & param
    ) const = 0;

    virtual void _evaluateModelMatrixDerivative(
        ndarray::Array<Pixel,3,3> const & modelMatrixDerivative,
        ndarray::Array<Pixel const,2,2> const & modelMatrix,
        ndarray::Array<Pixel const,1,1> const & param
    ) const;

    virtual void _writeInitialParameters(ndarray::Array<Pixel,1,1> const & param) const = 0;

    int const _coefficientSize;
    int const _parameterSize;
    ndarray::Array<Pixel,1,1> _dataVector;
    ndarray::Array<Pixel const,1,1> _constraintVector;
    ndarray::Array<Pixel const,2,2> _constraintMatrix;

private:
    void operator=(BaseEvaluator const &) {}
};

}}} // namespace lsst::meas::multifit

#endif // !LSST_MEAS_MULTIFIT_BaseEvaluator