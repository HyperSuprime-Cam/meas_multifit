Use with Valgrind encounters a problem:

terminate called after throwing an instance of 'boost::exception_detail::clone_i
mpl<boost::exception_detail::error_info_injector<std::overflow_error> >'
  what():  Error in function boost::math::erfc_inv<e>(e, e): Overflow Error
Aborted

This is documented here: https://svn.boost.org/trac/boost/ticket/10005

The fix may be in boost 1.56.0.  If the fix is not available, it seems that
either the code needs to be compiled unoptimised or boost/math/special_functions/erf.hpp
not be included (e.g., by using <cmath> for erf and erfc, and inventing
a dummy erf_inv and erfc_inv).
