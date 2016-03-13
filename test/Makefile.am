AM_CXXFLAGS = -I$(top_srcdir)/src/lib @AM_CXXFLAGS@
check_PROGRAMS = unittest

EXTRA_DIST =

unittest_SOURCES = \
test_pca.cpp \
test_covariance_matrix.cpp \
test_shuffled_matrix.cpp

unittest_LDADD = $(top_builddir)/src/lib/.libs/libarmapca.a -lunittest $(AM_LDFLAGS)
