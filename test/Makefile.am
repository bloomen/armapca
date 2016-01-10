AM_CXXFLAGS = -I$(top_srcdir)/src/lib @AM_CXXFLAGS@
check_PROGRAMS = unittest

EXTRA_DIST =

unittest_SOURCES = \
test_pca.cpp

unittest_LDADD = $(top_builddir)/src/lib/.libs/libpca2.a -lunittest $(AM_LDFLAGS)
