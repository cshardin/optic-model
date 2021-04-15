#!/bin/sh
# Run this from directory with source files.  Modifies code!
# We replace "class Foo():" with "class Foo:".
# Obviously you should really use flake8 for things like this.
# Note that we need extra backslashes so that perl sees \(\).
perl -pi -e "s/\\(\\)//g if /^class /" *.py
