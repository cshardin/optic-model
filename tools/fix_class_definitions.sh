#!/bin/sh
# Run this from directory with source files.  Modifies code!
# We replace "class Foo():" with "class Foo:".
# Extra backslashes are because there's also bash...
perl -pi -e "s/\\(\\)//g if /^class /" *.py
