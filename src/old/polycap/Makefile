CXX       = g++
CXXFLAGS  = -std=c++17 -O2 \
            -I./polycap-1.2/include \
            $(shell pkg-config --cflags libxrl) \
            $(shell pkg-config --cflags gsl)

LDFLAGS   = -L./polycap-1.2/build/src \
            -lpolycap \
            $(shell pkg-config --libs libxrl) \
            $(shell pkg-config --libs gsl) \
            -lm

DYLD_PATH = ./polycap-1.2/build/src:/usr/local/lib:/opt/homebrew/lib

.PHONY: all run plot clean

all: test_comparison

test_comparison: test_comparison.cpp PolyCap.hpp
	$(CXX) $(CXXFLAGS) test_comparison.cpp -o test_comparison $(LDFLAGS)

run: test_comparison
	DYLD_LIBRARY_PATH="$(DYLD_PATH):$$DYLD_LIBRARY_PATH" ./test_comparison 5000 42

plot: polycap_exit.csv polycap_hpp_exit.csv
	python3 plot_beams.py polycap_exit.csv polycap_hpp_exit.csv

clean:
	rm -f test_comparison polycap_exit.csv polycap_hpp_exit.csv \
	      beam_comparison*.png
