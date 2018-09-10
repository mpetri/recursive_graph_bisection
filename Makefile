$(CXX) = g++
all:
	$(CXX) -Wall -Wextra -g -fcilkplus -std=c++17 -O3 -march=native -o rec_graph_bisect.x main.cpp
