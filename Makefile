
all:
	g++ -g -fcilkplus -O3 -march=native -o rec_graph_bisect.x main.cpp 
