default:
	g++ -o shape_detect shape_detect.cpp `pkg-config opencv --libs --cflags`
	g++ -o threshold threshold.cpp `pkg-config opencv --libs --cflags`
