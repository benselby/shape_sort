default:
	g++ -o shape_detect shape_detect.cpp `pkg-config opencv --libs --cflags`
	g++ -o biye Source.cpp `pkg-config opencv --libs --cflags`
	g++ -o threshold threshold.cpp `pkg-config opencv --libs --cflags`
	#g++ -o biye2 biye2.cpp `pkg-config opencv --libs --cflags`
	g++ -o shape_detect_test img_proc.cpp `pkg-config opencv --libs --cflags`

