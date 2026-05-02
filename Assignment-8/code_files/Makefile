CC = mpicxx
CFLAGS = -O3 -fopenmp -Wall
LDFLAGS = -fopenmp

TARGET = interpolation_parallel
SRCS = main.cpp init.cpp utils.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) Mesh.out
