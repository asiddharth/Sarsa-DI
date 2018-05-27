
#CC = condor_compile /lusr/opt/gcc-3.4.0/bin/g++
#CC = condor_compile /lusr/opt/gcc-3.3.4/bin/g++
CC = g++
COMPILE = $(CC) $(INCLUDES) ${FLAGS} $(LIBS)

SRC_DIR = .
DST_DIR = .

TARGET = experiment

#Includes
INCLUDES= -I/lusr/gnu/include

#Libs
LIBS = -L/lusr/gnu/lib 

#Flags
OSFLAGS = -Dlinux
CFLAGS = -O3 -Wall -Wno-write-strings
FLAGS = ${OSFLAGS} ${CFLAGS}

#Linker flags
LDFLAGS = -lgsl -lgslcblas

#Sources
SRCS =	mdp.cpp \
	randomagent.cpp \
	sarsalambdaagent.cpp \
	expsarsalambdaagent.cpp \
	greedygqlambdaagent.cpp \
	qlearningagent.cpp \
	crossentropyagent.cpp \
	cmaes.cpp \
	cmaesagent.cpp \
	geneticalgorithmagent.cpp \
	rwgagent.cpp \
	agent.cpp \
	main.cpp \
	transferagent.cpp \
	tdlambdanstepagent.cpp \
	tdnstep.cpp \
	sarsalambdatruncagent.cpp

#Objects
OBJS = $(SRCS:.cpp=.o)

#Rules 	

all:: 	$(TARGET) 

.cpp.o:
	$(COMPILE) -c -o $@ $(@F:%.o=%.cpp)

experiment: $(OBJS)
	$(COMPILE) -o $@ $(OBJS) ${LDFLAGS}

clean:
	rm -f $(TARGET) $(OBJS) *~

