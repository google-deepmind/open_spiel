#ifndef STATE_H_
#define STATE_H_

#include "jni.h"
#include "container_state.h"
#include <vector>

class State{
	
public:

	State(JNIEnv *env, jobject state);

	std::vector<ContainerState> ContainerStates();

	int Mover();
	
private:

	JNIEnv *env;
	jobject state;

};

#endif