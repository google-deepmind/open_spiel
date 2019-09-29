#include "container_state.h"
#include <iostream>

ContainerState::ContainerState(JNIEnv *env, jobject container_state):env(env),container_state(container_state){}


Region ContainerState::Empty(){

	jclass ContainerStateClass = env->FindClass("util/state/containerState/ContainerState");
	jmethodID empty_id = env->GetMethodID(ContainerStateClass,"empty","()Lutil/Region;");
	jobject region_obj = env->CallObjectMethod(container_state,empty_id);

	return Region(env,region_obj);
}

ChunkSet ContainerState::CloneWho(){
	jclass ContainerStateClass = env->FindClass("util/state/containerState/ContainerState");
	jmethodID cloneWho_id = env->GetMethodID(ContainerStateClass,"cloneWho","()Lutil/ChunkSet;");
	jobject chunkset_obj = env->CallObjectMethod(container_state,cloneWho_id);

	return ChunkSet(env,chunkset_obj);
}

ChunkSet ContainerState::CloneWhat(){
	jclass ContainerStateClass = env->FindClass("util/state/containerState/ContainerState");
	jmethodID cloneWhat_id = env->GetMethodID(ContainerStateClass,"cloneWhat","()Lutil/ChunkSet;");
	jobject chunkset_obj = env->CallObjectMethod(container_state,cloneWhat_id);

	return ChunkSet(env,chunkset_obj);
}