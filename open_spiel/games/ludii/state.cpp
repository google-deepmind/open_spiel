#include "state.h"


State::State(JNIEnv *env, jobject state): env(env), state(state) {}


std::vector<ContainerState> State::ContainerStates() const
{

    std::vector<ContainerState> containerStateVector;

    jclass stateClass = env->FindClass("util/state/State");
    jmethodID containerStates_id = env->GetMethodID(stateClass, "containerStates", "()[Lutil/state/containerState/ContainerState;");
    jobjectArray containerStateArray = (jobjectArray) env->CallObjectMethod(state, containerStates_id);
    int containerStateCount = env->GetArrayLength(containerStateArray);

    for (int i = 0; i < containerStateCount; i++)
    {
        jobject containerStateObj = env->GetObjectArrayElement(containerStateArray, i);
        containerStateVector.push_back(ContainerState(env, containerStateObj));
    }

    return containerStateVector;
}


int State::Mover() const
{
    jclass stateClass = env->FindClass("util/state/State");
    jmethodID mover_id = env->GetMethodID(stateClass, "mover", "()I");

    return (int) env->CallIntMethod(state, mover_id);
}
