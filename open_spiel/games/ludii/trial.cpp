#include "trial.h"

Trial::Trial(JNIEnv *env, Game game): env(env)
{

    jclass trial_class = env->FindClass("util/Trial");
    jmethodID trial_const_id = env->GetMethodID(trial_class, "<init>", "(Lgame/Game;)V");
    jobject trial_obj  = env->NewObject(trial_class, trial_const_id, game.GetObj());

    trial = trial_obj;
}


jobject Trial::GetObj() const
{
    return trial;
}

State Trial::GetState() const
{

    jclass trial_class = env->FindClass("util/Trial");
    jmethodID state_id = env->GetMethodID(trial_class, "state", "()Lutil/state/State;");
    jobject state_obj = env->CallObjectMethod(trial, state_id);

    return State(env, state_obj);
}


bool Trial::Over() const
{

    jclass trial_class = env->FindClass("util/Trial");
    jmethodID over_id = env->GetMethodID(trial_class, "over", "()Z");

    return (bool) env->CallObjectMethod(trial, over_id);
}