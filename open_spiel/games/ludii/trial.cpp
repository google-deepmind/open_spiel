#include "trial.h"
#include <iostream>

Trial::Trial(JNIEnv * env, Game game):env(env){

	//create trial object using existing game object
	jclass trial_class = env->FindClass("util/Trial");
	jmethodID trial_const_id = env->GetMethodID(trial_class,"<init>","(Lgame/Game;)V");
	jobject trial_obj  = env->NewObject(trial_class,trial_const_id,game.GetObj());

	trial = trial_obj;
}


jobject Trial::GetObj(){
	return trial;
}

State Trial::GetState(){

	jclass trial_class = env->FindClass("util/Trial");
	jmethodID state_id = env->GetMethodID(trial_class,"state","()Lutil/state/State;");
	jobject state_obj = env->CallObjectMethod(trial,state_id);

	return State(env,state_obj);
}


bool Trial::Over(){

	jclass trial_class = env->FindClass("util/Trial");
	jmethodID over_id = env->GetMethodID(trial_class,"over","()Z");

	return (bool) env->CallObjectMethod(trial,over_id);
}