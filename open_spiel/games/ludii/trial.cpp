#include "trial.h"
#include <iostream>

Trial::Trial(JNIEnv * env, Game game):env(env){

	//create trial object using existing game object
	jclass trial_class = env->FindClass("util/Trial");
	jmethodID trial_const_id = env->GetMethodID(trial_class,"<init>","(Lgame/Game;)V");
	jobject trial_obj  = env->NewObject(trial_class,trial_const_id,game.getGameObj());

	trial = trial_obj;
}


jobject Trial::getTrialObj(){
	return trial;
}