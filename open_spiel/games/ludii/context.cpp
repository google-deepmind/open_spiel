#include "context.h"
#include <iostream>
#include "game.h"


Context::Context(JNIEnv *env,Game game, Trial trial):env(env){

	//create context object using existing game object and trial object
	jclass context_class = env->FindClass("util/Context");
	jmethodID context_const_id = env->GetMethodID(context_class,"<init>","(Lgame/Game;Lutil/Trial;)V");
	jobject context_obj  = env->NewObject(context_class,context_const_id,game.GetObj(),trial.GetObj());

	context = context_obj;
}

jobject Context::GetObj(){
	return context;
}