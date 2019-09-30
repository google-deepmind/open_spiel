#ifndef MODE_H_
#define MODE_H_

#include "jni.h"

class Mode
{

public:
    Mode(JNIEnv *env, jobject mode);

    int NumPlayers() const;

private:

    JNIEnv *env;
    jobject mode;

};


#endif