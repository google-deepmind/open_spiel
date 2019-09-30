#include "jni_utils.h"
#include <iostream>
#include <string>
#include <cstring>

JNIUtils::JNIUtils(std::string jar_location)
{
    InitJVM(jar_location);
}

JNIUtils::~JNIUtils()
{
    CloseJVM();
}

JNIEnv *JNIUtils::GetEnv() const
{
    return env;
}

void JNIUtils::InitJVM(std::string jar_location)
{

    std::cout << "intializing JVM" << std::endl;
#ifdef JNI_VERSION_1_2
    JavaVMInitArgs vm_args;
    JavaVMOption options[1];
    char javaClassPath[] = "-Djava.class.path=";
    char charJarLocation[1024];
    strcpy(charJarLocation, jar_location.c_str());
    std::strcat(javaClassPath, charJarLocation);
    options[0].optionString = javaClassPath;
    vm_args.version = 0x00010002;
    vm_args.options = options;
    vm_args.nOptions = 1;
    vm_args.ignoreUnrecognized = JNI_TRUE;
    /* Create the Java VM */
    res = JNI_CreateJavaVM(&jvm, (void **)&env, &vm_args);
#else
    JDK1_1InitArgs vm_args;
    char classpath[1024];
    vm_args.version = 0x00010001;
    JNI_GetDefaultJavaVMInitArgs(&vm_args);
    /* Append jar location to the default system class path */
    sprintf(classpath, "%s%c%s", vm_args.classpath, ';', jar_location);
    vm_args.classpath = classpath;
    /* Create the Java VM */
    res = JNI_CreateJavaVM(&jvm, &env, &vm_args);
#endif /* JNI_VERSION_1_2 */

}

void JNIUtils::CloseJVM()
{
    std::cout << "destroying JVM" << std::endl;
    jvm->DestroyJavaVM();
}

