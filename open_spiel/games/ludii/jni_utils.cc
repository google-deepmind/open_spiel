// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/ludii/jni_utils.h"

#include <cstring>
#include <iostream>
#include <string>

namespace open_spiel {
namespace ludii {

JNIUtils::JNIUtils(std::string jar_location) { InitJVM(jar_location); }

JNIUtils::~JNIUtils() { CloseJVM(); }

JNIEnv *JNIUtils::GetEnv() const { return env; }

void JNIUtils::InitJVM(std::string jar_location) {
  std::cout << "intializing JVM" << std::endl;
#ifdef JNI_VERSION_1_2
  JavaVMInitArgs vm_args;
  JavaVMOption options[1];
  std::string java_classpath = "-Djava.class.path=" + jar_location;
  char *c_classpath = strdup(java_classpath.c_str());
  options[0].optionString = c_classpath;
  vm_args.version = 0x00010002;
  vm_args.options = options;
  vm_args.nOptions = 1;
  vm_args.ignoreUnrecognized = JNI_TRUE;
  /* Create the Java VM */
  res = JNI_CreateJavaVM(&jvm, (void **)&env, &vm_args);
  free(c_classpath);
#else
  JDK1_1InitArgs vm_args;
  std::string classpath = vm_args.classpath + ";" + jar_location;
  char* c_classpath = strdup(java_classpath.c_str());
  vm_args.version = 0x00010001;
  JNI_GetDefaultJavaVMInitArgs(&vm_args);
  /* Append jar location to the default system class path */
  vm_args.classpath = c_classpath;
  /* Create the Java VM */
  res = JNI_CreateJavaVM(&jvm, &env, &vm_args);
  free(c_classpath);
#endif /* JNI_VERSION_1_2 */
}

void JNIUtils::CloseJVM() {
  std::cout << "destroying JVM" << std::endl;
  jvm->DestroyJavaVM();
}

}  // namespace ludii
}  // namespace open_spiel
