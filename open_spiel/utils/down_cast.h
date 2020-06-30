// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_UTILS_DOWN_CAST_H_
#define OPEN_SPIEL_UTILS_DOWN_CAST_H_

#include <memory>
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Use implicit_cast as a safe version of static_cast or const_cast
// for upcasting in the type hierarchy (i.e. casting a pointer to Foo
// to a pointer to SuperclassOfFoo or casting a pointer to Foo to
// a const pointer to Foo).
// When you use implicit_cast, the compiler checks that the cast is safe.
// Such explicit implicit_casts are necessary in surprisingly many
// situations where C++ demands an exact type match instead of an
// argument type convertible to a target type.
//
// The From type can be inferred, so the preferred syntax for using
// implicit_cast is the same as for static_cast etc.:
//
//   implicit_cast<ToType>(expr)
//
// implicit_cast would have been part of the C++ standard library,
// but the proposal was submitted too late.  It will probably make
// its way into the language in the future.
template<typename To, typename From>
inline To implicit_cast(From const& f) {
  return f;
}

// When you upcast (that is, cast a pointer from type Foo to type
// SuperclassOfFoo), it's fine to use implicit_cast<>, since upcasts
// always succeed.  When you downcast (that is, cast a pointer from
// type Foo to type SubclassOfFoo), static_cast<> isn't safe, because
// how do you know the pointer is really of type SubclassOfFoo?  It
// could be a bare Foo, or of type DifferentSubclassOfFoo.  Thus,
// when you downcast, you should use macro down_cast<>. In debug mode,
// we use dynamic_cast<> to double-check the downcast is legal (we die
// if it's not).  In normal mode, we do the efficient static_cast<>
// instead.  Thus, it's important to test in debug mode to make sure
// the cast is legal!
//    This is the only place in the code we should use dynamic_cast<>.
// In particular, you SHOULDN'T be using dynamic_cast<> in order to
// do RTTI (eg code like this:
//    if (dynamic_cast<Subclass1>(foo)) HandleASubclass1Object(foo);
//    if (dynamic_cast<Subclass2>(foo)) HandleASubclass2Object(foo);
// You should design the code some other way so you do not need to
// do this.


// use like this: std::shared_ptr<T> bar = down_cast<T>(foo);
template<typename To, typename From>
inline std::shared_ptr<To> down_cast(std::shared_ptr<From>& f) {
  // Ensures that To is a sub-type of From *.  This test is here only
  // for compile-time type checking, and has no overhead in an
  // optimized build at run-time, as it will be optimized away
  // completely.
  if (false) {
    implicit_cast<From*, To*>(0);
  }

#if !defined(NDEBUG)
  // RTTI: debug mode only!
  assert(std::dynamic_pointer_cast<To>(f) != nullptr);
#endif
  return std::static_pointer_cast<To>(f);
}

// use like this if you have a function like the lambda:
// auto MakeYButLookLikeX =
//   []() -> std::shared_ptr<X> {  return std::make_shared<Y>(); };
// std::shared_ptr<Y> y = open_spiel::down_cast<Y>(MakeYButLookLikeX());
template<typename To, typename From>
inline std::shared_ptr<To> down_cast(std::shared_ptr<From>&& f) {
  // Ensures that To is a sub-type of From *.  This test is here only
  // for compile-time type checking, and has no overhead in an
  // optimized build at run-time, as it will be optimized away
  // completely.
  if (false) {
    implicit_cast<From*, To*>(0);
  }

#if !defined(NDEBUG)
  // RTTI: debug mode only!
  assert(std::dynamic_pointer_cast<To>(f) != nullptr);
#endif
  return std::static_pointer_cast<To>(f);
}

// We need to move the object, because it's a unique_ptr!
// use like this:   auto bar = down_cast<T>(std::move(foo));
template <typename To, typename From, typename Deleter>
std::unique_ptr<To, Deleter> down_cast(std::unique_ptr<From, Deleter>&& p) {
  // Ensures that To is a sub-type of From.  This test is here only
  // for compile-time type checking, and has no overhead in an
  // optimized build at run-time, as it will be optimized away
  // completely.
  if (false) {
    implicit_cast<From*, To*>(0);
  }

  if (To* cast = dynamic_cast<To*>(p.get())) {
    std::unique_ptr<To, Deleter> result(cast, std::move(p.get_deleter()));
    p.release();
    return result;
  }
  SpielFatalError("Could not down_cast the unique_ptr");
}

// use like this: down_cast<T*>(foo);
template<typename To, typename From>
inline To down_cast(From* f) {
  // Ensures that To is a sub-type of From *.  This test is here only
  // for compile-time type checking, and has no overhead in an
  // optimized build at run-time, as it will be optimized away
  // completely.
  if (false) {
    implicit_cast<From*, To>(0);
  }

#if !defined(NDEBUG)
  // RTTI: debug mode only!
  assert(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif
  return static_cast<To>(f);
}

template<typename To, typename From>
// use like this: down_cast<T&>(foo);
inline To down_cast(From& f) {
  typedef typename std::remove_reference<To>::type* ToAsPointer;
  // Ensures that To is a sub-type of From *.  This test is here only
  // for compile-time type checking, and has no overhead in an
  // optimized build at run-time, as it will be optimized away
  // completely.
  if (false) {
    implicit_cast<From*, ToAsPointer>(0);
  }

#if !defined(NDEBUG)
  // RTTI: debug mode only!
  assert(dynamic_cast<ToAsPointer>(&f) != nullptr);
#endif
  return *static_cast<ToAsPointer>(&f);
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_DOWN_CAST_H_
