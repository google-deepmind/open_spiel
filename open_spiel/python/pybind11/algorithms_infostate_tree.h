// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_H
#define OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_H

#include "open_spiel/python/pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"

namespace open_spiel {

void init_pyspiel_infostate_tree(::pybind11::module &m);

void init_pyspiel_infostate_node(::pybind11::module &m);

template < typename T >
void init_pyspiel_treevector_bundle(::pybind11::module &m, std::string &typestr);

template < typename Self >
void init_pyspiel_node_id(::pybind11::module &m, const std::string &class_name);

// Bind the Range class
template < class Id >
void init_pyspiel_range(::pybind11::module &m, const std::string &name);

}  // namespace open_spiel

// include the template definition file
#include "open_spiel/python/pybind11/algorithms_infostate_tree.tcc"

/// An exception wrapping a forbidden action with given reason.
class ForbiddenException: public std::exception {
  public:
   explicit ForbiddenException(const char *reason) : m_reason(reason) {}

   [[nodiscard]] const char *what() const noexcept override { return m_reason.c_str(); }

  private:
   std::string m_reason;
};

/// A smart holder that mimicks the unique pointer api, but doesn't delete the contained object.
///
/// This class is used for epxosing c++ objects with c++ maintained lifetimes on the python side
/// without running into the risk of double free.
/// While std::unique_ptr< T, py::nodelete> would fulfill the same, such a pointer is not copyable
/// and thus prohibits other bindings, e.g. bind_vector of such a pointer.
/// The MockUniquePtr is copyable, since it doesn't manage any lifetime, and can therefore be used
/// more easily.
template < typename T >
class MockUniquePtr {
  public:
   MockUniquePtr() noexcept : ptr_(nullptr) {}
   explicit MockUniquePtr(T *ptr) noexcept : ptr_(ptr) {}
   ~MockUniquePtr() = default;
   MockUniquePtr(const MockUniquePtr &other) noexcept : ptr_(other.get()) {}
   MockUniquePtr &operator=(const MockUniquePtr &other) noexcept
   {
      reset(other.get());
      return *this;
   }
   MockUniquePtr(MockUniquePtr &&other) noexcept : ptr_(other.release()) {}
   MockUniquePtr &operator=(MockUniquePtr &&other) noexcept
   {
      reset(other.release());
      return *this;
   }

   [[nodiscard]] T *get() const noexcept { return ptr_; }
   T *release() noexcept
   {
      T *ptr = ptr_;
      ptr_ = nullptr;
      return ptr;
   }
   void reset(T *ptr = nullptr) noexcept { ptr_ = ptr; }

   T &operator*() const noexcept { return *ptr_; }
   T *operator->() const noexcept { return ptr_; }
   explicit operator bool() const noexcept { return ptr_ != nullptr; }

  private:
   T *ptr_;
};

PYBIND11_DECLARE_HOLDER_TYPE(T, MockUniquePtr< T >, true);

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_H
