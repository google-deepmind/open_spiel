
#ifndef OPEN_SPIEL_HETEROGENOUS_LOOKUP_H
#define OPEN_SPIEL_HETEROGENOUS_LOOKUP_H

#include "open_spiel/utils/type_traits.h"
#include <cstring>
#include <string>

namespace open_spiel::internal {

struct StringCmp {
  using is_transparent = std::true_type;

  template <typename T1, typename T2,
            typename = std::enable_if_t<
                logical_and_v<in_string_family<std::decay_t<T1>>,
                              in_string_family<std::decay_t<T2>>>>>
  bool operator()(T1 &&t1, T2 &&t2) const {
    if constexpr (all_of_v<const char *, std::decay_t<T1>, std::decay_t<T2>>) {
      return std::strcmp(t1, t2) < 0;
    } else {
      return t1 < t2;
    }
  }
};

struct StringPairCmp {
  using is_transparent = std::true_type;

  // the operator() only works with std::pair types. We use static_asserts here
  // to avoid duplicating code for deudcing different ref and const qualifiers
  // of the pairs.
  template <typename P1, typename P2> bool operator()(P1 &&p1, P2 &&p2) const {
    static_assert(logical_and_v<is_specialization<std::decay_t<P1>, std::pair>,
                                is_specialization<std::decay_t<P2>, std::pair>>,
                  "Passed parameter is not a specialization of std::pair.");

    using T1 = std::decay_t<typename std::decay_t<P1>::first_type>;
    using T2 = std::decay_t<typename std::decay_t<P1>::second_type>;
    using T3 = std::decay_t<typename std::decay_t<P2>::first_type>;
    using T4 = std::decay_t<typename std::decay_t<P2>::second_type>;

    static_assert(
        logical_and_v<in_string_family<T1>, in_string_family<T2>,
                      in_string_family<T3>, in_string_family<T4>>,
        "Passed element type of pair is not part of the string family.");
    constexpr StringCmp element_cmp;
    if (element_cmp(p1.first, p2.first)) {
      return true;
    } else if (element_cmp(p2.first, p1.first)) {
      return false;
    } else {
      // first elements of both pairs are equal, compare the second.
      return element_cmp(p1.second, p2.second);
    }
  }
};

struct StringEq {
  using is_transparent = std::true_type;

  template <typename T1, typename T2,
            typename = std::enable_if_t<
                logical_and_v<in_string_family<std::decay_t<T1>>,
                              in_string_family<std::decay_t<T2>>>>>
  bool operator()(T1 &&t1, T2 &&t2) const {
    if constexpr (all_of_v<const char *, std::decay_t<T1>, std::decay_t<T2>>) {
      return std::strcmp(t1, t2) == 0;
    } else {
      return t1 == t2;
    }
  }
};

struct StringPairEq {
  using is_transparent = std::true_type;

  // the operator() only works with std::pair types. We use static_asserts here
  // to avoid duplicating code for deudcing different ref and const qualifiers
  // of the pairs.
  template <typename P1, typename P2> bool operator()(P1 &&p1, P2 &&p2) const {
    static_assert(logical_and_v<is_specialization<std::decay_t<P1>, std::pair>,
                                is_specialization<std::decay_t<P2>, std::pair>>,
                  "Passed parameter is not a specialization of std::pair.");

    using T1 = std::decay_t<typename std::decay_t<P1>::first_type>;
    using T2 = std::decay_t<typename std::decay_t<P1>::second_type>;
    using T3 = std::decay_t<typename std::decay_t<P2>::first_type>;
    using T4 = std::decay_t<typename std::decay_t<P2>::second_type>;

    static_assert(
        logical_and_v<in_string_family<T1>, in_string_family<T2>,
                      in_string_family<T3>, in_string_family<T4>>,
        "Passed element type of pair is not part of the string family.");
    constexpr StringEq element_eq;
    return element_eq(p1.first, p2.first) && element_eq(p1.second, p2.second);
  }
};

struct StringHasher {
  using is_transparent = std::true_type;

  template <typename T,
            typename = std::enable_if_t<in_string_family_v<std::decay_t<T>>>>
  size_t operator()(T &&t) const {
    return std::hash<std::string_view>{}(t);
  }
};

} // namespace open_spiel::internal

#endif // OPEN_SPIEL_HETEROGENOUS_LOOKUP_H
