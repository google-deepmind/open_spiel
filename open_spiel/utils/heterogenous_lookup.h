
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
                std::conjunction_v<in_string_family<T1>, in_string_family<T2>>>>
  bool operator()(T1 &&t1, T2 &&t2) const {
    if constexpr (all_of_v<const char *, T1, T2>) {
      return std::strcmp(t1, t2) < 0;
    } else {
      return t1 < t2;
    }
  }
};

struct StringEq {
  using is_transparent = std::true_type;

  template <typename T1, typename T2,
            typename = std::enable_if_t<
                std::conjunction_v<in_string_family<T1>, in_string_family<T2>>>>
  bool operator()(T1 &&t1, T2 &&t2) const {
    if constexpr (all_of_v<const char *, T1, T2>) {
      return std::strcmp(t1, t2) == 0;
    } else {
      return t1 == t2;
    }
  }
};

struct StringHasher {
  using is_transparent = std::true_type;

  template <typename T, typename = std::enable_if_t<
                            any_of_v<remove_cvref_t<T>, std::string,
                                     std::string_view, const char *>>>
  size_t operator()(T &&t) const {
    return std::hash<remove_cvref_t<T>>{}(t);
  }
};

} // namespace open_spiel::internal

#endif // OPEN_SPIEL_HETEROGENOUS_LOOKUP_H
