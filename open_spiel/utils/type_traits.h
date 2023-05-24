
#ifndef OPEN_SPIEL_TYPE_TRAITS_H
#define OPEN_SPIEL_TYPE_TRAITS_H

#include <string>
#include <type_traits>

namespace open_spiel::internal {

/// check if type T matches any of the given types in Ts...
template <typename T, typename... Ts>
struct any_of : std::disjunction<std::is_same<T, Ts>...> {};
/// helper variable to get the contained value of the trait
template <typename T, typename... Ts>
constexpr bool any_of_v = any_of<T, Ts...>::value;

/// check if type T matches all of the given types in Ts...
template < class T, class... Ts >
struct all_of: ::std::conjunction< ::std::is_same< T, Ts >... > {};
/// helper variable to get the contained value of the trait
template < class T, class... Ts >
inline constexpr bool all_of_v = all_of< T, Ts... >::value;

/// logical xor of the conditions (using fold expressions and bitwise xor)
template <typename... Conditions>
struct logical_xor : std::integral_constant<bool, (Conditions::value ^ ...)> {};
/// helper variable to get the contained value of the trait
template <typename... Conditions>
constexpr bool logical_xor_v = logical_xor<Conditions...>::value;

/// remove const and reference (&, &&) qualifiers of the type to attain
/// the underlying type of T
template <typename T>
struct remove_cvref : std::remove_cv<std::remove_reference_t<T>> {};
/// helper delcaration to fetch the contained type of the trait
template <typename T> using remove_cvref_t = typename remove_cvref<T>::type;

template <typename T>
using in_string_family =
    any_of<remove_cvref_t<T>, std::string, std::string_view, const char *>;

} // namespace open_spiel::internal

#endif // OPEN_SPIEL_TYPE_TRAITS_H
