
#ifndef OPEN_SPIEL_TYPE_TRAITS_H
#define OPEN_SPIEL_TYPE_TRAITS_H

#include <string>
#include <type_traits>

namespace open_spiel::internal {

/// is_specialization checks whether T is a specialized template class of
/// 'Template' This has the limitation of not working with non-type parameters,
/// i.e. templates such as std::array will not be compatible with this type
/// Usage:
///     bool is_vector = is_specialization_v< std::vector< int >, std::vector>;
template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

template < class T, template < class... > class Template >
constexpr bool is_specialization_v = is_specialization< T, Template >::value;

/// logical xor of the conditions (using fold expressions and bitwise xor)
template <typename... Conditions>
struct logical_xor : std::integral_constant<bool, (Conditions::value ^ ...)> {};
/// helper variable to get the contained value of the trait
template <typename... Conditions>
constexpr bool logical_xor_v = logical_xor<Conditions...>::value;

/// logical and of the conditions (merely aliased)
template <typename... Conditions>
using logical_and = std::conjunction<Conditions...>;
/// helper variable to get the contained value of the trait
template <typename... Conditions>
constexpr bool logical_and_v = logical_and<Conditions...>::value;

/// logical or of the conditions (merely aliased)
template <typename... Conditions>
using logical_or = std::disjunction<Conditions...>;
/// helper variable to get the contained value of the trait
template <typename... Conditions>
constexpr bool logical_or_v = logical_or<Conditions...>::value;
/// check if type T matches any of the given types in Ts...

template <typename T, typename... Ts>
struct any_of : logical_or<std::is_same<T, Ts>...> {};
/// helper variable to get the contained value of the trait
template <typename T, typename... Ts>
constexpr bool any_of_v = any_of<T, Ts...>::value;

/// check if type T matches all of the given types in Ts...
template <class T, class... Ts>
struct all_of : logical_and<::std::is_same<T, Ts>...> {};
/// helper variable to get the contained value of the trait
template <class T, class... Ts>
inline constexpr bool all_of_v = all_of<T, Ts...>::value;

/// remove const and reference (&, &&) qualifiers of the type to attain
/// the underlying type of T
template <typename T>
struct remove_cvref : std::remove_cv<std::remove_reference_t<T>> {};
/// helper delcaration to fetch the contained type of the trait
template <typename T> using remove_cvref_t = typename remove_cvref<T>::type;

template <typename T>
using in_string_family = any_of<T, std::string, std::string_view, const char *>;

template <typename T>
constexpr bool in_string_family_v =
    any_of_v<T, std::string, std::string_view, const char *>;

} // namespace open_spiel::internal

#endif // OPEN_SPIEL_TYPE_TRAITS_H
