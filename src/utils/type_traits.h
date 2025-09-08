#pragma once

// 获取模板参数包的第一个类型
template<typename... Ts>
struct first_type;

template<typename T, typename... Ts>
struct first_type<T, Ts...> {
    using type = T;
};

// 简化用法
template<typename... Ts>
using first_type_t = typename first_type<Ts...>::type;

template<typename Tuple>
struct tuple_first_type;

template<typename T, typename... Ts>
struct tuple_first_type<std::tuple<T, Ts...>> {
    using type = T;
};

template<typename Tuple>
using tuple_first_type_t = typename tuple_first_type<Tuple>::type;