#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

template <typename T, std::size_t N, std::size_t... Ns>
struct MultiArrayImpl {
    using Type = std::array<typename MultiArrayImpl<T, Ns...>::Type, N>;
};

template <typename T, std::size_t N>
struct MultiArrayImpl<T, N> {
    using Type = std::array<T, N>;
};

template <typename T, std::size_t... Ns>
using MultiArray = typename MultiArrayImpl<T, Ns...>::Type;

constexpr i32 charToI32(const char myChar) { return myChar - '0'; }

constexpr void ltrim(std::string& s) {
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

constexpr void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); })
                .base(),
            s.end());
}

constexpr void trim(std::string& s) {
    ltrim(s);
    rtrim(s);
}

constexpr std::vector<std::string> split(const std::string& str, const char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        trim(token);

        if (!token.empty()) {
            result.push_back(token);
        }
    }

    return result;
}
