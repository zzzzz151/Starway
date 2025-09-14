#pragma once

#include <array>
#include <cassert>

#include "utils.hpp"

template <typename T, std::size_t N>
struct ArrayVec {
   private:
    std::array<T, N> mArr;
    std::size_t mSize = 0;

   public:
    constexpr const T& operator[](const std::size_t i) const {
        assert(i < mSize);
        return mArr[i];
    }

    constexpr T& operator[](const std::size_t i) {
        assert(i < mSize);
        return mArr[i];
    }

    constexpr std::size_t size() const { return mSize; }

    constexpr auto begin() const { return mArr.begin(); }

    constexpr auto end() const { return mArr.begin() + static_cast<std::ptrdiff_t>(mSize); }

    constexpr auto begin() { return mArr.begin(); }

    constexpr auto end() { return mArr.begin() + static_cast<std::ptrdiff_t>(mSize); }

    constexpr void clear() { mSize = 0; }

    constexpr void pushBack(const T x) {
        assert(mSize < N);
        mArr[mSize++] = x;
    }

    constexpr void popBack() {
        assert(mSize > 0);
        mSize--;
    }

    constexpr bool contains(const T x) const {
        for (const T elem : *this) {
            if (elem == x) {
                return true;
            }
        }

        return false;
    }

};  // struct ArrayVec
