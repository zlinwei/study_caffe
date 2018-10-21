//
// Created by linwei on 2018/10/20.
//

#include <iostream>
#include <array>
#include <gtest/gtest.h>

TEST(test_case_name, test_name) {
    GTEST_ASSERT_EQ(1, 1);
    GTEST_ASSERT_NE(1, 2);
}

TEST(TestStdArray, StdArrayCopy) {
    std::array<int, 2> a = {1, 2};
    std::array<int, 2> b = a;
    std::array<int, 2> &c = a;
    std::cout << &a << " " << &b << std::endl;
    std::cout << &a.front() << " " << &b.front() << std::endl;
    std::cout << &a.front() << " " << &c.front() << std::endl;
    GTEST_ASSERT_EQ(&a.front(), &c.front());
    GTEST_ASSERT_NE(&a.front(), &b.front());
}