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

// template specialization
#include <iostream>
using namespace std;

// class template:
template <class T>
class mycontainer {
    T element;
public:
    mycontainer (T arg) {element=arg;}
    T increase () {return ++element;}
};

// class template specialization:
template <>
class mycontainer <char> {
    char element;
public:
    mycontainer (char arg) {element=arg;}
    char uppercase ()
    {
        if ((element>='a')&&(element<='z'))
            element+='A'-'a';
        return element;
    }
};

TEST(TestTemplate,TestTemplate_Test_Test) {
    mycontainer<int> myint (7);
    mycontainer<char> mychar ('j');
    cout << myint.increase() << endl;
    cout << mychar.uppercase() << endl;
    return 0;
}