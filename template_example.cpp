#include<iostream>
#include<string>
 
using namespace std;

// see how template function is called in main
template <typename T>
T sum(T a, T b){
    return a + b;
}
 
int main(void) {
    int a=1, b =2;
    
    double d1 = 2.2;
    double d2 = 3.3;
 
    string s1 = "Show me ";
    string s2 = "The Money 6";
    
    cout << "int 합 : " << sum<int>(a, b) << endl;
    cout << "double 합 : " << sum<double>(d1, d2) << endl;
    cout << "string 합 : " << sum<string>(s1, s2) << endl;
    
    return 0;
}