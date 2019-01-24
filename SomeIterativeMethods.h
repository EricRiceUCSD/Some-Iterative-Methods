// SomeIterativeMethods.h
// Eric Rice

#ifndef SomeIterativeMethods_h
#define SomeIterativeMethods_h

#include <vector>
#include <iostream>
using namespace std;

void print(vector<double> const &input);

double dot(vector<double> u, vector<double> v);

vector<double> vecAdd(vector<double> u, vector<double> v);

vector<double> scalMult(double c, vector<double> v);

vector<double> matMul(vector<vector<double> > A, vector<double> x);

void jacobi(vector<vector<double> > A, vector<double> b, vector<double> &x);

void gaussSeidel(vector<vector<double> > A, vector<double> b, vector<double> &x);

void steepestDescent(vector<vector<double> > A, vector<double> b, vector<double> &x);

void preconSteepDesc(vector<vector<double> > A, vector<double> b,
	vector<double> &x, vector<vector<double> > Pinv);

#endif