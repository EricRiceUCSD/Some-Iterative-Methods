// SomeIterativeMethods_funcs.cpp
// Eric Rice

#include "SomeIterativeMethods.h"

void print(vector<double> const &input)
{
	cout << "{ ";
	for (int i = 0; i < input.size() - 1; i++)
	{
		cout << input.at(i) << ", ";
	}
	cout << input.back() << " }";
}

double dot(vector<double> u, vector<double> v)
{
	// Standard Euclidean dot product

	double summ = 0.0;

	for (int i = 0; i < u.size(); i++)
	{
		summ += u.at(i) * v.at(i);
	}
	return summ;
}

vector<double> vecAdd(vector<double> u, vector<double> v)
{
	// Standard vector addition

	vector<double> result(u.size(), 0.0);

	for (int i = 0; i < u.size(); i++)
	{
		result.at(i) = u.at(i) + v.at(i);
	}
	return result;
}

vector<double> scalMult(double c, vector<double> v)
{
	// Standard scalar multiplication

	vector<double> result(v.size(), 0.0);

	for (int i = 0; i < v.size(); i++)
	{
		result.at(i) = c * v.at(i);
	}
	return result;
}

vector<double> matMul(vector<vector<double> > A, vector<double> x)
{
	// Computes the matrix product Ax

	vector<double> result(x.size(), 0.0);

	for (int i = 0; i < x.size(); i++)
	{
		result.at(i) = dot(A.at(i), x);
	}
	return result;
}

void jacobi(vector<vector<double> > A, vector<double> b, vector<double> &x)
{
	/* Computes one iteration of the Jacobi method for the system Ax = b, with
	the initial guess x.*/

	vector<double> y(b.size(), 0.0);

	for (int i = 0; i < b.size(); i++)
	{
		double sum_below_i = 0.0;
		double sum_above_i = 0.0;

		for (int j = 0; j < i; j++)
		{
			sum_below_i += A.at(i).at(j) * x.at(j);
		}
		for (int j = i + 1; j < b.size(); j++)
		{
			sum_above_i += A.at(i).at(j) * x.at(j);
		}
		y.at(i) = b.at(i) - sum_below_i - sum_above_i;
	}
	for (int i = 0; i < b.size(); i++)
	{
		x.at(i) = (1 / A.at(i).at(i)) * y.at(i);
	}
}

void gaussSeidel(vector<vector<double> > A, vector<double> b, vector<double> &x)
{
	/* Computes one iteration of the Gauss-Seidel method for the system Ax = b, with
	the initial guess x.*/

	vector<double> y(b.size(), 0.0);

	for (int i = 0; i < b.size(); i++)
	{
		double sum_above_i = 0.0;

		for (int j = i + 1; j < b.size(); j++)
		{
			sum_above_i += A.at(i).at(j) * x.at(j);
		}
		y.at(i) = b.at(i) - sum_above_i;
	}
	for (int i = 0; i < b.size(); i++)
	{
		double sum_below_i = 0.0;

		for (int j = 0; j < i; j++)
		{
			sum_below_i += A.at(i).at(j) * x.at(j);
		}
		x.at(i) = (1 / A.at(i).at(i)) * (y.at(i) - sum_below_i);
	}
}

void steepestDescent(vector<vector<double> > A, vector<double> b, vector<double> &x)
{
	/* Computes one iteration of the steepest descent method for the system Ax = b,
	with the initial guess x.*/

	vector<double> r = vecAdd(b, scalMult(-1, matMul(A, x)));
	x = vecAdd(x, scalMult(dot(r, r) / dot(matMul(A, r), r), r));
}

void preconSteepDesc(vector<vector<double> > A, vector<double> b,
	vector<double> &x, vector<vector<double> > Pinv)
{
	/* Computes one iteration of the preconditioned steepest descent method for the system Ax = b,
	with the initial guess x and inverse of preconditioner Pinv.*/

	vector<double> r = vecAdd(b, scalMult(-1, matMul(A, x)));
	vector<double> z = matMul(Pinv, r);
	x = vecAdd(x, scalMult(dot(r, z) / dot(matMul(A, z), z), z));
}