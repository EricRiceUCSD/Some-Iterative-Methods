// main.cpp
// Eric Rice

#include "SomeIterativeMethods.h"

vector<vector<double> > A;
vector<double> b;
vector<double> x;
vector<double> xstar;
vector<double> error;
vector<vector<double> > Pinv;

int main()
{
	A = { {  4.0,  1.0,  1.0,  0.0 },
		  { -1.0,  2.0, -1.0,  0.0 },
		  {  0.0,  1.0, -2.0,  0.0 },
		  {  2.0,  0.0,  0.0,  4.0 } };
	b = { -4.0, 1.0, -5.0, -8.0 };
	xstar = { -2.0, 1.0, 3.0, -1.0 };

	x = { 0.0, 0.0, 0.0, 0.0 };
	cout << "Jacobi Calculation for (1):\n" << endl;
	for (int i = 1; i <= 5; i++)
	{
		jacobi(A, b, x);
		cout << "x^(" << i << ") = ";
		print(x);
		error = vecAdd(xstar, scalMult(-1, x));
		cout << "\n    with error ||x* - x^(" << i << ")|| = "
			 << sqrt(dot(error, error)) << "\n\n";
	}

	x = { 0.0, 0.0, 0.0, 0.0 };
	cout << "\nGauss-Seidel Calculation for (1):\n" << endl;
	for (int i = 1; i <= 5; i++)
	{
		gaussSeidel(A, b, x);
		cout << "x^(" << i << ") = ";
		print(x);
		error = vecAdd(xstar, scalMult(-1, x));
		cout << "\n    with error ||x* - x^(" << i << ")|| = "
			 << sqrt(dot(error, error)) << "\n\n";
	}

	A = { {  1.0, -1.0, -1.0, -1.0 },
		  { -1.0,  2.0,  2.0,  2.0 },
		  { -1.0,  2.0,  3.0,  1.0 },
		  { -1.0,  2.0,  1.0,  4.0 } };
	b = { -1.0, 1.0, 6.0, -7.0 };
	xstar = { -1.0, 1.0, 2.0, -3.0 };

	x = { 0.0, 0.0, 0.0, 0.0 };
	cout << "\nSteepest Descent Calculation for (2):\n" << endl;
	for (int i = 1; i <= 2; i++)
	{
		steepestDescent(A, b, x);
		cout << "x^(" << i << ") = ";
		print(x);
		error = vecAdd(xstar, scalMult(-1, x));
		cout << "\n    with error ||x* - x^(" << i << ")|| = "
			 << sqrt(dot(error, error)) << "\n\n";
	}

	A = { { 4.0,  1.0,  2.0 },
		  { 1.0,  9.0,  1.0 },
		  { 2.0,  1.0, 16.0 } };
	b = { 0.0, 18.0, 16.0 };
	xstar = { -1.0, 2.0, 1.0 };

	x = { 0.0, 0.0, 0.0 };
	cout << "\nSteepest Descent Calculation for (3):\n" << endl;
	for (int i = 1; i <= 3; i++)
	{
		steepestDescent(A, b, x);
		cout << "x^(" << i << ") = ";
		print(x);
		error = vecAdd(xstar, scalMult(-1, x));
		cout << "\n    with error ||x* - x^(" << i << ")|| = "
			 << sqrt(dot(error, error)) << "\n\n";
	}

	Pinv = { { 1.0 / 4.0,       0.0,        0.0 },
			 {       0.0, 1.0 / 9.0,        0.0 },
			 {       0.0,       0.0, 1.0 / 16.0 } };

	x = { 0.0, 0.0, 0.0 };
	cout << "\n(Jacobi) Preconditioned Steepest Descent Calculation for (3):\n" << endl;
	for (int i = 1; i <= 3; i++)
	{
		preconSteepDesc(A, b, x, Pinv);
		cout << "x^(" << i << ") = ";
		print(x);
		error = vecAdd(xstar, scalMult(-1, x));
		cout << "\n    with error ||x* - x^(" << i << ")|| = "
			<< sqrt(dot(error, error)) << "\n\n";
	}

	cin.get();
	return 0;
}