#include <iostream>
#include <armadillo>
#include <boost/lambda/lambda.hpp>


using namespace std;
using namespace arma;

int main(int argc, char **argv){
	mat A = randu<mat>(4,5);
	mat B = randu<mat>(4,5);

	cout << A * B.t()<<endl;
	using namespace boost::lambda;
    typedef std::istream_iterator<int> in;

    std::for_each(
        in(std::cin), in(), std::cout << (_1 * 3) << " " );
	return 0;
}