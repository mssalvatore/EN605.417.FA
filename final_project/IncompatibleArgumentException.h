#include <exception>
#include <string>

class IncompatibleArgumentException: public std::exception
{
    public:
        IncompatibleArgumentException(std::string msg);
        virtual ~IncompatibleArgumentException() throw();

        virtual const char* what() const throw();

    protected:
        std::string msg;
};

