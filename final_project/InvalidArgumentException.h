#include <exception>
#include <string>

class InvalidArgumentException: public std::exception
{
    public:
        InvalidArgumentException(char * argument, std::string type);
        virtual ~InvalidArgumentException() throw();

        virtual const char* what() const throw();

    protected:
        char * argument;
        std::string type;
};

