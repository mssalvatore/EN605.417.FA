#include <exception>
#include <string>

class InvalidArgumentException: public std::exception
{
    public:
        InvalidArgumentException(const char * argument, std::string type);
        virtual ~InvalidArgumentException() throw();

        virtual const char* what() const throw();

    protected:
        const char * argument;
        std::string type;
};

